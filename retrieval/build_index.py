"""Build the BCP dense index; supports single GPU and torchrun sharding."""

import argparse
import datetime
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd


def build_corpus_from_file(corpus_file: str) -> dict:
    """Load corpus from official BC-Plus parquet file (HuggingFace format)."""
    print(f"Loading corpus from file: {corpus_file} ...", flush=True)
    df = pd.read_parquet(corpus_file)
    seen = {}
    for _, row in df.iterrows():
        docid = str(row.get("docid", ""))
        text = str(row.get("text", ""))
        url = str(row.get("url", ""))
        if docid and text:
            seen[docid] = {"docid": docid, "text": text, "url": url}
    print(f"  Loaded {len(seen)} unique documents from file.", flush=True)
    return seen


def build_corpus(data_dir: str) -> dict:
    print("Loading parquet files ...", flush=True)
    dfs = []
    for split in ["train.parquet", "test.parquet"]:
        path = f"{data_dir}/{split}"
        try:
            dfs.append(pd.read_parquet(path))
        except FileNotFoundError:
            print(f"  Warning: {path} not found, skipping.", flush=True)
    if not dfs:
        raise RuntimeError(f"No parquet files found in {data_dir}")
    df = pd.concat(dfs, ignore_index=True)

    seen = {}
    for col in ["gold_docs", "negative_docs", "evidence_docs"]:
        if col not in df.columns:
            continue
        for row_docs in df[col].dropna():
            if row_docs is None:
                continue
            for doc in row_docs:
                if not isinstance(doc, dict):
                    continue
                docid = str(doc.get("docid", ""))
                if docid and docid not in seen:
                    seen[docid] = doc

    print(f"  Total unique documents: {len(seen)}", flush=True)
    return seen


def extract_title(text: str, url: str = "") -> str:
    m = re.search(r"^---\s*\ntitle:\s*(.+?)\n", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    if url:
        return url.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")
    return "Unknown"


def encode_all(texts, tokenizer, model, device, batch_size, max_length):
    import torch
    import torch.nn.functional as F

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            # Embedding extraction does not reuse autoregressive KV state.
            # Disabling it avoids repeatedly materializing a large DynamicCache.
            out = model(**enc, use_cache=False)
        # Last token pooling (Qwen3-Embedding decoder arch)
        seq_lens = enc["attention_mask"].sum(dim=1) - 1
        batch_size_actual = out.last_hidden_state.shape[0]
        emb = out.last_hidden_state[
            torch.arange(batch_size_actual, device=device), seq_lens
        ]
        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu().float().numpy())

        done = i + len(batch)
        if done % (batch_size * 5) == 0 or done == len(texts):
            print(f"  [{done}/{len(texts)}] encoded", flush=True)

    return np.vstack(all_embs)


def restore_corpus_order(embeddings, indices):
    restored = np.empty_like(embeddings)
    restored[indices] = embeddings
    return restored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/bcp")
    parser.add_argument("--corpus_file", required=True,
                        help="Path to official BC-Plus corpus parquet. If set, overrides --data_dir.")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_doc_length", type=int, default=4096)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # torchrun support: each process encodes a strided shard on its local GPU,
    # then rank 0 restores corpus order and writes the usual cache format.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        import torch.distributed as dist

        dist.init_process_group(backend="gloo", timeout=datetime.timedelta(hours=6))
        args.device = f"cuda:{local_rank}"
        print(f"[rank {rank}/{world_size}] using {args.device}", flush=True)

    # Check faiss
    try:
        import faiss
    except ImportError:
        print("ERROR: faiss not found. Install with:", flush=True)
        print("  pip install faiss-gpu   # GPU (recommended)", flush=True)
        print("  pip install faiss-cpu   # CPU fallback", flush=True)
        sys.exit(1)

    import torch
    from transformers import AutoTokenizer, AutoModel

    # Build corpus
    doc_map = build_corpus_from_file(args.corpus_file) if args.corpus_file else build_corpus(args.data_dir)
    docids = list(doc_map.keys())
    docs = [doc_map[d] for d in docids]

    contents = []
    raw_texts = []
    for doc in docs:
        title = extract_title(doc.get("text", ""), doc.get("url", ""))
        body = doc.get("text", "")
        contents.append(f"{title}\n{body}")
        raw_texts.append(body)

    # Encode documents
    # Length-bucket before batching. BrowseComp documents range from a few
    # characters to millions, so random batches waste most FLOPs on padding.
    # Assign whole sorted batches round-robin to keep ranks balanced while
    # retaining similarly sized documents within each batch.
    sorted_indices = np.argsort(
        np.fromiter((len(text) for text in raw_texts), dtype=np.int64),
        kind="stable",
    )
    batches = [
        sorted_indices[i:i + args.batch_size]
        for i in range(0, len(sorted_indices), args.batch_size)
    ]
    if len(batches) < world_size:
        raise ValueError("Need at least one corpus batch per worker")
    local_indices = np.concatenate(batches[rank::world_size])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    shard_path = f"{args.output}.rank{rank}.part.pkl"
    local_embeddings = None
    reused_shard = False
    if distributed and os.path.exists(shard_path):
        with open(shard_path, "rb") as f:
            existing_shard = pickle.load(f)
        if np.array_equal(existing_shard.get("indices"), local_indices):
            local_embeddings = existing_shard.get("embeddings")
            reused_shard = True
            print(f"[rank {rank}] Reusing completed shard {shard_path}", flush=True)

    if local_embeddings is None:
        device = torch.device(args.device)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        print(f"[rank {rank}] Loading {args.model} on {args.device} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", trust_remote_code=True)
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if device.type == "cuda" else "eager",
        )
        model = model.to(device).eval()
        print("  Model loaded.", flush=True)
        local_texts = [raw_texts[i] for i in local_indices]
        print(
            f"[rank {rank}] Encoding {len(local_texts)}/{len(raw_texts)} documents "
            f"(batch_size={args.batch_size}) ...",
            flush=True,
        )
        local_embeddings = encode_all(
            local_texts, tokenizer, model, device, args.batch_size, args.max_doc_length
        )
        print(f"[rank {rank}] Embeddings shape: {local_embeddings.shape}", flush=True)

    if distributed:
        if not reused_shard:
            with open(shard_path, "wb") as f:
                pickle.dump({"indices": local_indices, "embeddings": local_embeddings}, f)
        dist.barrier()
        if rank != 0:
            dist.destroy_process_group()
            return

        embeddings = np.empty(
            (len(raw_texts), local_embeddings.shape[1]), dtype=local_embeddings.dtype
        )
        for shard_rank in range(world_size):
            shard_path = f"{args.output}.rank{shard_rank}.part.pkl"
            with open(shard_path, "rb") as f:
                shard = pickle.load(f)
            embeddings[shard["indices"]] = shard["embeddings"]
        print(f"[rank 0] Merged embeddings shape: {embeddings.shape}", flush=True)
    else:
        embeddings = restore_corpus_order(local_embeddings, local_indices)

    # Build FAISS index
    dim = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP (dim={dim}) ...", flush=True)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  Index built: {index.ntotal} vectors", flush=True)

    # Quick sanity check
    q = embeddings[:1]
    scores, ids = index.search(q, 3)
    print(f"  Sanity check: top-3 scores for doc[0]: {scores[0].tolist()}", flush=True)

    # Save cache
    print(f"Saving to {args.output} ...", flush=True)
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({
            "docids": docids,
            "contents": contents,
            "embeddings": embeddings,
        }, f)
    if distributed:
        for shard_rank in range(world_size):
            os.remove(f"{args.output}.rank{shard_rank}.part.pkl")
        dist.destroy_process_group()
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"  Saved {size_mb:.0f} MB to {args.output}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
