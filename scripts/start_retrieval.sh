#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$PROJECT_DIR"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/bcp}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
for path in "$DATA_DIR/corpus.parquet" "$DATA_DIR/dense_index.pkl"; do
    [[ -f "$path" ]] || { printf 'Missing retrieval input: %s\n' "$path" >&2; exit 1; }
done
exec "$PYTHON_BIN" -m retrieval.server \
    --mode dense --model "${RETRIEVER_MODEL_PATH:?Set RETRIEVER_MODEL_PATH}" \
    --device "${RAG_DEVICE:-cuda:0}" \
    --corpus_file "$DATA_DIR/corpus.parquet" \
    --dense_cache "$DATA_DIR/dense_index.pkl" \
    --host "${RAG_HOST:-0.0.0.0}" --port "${RAG_PORT:-8000}" \
    --batch_size "${RAG_BATCH_SIZE:-4}" "$@"
