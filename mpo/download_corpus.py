"""Download the official fixed BCP RAG corpus, without question labels."""
import argparse
import json
from pathlib import Path


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/bcp/corpus.parquet'))
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    repo = 'Tevatron/browsecomp-plus-corpus'
    revision = HfApi().dataset_info(repo, revision=args.revision).sha
    corpus = load_dataset(repo, split='train', revision=revision).select_columns(['docid', 'text', 'url'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(str(args.output))
    args.output.with_suffix('.provenance.json').write_text(json.dumps(
        dict(dataset=repo, revision=revision, split='train', documents=len(corpus)), indent=2) + '\n')
    print(f'Saved {len(corpus)} documents to {args.output}')


if __name__ == '__main__':
    main()
