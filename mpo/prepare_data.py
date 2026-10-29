"""Prepare only the exact MPO BrowseComp-Plus train/validation partition."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from mpo.prompts import build_prompt

ROOT = Path(__file__).resolve().parents[1]


def prepare_frame(frame, manifest):
    required = {'query_id', 'query', 'answer'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing question columns: {required - set(frame.columns)}')
    frame = frame.copy()
    frame['query_id'] = frame['query_id'].astype(str)
    if frame['query_id'].duplicated().any():
        raise ValueError('Duplicate query IDs in question data')
    train, validation = manifest['train'], manifest['validation']
    if len(set(train)) != len(train) or len(set(validation)) != len(validation) or set(train) & set(validation):
        raise ValueError('Split manifest contains duplicate or overlapping query IDs')
    expected = set(train) | set(validation)
    if expected != set(frame['query_id']):
        raise ValueError('Question IDs differ from the committed BCP split manifest')
    indexed = frame.set_index('query_id', drop=False)
    outputs = {}
    for split in ('train', 'validation'):
        records = []
        for index, query_id in enumerate(manifest[split]):
            row = indexed.loc[query_id]
            query, answer = str(row['query']).strip(), str(row['answer'])
            if not query or not answer.strip() or pd.isna(row['query']) or pd.isna(row['answer']):
                raise ValueError(f'Missing question or answer for {query_id}')
            records.append(dict(
                query_id=query_id, query=query, answer=answer,
                prompt=build_prompt(row, tool_style='hermes'),
                data_source='browsecomp_plus', ability='search',
                reward_model={'ground_truth': answer, 'style': 'rule'},
                extra_info={'index': index, 'query_id': query_id, 'query': query, 'split': split},
            ))
        outputs[split] = pd.DataFrame(records)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/bcp')
    parser.add_argument('--questions-file', type=Path, help='Offline official question parquet')
    parser.add_argument('--revision', default='main', help='Official dataset revision; resolved SHA is recorded')
    args = parser.parse_args()
    manifest_path = ROOT / 'data/splits/bcp_plus.json'
    manifest = json.loads(manifest_path.read_text())
    if args.questions_file:
        frame = pd.read_parquet(args.questions_file, columns=['query_id', 'query', 'answer'])
        provenance = {'input_sha256': hashlib.sha256(args.questions_file.read_bytes()).hexdigest()}
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(manifest['dataset'], revision=args.revision).sha
        dataset = load_dataset(manifest['dataset'], split=manifest['source_split'], revision=revision)
        frame = dataset.select_columns(['query_id', 'query', 'answer']).to_pandas()
        provenance = {'revision': revision}
    outputs = prepare_frame(frame, manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in outputs.items():
        frame.to_parquet(args.output_dir / f'{split}.parquet', index=False)
        print(f'{split}: {len(frame)} questions')
    provenance.update(dataset=manifest['dataset'], source_split=manifest['source_split'],
                      split_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                      sizes={split: len(frame) for split, frame in outputs.items()})
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
