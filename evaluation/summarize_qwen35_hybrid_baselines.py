"""Validate and summarize the completed dense and nine PaLU PPL results."""
import json
import math
from pathlib import Path

import torch
import transformers

from evaluation.qwen35_hybrid_common import atomic_save, load_bank, sha256


def main():
    root = Path('results/q35_hybrid')
    names = ['dense'] + [f'palu_{family}_v{rank}' for family in ('mlrd', 'glrd2', 'glrd4') for rank in (64, 80, 96)]
    rows = []
    for name in names:
        path = root / f'{name}_ppl.json'
        result = json.loads(path.read_text())
        data = result['datasets']
        assert data['wikitext']['predicted_tokens'] == 297047 and data['wikitext']['windows'] == 146
        assert data['c4_eval']['predicted_tokens'] == 262016 and data['c4_eval']['windows'] == 128
        assert all(math.isfinite(d['ppl']) and d['ppl'] > 0 for d in data.values())
        row = {'name': name, 'wikitext_ppl': data['wikitext']['ppl'], 'c4_ppl': data['c4_eval']['ppl'],
            'ppl_file_sha256': sha256(path)}
        if name != 'dense':
            bank_path = root / 'banks' / f'{name}.pt'
            bank = load_bank(bank_path)
            row.update({'realized_v_retention': bank['realized_v_retention'], 'factor_sha256': bank['factor_sha256'],
                'bank_file_sha256': sha256(bank_path), 'rank_map': bank['rank_map']})
        rows.append(row)
        print(json.dumps(row), flush=True)
    model_files = sorted((root / 'model').glob('*.safetensors'))
    manifest = {'scope': 'dense_and_nine_palu_baselines_only_C1_pending', 'rows': rows,
        'model_revision': (root / 'model/.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0],
        'model_files_sha256': {p.name: sha256(p) for p in model_files},
        'config_sha256': sha256(root / 'model/config.json'), 'windows_sha256': sha256(root / 'data/windows.pt'),
        'torch': torch.__version__, 'transformers': transformers.__version__, 'environment': 'lowrank',
        'execution': 'local A100 subprocesses; no Slurm on this host', 'cpu_threads_per_job': 2,
        'calibration_fit': [256, 2048], 'ppl_wikitext': 'complete raw test, final short block included',
        'ppl_c4': [128, 2048], 'fisher': 'official PaLU double-shift convention, 256 matched fit windows',
        'fisher_files_sha256': {p.name: sha256(p) for p in sorted((root / 'fisher').glob('shard*.pt'))},
        'whitening_sha256': sha256(root / 'whitening.pt')}
    atomic_save(root / 'baseline_summary.json', manifest)


if __name__ == '__main__':
    main()
