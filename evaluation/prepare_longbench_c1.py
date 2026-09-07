"""Freeze a deterministic LongBench-v1 pilot, official prompts, and token inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json

TASKS = ('qasper','multifieldqa_en','hotpotqa','2wikimqa','gov_report','qmsum')


def bounded_tokens(tokens, total_limit, generation_cap):
    budget = total_limit - generation_cap
    assert budget > 0 and len(tokens) > 0
    if len(tokens) <= budget:
        return list(tokens)
    left = budget // 2
    return list(tokens[:left]) + list(tokens[-(budget-left):])


def choose_rows(rows, task, count, seed):
    assert len(rows) >= count and len({r['_id'] for r in rows}) == len(rows)
    key = lambda r: (hashlib.sha256(f"{seed}:{task}:{r['_id']}".encode()).hexdigest(),r['_id'])
    return sorted(rows,key=key)[:count]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--official-root',type=Path,default=ROOT/'external/LongBench')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--samples-per-task',type=int,default=32)
    p.add_argument('--sequence-length',type=int,default=32768)
    p.add_argument('--seed',type=int,default=73)
    args = p.parse_args()
    torch.set_num_threads(2)
    assert not (args.output_dir/'manifest.json').exists()
    official = args.official_root/'LongBench'
    prompts = json.loads((official/'config/dataset2prompt.json').read_text())
    caps = json.loads((official/'config/dataset2maxlen.json').read_text())
    revision = HfApi().dataset_info('THUDM/LongBench').sha
    archive = Path(hf_hub_download('THUDM/LongBench','data.zip',repo_type='dataset',revision=revision))
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    rows, tensors, hashes = [], {}, {}
    with zipfile.ZipFile(archive) as data:
        for task in TASKS:
            names = [n for n in data.namelist() if n == f'{task}.jsonl' or n.endswith(f'/{task}.jsonl')]
            assert len(names) == 1
            raw = data.read(names[0])
            hashes[task] = hashlib.sha256(raw).hexdigest()
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            chosen = choose_rows(records,task,args.samples_per_task,args.seed)
            for ordinal,record in enumerate(chosen):
                prompt = prompts[task].format(**record)
                token_ids = tokenizer(prompt,add_special_tokens=True)['input_ids']
                selected = bounded_tokens(token_ids,args.sequence_length,caps[task])
                index = len(rows)
                tensor = torch.tensor(selected,dtype=torch.int32)
                tensors[f'sample_{index:03d}'] = tensor
                rows.append(dict(index=index,task=task,ordinal=ordinal,source_id=record['_id'],
                    answers=record['answers'],all_classes=record['all_classes'],
                    official_length=record['length'],original_prompt_tokens=len(token_ids),
                    prompt_tokens=len(selected),truncated=len(selected)<len(token_ids),
                    maximum_tokens=caps[task],prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    input_ids_sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest()))
            print(f'prepared {task}: {len(chosen)} samples, cap={caps[task]}',flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    save_file(tensors,str(args.output_dir/'tokens.safetensors'))
    write_json(args.output_dir/'samples.json',rows)
    specification = dict(format='basisserve.longbench_c1_inputs.v1',tasks=list(TASKS),
        samples_per_task=args.samples_per_task,sequence_length=args.sequence_length,seed=args.seed,
        sample_selection='sort SHA256(seed:task:source_id), fixed before any predictions; no label/length filtering',
        dataset_repo='THUDM/LongBench',dataset_revision=revision,data_zip_sha256=sha256(archive),
        task_jsonl_sha256=hashes,official_root=str(args.official_root.resolve()),
        official_revision=subprocess.check_output(['git','-C',str(args.official_root),'rev-parse','HEAD'],text=True).strip(),
        official_sha256={name:sha256(official/name) for name in ('config/dataset2prompt.json','config/dataset2maxlen.json','eval.py','metrics.py')},
        model=str(args.model.resolve()),model_config_sha256=sha256(args.model/'config.json'),
        prompt='official task template; base model, no chat template; add_special_tokens=True',
        truncation='token-level keep prefix/suffix equally; total limit includes official task generation cap; no padding to32K; unlike official decode/re-encode truncation, retained token IDs are preserved exactly',
        source_sha256=sha256(Path(__file__)))
    write_json(args.output_dir/'manifest.json',dict(status='complete',protocol=specification,
        tokens_sha256=sha256(args.output_dir/'tokens.safetensors'),samples_sha256=sha256(args.output_dir/'samples.json')))
    print(json.dumps({'count':len(rows),'truncated':sum(r['truncated'] for r in rows),
                      'min_tokens':min(r['prompt_tokens'] for r in rows),'max_tokens':max(r['prompt_tokens'] for r in rows)}),flush=True)


if __name__ == '__main__':
    main()
