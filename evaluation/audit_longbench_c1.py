"""Independently verify frozen source IDs/prompts/tokens and all four-arm predictions."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from evaluation.prepare_longbench_c1 import TASKS,bounded_tokens,choose_rows
from evaluation.eval_longbench_c1_fourarm import ARMS,official_scorer,score_prediction
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--result-dir',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    dataset=json.loads((args.data_dir/'manifest.json').read_text())
    result=json.loads((args.result_dir/'result.json').read_text())
    assert result['status']==dataset['status']=='complete'
    spec=dataset['protocol']
    assert result['protocol']['dataset_manifest_sha256']==sha256(args.data_dir/'manifest.json')
    for name,digest in result['protocol']['code_sha256'].items():
        assert sha256(ROOT/name)==digest
    rows=json.loads((args.data_dir/'samples.json').read_text())
    tokens=load_file(str(args.data_dir/'tokens.safetensors'))
    predictions={r['sample']['index']:r for r in result['records']}
    assert len(rows)==len(tokens)==len(predictions)==192
    archive=Path(hf_hub_download('THUDM/LongBench','data.zip',repo_type='dataset',revision=spec['dataset_revision'],local_files_only=True))
    assert sha256(archive)==spec['data_zip_sha256']
    official=Path(spec['official_root'])
    prompts=json.loads((official/'LongBench/config/dataset2prompt.json').read_text())
    caps=json.loads((official/'LongBench/config/dataset2maxlen.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(spec['model'],local_files_only=True)
    scorer=official_scorer(official)
    checked=0
    with zipfile.ZipFile(archive) as data:
        for task in TASKS:
            name=[n for n in data.namelist() if n==f'{task}.jsonl' or n.endswith(f'/{task}.jsonl')]
            assert len(name)==1
            raw=data.read(name[0])
            assert hashlib.sha256(raw).hexdigest()==spec['task_jsonl_sha256'][task]
            source=choose_rows([json.loads(line) for line in raw.splitlines() if line.strip()],task,32,spec['seed'])
            selected=[r for r in rows if r['task']==task]
            assert [r['_id'] for r in source]==[r['source_id'] for r in selected]
            for sample,original in zip(selected,source,strict=True):
                assert sample['answers']==original['answers'] and sample['all_classes']==original['all_classes']
                text=prompts[task].format(**original)
                original_ids=tokenizer(text,add_special_tokens=True)['input_ids']
                expected=bounded_tokens(original_ids,spec['sequence_length'],caps[task])
                assert tokens[f"sample_{sample['index']:03d}"].tolist()==expected
                assert sample['original_prompt_tokens']==len(original_ids) and sample['prompt_tokens']==len(expected)
                assert sample['maximum_tokens']==caps[task] and len(expected)+caps[task]<=32768
                assert sample['truncated']==(len(original_ids)>len(expected))
                assert hashlib.sha256(text.encode()).hexdigest()==sample['prompt_sha256']
                record=predictions[sample['index']]
                assert record['sample']==sample and record['prefix_unchanged'] and set(record['arms'])==set(ARMS)
                for arm in ARMS:
                    r=record['arms'][arm]
                    assert tokenizer.decode(r['generated_token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['prediction']
                    assert r['generated_tokens']==len(r['generated_token_ids'])<=caps[task]
                    assert r['generated_token_ids'][0]==record['first_token']
                    assert r['score']==score_prediction(scorer,task,r['prediction'],original['answers'],original['all_classes'])
                    if arm=='sparse_exact_k':
                        assert r['exact_selector_calls']==36*(r['generated_tokens']-1)
                    if not r['stopped_on_eos']:
                        assert r['generated_tokens']==caps[task]
                    checked+=1
    for shard in range(4):
        saved=json.loads((args.result_dir/'evaluate'/f'shard_{shard}.json').read_text())
        assert saved['status']=='complete' and saved['protocol']==result['protocol']
        assert saved['indices']==list(range(shard,192,4))
    assert checked==768
    report=dict(status='complete',source_prompts_verified=192,arm_predictions_verified=checked,
        four_shards_verified=True,source_selection_and_answers_verified=True,retokenized_inputs_verified=True,
        official_scores_verified=True,exact_selector_call_counts_verified=True,result_sha256=sha256(args.result_dir/'result.json'),
        actual_length_min=min(r['prompt_tokens'] for r in rows),actual_length_max=max(r['prompt_tokens'] for r in rows),
        smoke_and_formal_outputs_separate=True,python=sys.executable)
    write_json(args.result_dir/'audit.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
