"""Official RULER Llama-3 Instruct prompts: 64K, 11 tasks, 30 samples each."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from transformers import AutoTokenizer
import nltk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.llama31_instruct_c1_64k import MODEL, RUN, LENGTH
from evaluation.v96kl_common import sha256, read_json, write_json
from evaluation.ruler_v1 import parse_tasks

TASKS = 'niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multiquery,niah_multivalue,vt,fwe,qa_1,qa_2'
REVISION = 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--run',type=Path,default=RUN)
    args = parser.parse_args()
    ruler = ROOT/'results/tools/RULER'
    actual = subprocess.check_output(['git','-C',str(ruler),'rev-parse','HEAD'],text=True).strip()
    assert actual == REVISION
    directory = args.run/'ruler_data'
    logs = args.run/'logs'
    logs.mkdir(parents=True,exist_ok=True)
    sources = ruler/'scripts/data/synthetic/json'
    for name,url in {
        'squad.json':'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json',
        'hotpotqa.json':'https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json',
    }.items():
        target = sources/name
        if not target.exists():
            partial = target.with_suffix('.download')
            subprocess.run(['curl','-fL',url,'-o',str(partial)],check=True)
            partial.replace(target)
        read_json(target)
    if not (sources/'PaulGrahamEssays.json').exists():
        with (logs/'ruler_essays.log').open('w') as log:
            subprocess.run([sys.executable,'download_paulgraham_essay.py'],
                cwd=sources,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert len(read_json(sources/'PaulGrahamEssays.json')['text']) > 100000
    nltk.download('punkt',quiet=True)
    nltk.download('punkt_tab',quiet=True)
    protocol = dict(model_config_sha256=sha256(args.model/'config.json'),
        tokenizer_config_sha256=sha256(args.model/'tokenizer_config.json'),
        ruler_revision=REVISION, sequence_length=LENGTH, samples_per_task=30,
        tasks=TASKS.split(','), seed=42, model_template_type='meta-llama3',
        prompt_margin=8, add_special_tokens=False,
        source_data_sha256={p.name:sha256(p) for p in sources.glob('*.json')})
    if (directory/'manifest.json').exists():
        record = read_json(directory/'manifest.json')
        assert record['status'] == 'complete' and record['protocol'] == protocol
        for item in record['artifacts'].values():
            assert sha256(directory/item['file']) == item['sha256']
        print('RULER data verified',flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    environment = dict(os.environ)
    environment['PATH'] = str(Path(sys.executable).parent)+':'+environment.get('PATH','')
    artifacts = {}
    for task in parse_tasks(TASKS):
        command = [sys.executable,str(ruler/'scripts/data/prepare.py'),
            '--save_dir',str(directory),'--benchmark','synthetic','--task',task.name,
            '--subset','validation','--tokenizer_path',str(args.model),'--tokenizer_type','hf',
            '--max_seq_length',str(LENGTH-8),'--num_samples','30','--random_seed','42',
            '--model_template_type','meta-llama3']
        print(shlex.join(command),flush=True)
        with (logs/f'ruler_data_{task.name}.log').open('a') as log:
            subprocess.run(command,cwd=ruler/'scripts/data',env=environment,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        path = directory/task.name/'validation.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 30
        lengths = []
        for row in rows:
            assert row['outputs'] and row['input'].startswith('<|begin_of_text|>')
            ids = tokenizer(row['input']+row.get('answer_prefix',''),add_special_tokens=False)['input_ids']
            assert len(ids)+task.tokens_to_generate <= LENGTH
            assert ids[0] == tokenizer.bos_token_id
            assert ids[1] != tokenizer.bos_token_id
            lengths.append(len(ids))
        artifacts[task.name] = dict(file=str(path.relative_to(directory)),sha256=sha256(path),
            samples=30,prompt_lengths=lengths,maximum_tokens=task.tokens_to_generate,
            command=shlex.join(command))
        print(f'{task.name}:30 validated; prompt range={min(lengths)}..{max(lengths)}',flush=True)
    write_json(directory/'manifest.json',dict(status='complete',protocol=protocol,
        artifacts=artifacts,command=shlex.join(sys.argv)))


if __name__ == '__main__':
    main()
