"""Prepare audited C4 windows and every LongBench-v1 task/example."""
import argparse
import hashlib
from pathlib import Path
import random
import subprocess
import sys
import zipfile

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.v96kl_common import (
    MODEL, CALIBRATION, DATA, OFFICIAL, configure, read_json, write_json, save_tensors, sha256, tensor_hash,
)
from evaluation.prepare_longbench_c1 import bounded_tokens

TASKS=('narrativeqa','qasper','multifieldqa_en','multifieldqa_zh','hotpotqa','2wikimqa','musique',
       'dureader','gov_report','qmsum','multi_news','vcsum','trec','triviaqa','samsum','lsht',
       'passage_count','passage_retrieval_en','passage_retrieval_zh','lcc','repobench-p')


def windows(args):
    out=args.calibration
    if (out/'manifest.json').exists():
        record=read_json(out/'manifest.json')
        assert record['status']=='complete' and record['sha256']==sha256(out/'windows.safetensors')
        assert record['model_config_sha256']==sha256(args.model/'config.json')
        assert record['fit_ids']==list(range(64)) and record['validation_ids']==list(range(64,80))
        assert record['shape']==[80,32768] and record['source_sha256']==sha256(Path(__file__))
        print('C4 windows already complete',flush=True)
        return
    revision='1588ec454efa1a09f29cd18ddd04fe05fc8653a2'
    seed=20260828
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    dataset=load_dataset('allenai/c4','en',split='train',streaming=True,revision=revision).shuffle(seed=seed,buffer_size=10000)
    rng=random.Random(seed)
    pieces,records,seen=[],[],set()
    for index,row in enumerate(dataset):
        document=hashlib.sha256(row['text'].encode()).hexdigest()
        if document in seen: continue
        ids=tokenizer(row['text'],add_special_tokens=False)['input_ids']
        if len(ids)<4096: continue
        start=rng.randint(0,len(ids)-4096)
        piece=torch.tensor(ids[start:start+4096],dtype=torch.int32)
        pieces.append(piece);seen.add(document)
        records.append(dict(document_sha256=document,stream_index=index,start=start,token_sha256=tensor_hash(piece)))
        if len(pieces)%8==0: print('prepared C4 packed windows',len(pieces)//8,'/80',flush=True)
        if len(pieces)==640: break
    assert len(pieces)==len(seen)==640
    packed=torch.stack(pieces).reshape(80,32768)
    save_tensors(out/'windows.safetensors',dict(input_ids=packed))
    write_json(out/'manifest.json',dict(status='complete',sha256=sha256(out/'windows.safetensors'),
        model_config_sha256=sha256(args.model/'config.json'),dataset='allenai/c4',revision=revision,
        seed=seed,shuffle_buffer=10000,fit_ids=list(range(64)),validation_ids=list(range(64,80)),
        shape=[80,32768],packing='8 document-disjoint 4096-token excerpts per 32K window; no separators',
        provenance='new capture on this server; same window geometry, not claimed bitwise identical to missing historical artifacts',
        records=records,source_sha256=sha256(Path(__file__))))


def longbench(args):
    out=args.data
    if (out/'manifest.json').exists():
        record=read_json(out/'manifest.json')
        assert record['status']=='complete' and record['tokens_sha256']==sha256(out/'tokens.safetensors')
        assert record['samples_sha256']==sha256(out/'samples.json')
        assert record['model_config_sha256']==sha256(args.model/'config.json')
        assert record['tasks']==list(TASKS) and record['source_sha256']==sha256(Path(__file__))
        for name,digest in record['official_sha256'].items(): assert sha256(OFFICIAL/name)==digest
        print('LongBench inputs already complete',flush=True)
        return
    import json
    prompts=read_json(OFFICIAL/'config/dataset2prompt.json')
    caps=read_json(OFFICIAL/'config/dataset2maxlen.json')
    revision=HfApi().dataset_info('THUDM/LongBench').sha
    archive=Path(hf_hub_download('THUDM/LongBench','data.zip',repo_type='dataset',revision=revision))
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    rows,tensors,counts,hashes=[],{}, {},{}
    with zipfile.ZipFile(archive) as data:
        for task in TASKS:
            names=[n for n in data.namelist() if n==task+'.jsonl' or n.endswith('/'+task+'.jsonl')]
            assert len(names)==1
            raw=data.read(names[0]);hashes[task]=hashlib.sha256(raw).hexdigest()
            records=[json.loads(line) for line in raw.splitlines() if line.strip()]
            counts[task]=len(records)
            assert records
            for ordinal,r in enumerate(records):
                prompt=prompts[task].format(**r)
                ids=tokenizer(prompt,add_special_tokens=True)['input_ids']
                ids_bounded=bounded_tokens(ids,32768,caps[task])
                index=len(rows);tensor=torch.tensor(ids_bounded,dtype=torch.int32)
                tensors[f'sample_{index:05d}']=tensor
                rows.append(dict(index=index,task=task,ordinal=ordinal,source_id=r['_id'],answers=r['answers'],
                    all_classes=r['all_classes'],official_length=r['length'],original_prompt_tokens=len(ids),
                    prompt_tokens=len(tensor),maximum_tokens=caps[task],input_ids_sha256=tensor_hash(tensor)))
            print('prepared LongBench',task,len(records),'total',len(rows),flush=True)
    save_tensors(out/'tokens.safetensors',tensors)
    write_json(out/'samples.json',rows)
    write_json(out/'manifest.json',dict(status='complete',tokens_sha256=sha256(out/'tokens.safetensors'),
        samples_sha256=sha256(out/'samples.json'),tasks=list(TASKS),counts=counts,total=len(rows),
        model_config_sha256=sha256(args.model/'config.json'),dataset_revision=revision,
        archive_sha256=sha256(archive),task_sha256=hashes,
        official_revision=subprocess.check_output(['git','-C',str(OFFICIAL.parent),'rev-parse','HEAD'],text=True).strip(),
        official_sha256={n:sha256(OFFICIAL/n) for n in ('eval.py','metrics.py','config/dataset2prompt.json','config/dataset2maxlen.json')},
        protocol='All 21 LongBench-v1 tasks, all examples in source order; official task prompts/caps; base model without chat template; prefix/suffix token truncation reserves generation within native 32768 context',
        source_sha256=sha256(Path(__file__))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('windows','longbench'))
    p.add_argument('--model',type=Path,default=MODEL)
    p.add_argument('--calibration',type=Path,default=CALIBRATION)
    p.add_argument('--data',type=Path,default=DATA)
    args=p.parse_args();configure()
    (windows if args.stage=='windows' else longbench)(args)


if __name__=='__main__': main()
