"""Prepare genuine C4 64x64K fit plus 16x64K held-out packed windows."""
import argparse
import hashlib
from pathlib import Path
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from evaluation.v96kl_common import read_json, write_json, save_tensors, sha256, tensor_hash
def windows(args):
    out=args.calibration
    if (out/'manifest.json').exists():
        record=read_json(out/'manifest.json')
        assert record['status']=='complete' and record['sha256']==sha256(out/'windows.safetensors')
        assert record['model_config_sha256']==sha256(args.model/'config.json')
        assert record['fit_ids']==list(range(64)) and record['validation_ids']==list(range(64,80))
        assert record['shape']==[80,65536] and record['source_sha256']==sha256(Path(__file__))
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
        if len(pieces)%16==0: print('prepared C4 packed windows',len(pieces)//16,'/80',flush=True)
        if len(pieces)==1280: break
    assert len(pieces)==len(seen)==1280
    packed=torch.stack(pieces).reshape(80,65536)
    save_tensors(out/'windows.safetensors',dict(input_ids=packed))
    write_json(out/'manifest.json',dict(status='complete',sha256=sha256(out/'windows.safetensors'),
        model_config_sha256=sha256(args.model/'config.json'),dataset='allenai/c4',revision=revision,
        seed=seed,shuffle_buffer=10000,fit_ids=list(range(64)),validation_ids=list(range(64,80)),
        shape=[80,65536],packing='16 document-disjoint 4096-token excerpts per 64K window; no separators',
        provenance='new capture on this server; same window geometry, not claimed bitwise identical to missing historical artifacts',
        records=records,source_sha256=sha256(Path(__file__))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    windows(p.parse_args())
