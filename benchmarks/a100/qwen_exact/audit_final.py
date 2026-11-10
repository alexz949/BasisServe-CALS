import csv,importlib.util,json,hashlib,statistics
from pathlib import Path
from collections import Counter
import numpy as np
from transformers import AutoTokenizer
ROOT=Path('/mnt/basiskv/qwen-exact')
spec=importlib.util.spec_from_file_location('qwen_worker',ROOT/'setup/worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
c=w.read(ROOT/'configuration.json');cfg=c['configuration'];key=c['hash'];assert w.e.fingerprint(cfg)==key
for p,digest in cfg['files'].items():assert w.sha(p)==digest,p
assert w.read(ROOT/'smoke.json')['configuration_hash']==key and w.read(ROOT/'smoke.json')['status']=='passed'
assert 'Qwen job exit code: 0' in (ROOT/'logs/launcher.log').read_text()
rows=w.read(ROOT/'prompts.json');tokenizer=AutoTokenizer.from_pretrained(cfg['assets']['model'],local_files_only=True)
files=list((ROOT/'evaluate').glob('sample_*.json'));assert len(files)==1100
new={}
for p in files:
 r=w.read(p);i=r['sample']['index'];assert i not in new;w.audit(r,rows[i],key,tokenizer);new[i]=r
assert set(new)==set(range(1100))
old=[json.loads(s) for s in (w.ARCHIVE/'predictions/b16r16_page4_fisher.jsonl').read_text().splitlines()];assert len(old)==1100
oldmap={r['sample']['index']:r for r in old};assert set(oldmap)==set(new)
a=cfg['assets'];identity=w.read(a['identity']);manifest=w.read(Path(a['checkpoint'])/'manifest.json');protocol=w.read(w.ARCHIVE/'protocols/b16r16_page4_fisher.json')
assert w.sha(a['identity'])==protocol['identity_sha256']
assert w.sha(Path(a['checkpoint'])/'manifest.json')==identity['manifest_sha256']
assert protocol['rope']==cfg['rope'] and protocol['runtime_config']['rope_parameters']==cfg['rope_parameters']
assert protocol['chat_template_kwargs']==cfg['chat_template_kwargs']
assert protocol['rank_schedule']==[96]*36
pairs=[]
for i in range(1100):
 a,b=oldmap[i],new[i];row=rows[i];assert a['status']=='complete' and a['sample']==row==b['sample']
 result=a['result'];ids=result['ids'];assert 0<len(ids)<=row['maximum_tokens']
 assert not any(t==151645 for t in ids[:-1]);assert result['stopped']==(ids[-1]==151645) and (result['stopped'] or len(ids)==row['maximum_tokens'])
 assert tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==result['prediction']
 assert w.sample_score(result['prediction'],row['answers'],row['match_type'])==result['score']
 if len(ids)>1:
  assert len(result['routing'])==36
  for s in result['routing']:assert (s['page_size'],s['token_budget'],s['recent_tokens'],s['sink_tokens'])==(4,2112,64,0) and s['selected_tokens_mean']<=2112
 pairs.append(dict(index=i,task=row['task'],prompt_sha256=row['input_sha256'],archived_score=result['score'],exact_score=b['result']['score'],difference=b['result']['score']-result['score']))
with (ROOT/'pairs.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
rng=np.random.default_rng(20260925);draws={};per_task={}
for task in w.matched.TASKS:
 group=[p for p in pairs if p['task']==task];assert len(group)==100
 d=np.array([p['difference'] for p in group]);draws[task]=100*d[rng.integers(0,100,size=(10000,100))].mean(axis=1)
 per_task[task]=dict(n=100,archived_percent=100*statistics.mean(p['archived_score'] for p in group),exact_percent=100*statistics.mean(p['exact_score'] for p in group),difference_pp=100*float(d.mean()),ci95_pp=np.quantile(draws[task],[.025,.975]).tolist())
def aggregate(tasks):
 dist=np.mean([draws[t] for t in tasks],axis=0)
 return dict(archived_percent=statistics.mean(per_task[t]['archived_percent'] for t in tasks),exact_percent=statistics.mean(per_task[t]['exact_percent'] for t in tasks),difference_pp=statistics.mean(per_task[t]['difference_pp'] for t in tasks),ci95_pp=np.quantile(dist,[.025,.975]).tolist())
summary=dict(status='passed',samples=1100,configuration_hash=key,environment='lowrank',command='bash /mnt/basiskv/qwen-exact/setup/run.sh',audit_command='python /mnt/basiskv/qwen-exact/setup/audit_final.py',all11=aggregate(w.matched.TASKS),paper10=aggregate([t for t in w.matched.TASKS if t!='niah_single_3']),per_task=per_task,bootstrap=dict(replicates=10000,seed=20260925,unit='paired samples resampled within each fixed task',interval='percentile 2.5/97.5',difference='exact minus archived'),audit=dict(ids=1100,prompt_hashes='all matched archived in prepare; frozen prompt and source hashes reverified',checkpoint='all 36 factor hashes verified; identity and manifest match archived protocol',budget='2048 historical + recent64, Page4, no sinks',scoring='2200 predictions rescored; generation caps and EOS verified'),hardware=dict(archived=dict(Counter(x['gpu'] for x in old)),exact=dict(Counter(x['gpu'] for x in new.values()))),mean_seconds=statistics.mean(x['result']['seconds'] for x in new.values()),limitations=['Archived baseline was not rerun on this A100 environment; paired differences may include hardware and implementation effects.','Bootstrap captures evaluation sample uncertainty, not cross-GPU numerical variation; inclusion of zero does not establish equivalence.'])
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['status','samples','all11','paper10','per_task','hardware']},indent=2))
