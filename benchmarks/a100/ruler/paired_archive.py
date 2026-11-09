import json,hashlib,csv
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import numpy as np
from transformers import AutoTokenizer
from evaluation import eval_llama_exact_sparse_page4 as e
ROOT=Path('/mnt/basiskv/ruler');OUT=ROOT/'a100/paired_archive';OUT.mkdir(exist_ok=True)
archive=Path('/home/sesterce/BasisServe-CALS/results/l31-v96-retrievalmix-ruler1100')
c,key=e.load_configuration(SimpleNamespace(output=ROOT/'archive/l31-exact-page4'),verify=True)
protocol=e.read(archive/'protocols/b16r16_page4_fisher.json')
assert c['frozen_original_protocol']==protocol
identity=e.read(c['identity']);assert e.file_hash(c['identity'])==protocol['identity_sha256']
original=e.ROOT/c['checkpoint_manifest_equivalence']['recovered_original']
assert e.file_hash(original)==identity['manifest_sha256']
m=e.read(Path(c['checkpoint'])/'manifest.json');assert m['layers']==e.read(original)['layers']
assert len(m['layers'])==32
for entry in m['layers']:assert e.file_hash(Path(c['checkpoint'])/entry['file'])==entry['sha256']
assert e.file_hash(Path(c['prompts'])/'prompts.json')==protocol['prompts_sha256']
assert c['numerical_policy']==protocol['numerical_policy']
assert protocol['source_sha256']['evaluation/ruler_v1.py']==e.file_hash(e.ROOT/'evaluation/ruler_v1.py')
old=[json.loads(line) for line in (archive/'predictions/b16r16_page4_fisher.jsonl').read_text().splitlines()]
new=[]
for folder in [ROOT/'archive/l31-exact-page4/exact_sparse_page4/evaluate',ROOT/'a100/resume/exact_sparse_page4']:
 new += [e.read(p) for p in folder.glob('sample_*.json') if e.read(p)['status']=='complete']
assert len(old)==len(new)==1100
oldmap={r['sample']['index']:r for r in old};newmap={r['sample']['index']:r for r in new}
assert set(oldmap)==set(newmap)==set(range(1100))
rows=e.read(Path(c['prompts'])/'prompts.json')['rows'];tokenizer=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
hashes=new[0]['bank_sha256'];pairs=[]
for i in range(1100):
 a,b=oldmap[i],newmap[i];row=rows[i]
 assert a['sample']==b['sample']==row
 e.audit_result(b,row,key,hashes,tokenizer,'exact_sparse_page4')
 assert a['status']=='complete' and a['eos_ids']==protocol['eos_ids']
 result=a['result'];ids=result['ids'];eos=set(a['eos_ids'])
 assert a['first_argmax']==ids[0] and 0<len(ids)<=row['maximum_tokens']
 assert not any(t in eos for t in ids[:-1])
 assert result['stopped']==(ids[-1] in eos) and (result['stopped'] or len(ids)==row['maximum_tokens'])
 assert tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==result['prediction']
 assert e.sample_score(result['prediction'],row['answers'],row['match_type'])==result['score']
 e.matched.verify_dispatch(result['kernel_calls'],'ours',32,len(ids))
 if len(ids)>1:
  assert len(result['routing'])==32
  for s in result['routing']:
   assert (s['page_size'],s['token_budget'],s['recent_tokens'],s['sink_tokens'],s['pinned_pages'])==(4,2112,64,0,0)
   assert s['selected_tokens_mean']<=2112
 pairs.append(dict(index=i,task=row['task'],prompt_sha256=row['input_sha256'],archived_score=result['score'],exact_score=b['result']['score'],difference=b['result']['score']-result['score'],archived_gpu=a['gpu'],exact_gpu=b['gpu']))
with (OUT/'pairs.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
rng=np.random.default_rng(c['bootstrap']['seed']);replicates=c['bootstrap']['replicates'];draws={};task_results={}
for task in c['tasks']:
 p=[x for x in pairs if x['task']==task];assert len(p)==100
 d=np.array([x['difference'] for x in p]);sample=rng.integers(0,100,size=(replicates,100));draws[task]=d[sample].mean(axis=1)*100
 task_results[task]=dict(n=100,archived_percent=np.mean([x['archived_score'] for x in p])*100,exact_percent=np.mean([x['exact_score'] for x in p])*100,difference_pp=d.mean()*100,ci95_pp=np.quantile(draws[task],[.025,.975]).tolist(),improved=int((d>0).sum()),worse=int((d<0).sum()))
def summary(tasks):
 dist=np.mean([draws[t] for t in tasks],axis=0)
 return dict(tasks=len(tasks),samples=100*len(tasks),archived_percent=float(np.mean([task_results[t]['archived_percent'] for t in tasks])),exact_percent=float(np.mean([task_results[t]['exact_percent'] for t in tasks])),difference_pp=float(np.mean([task_results[t]['difference_pp'] for t in tasks])),ci95_pp=np.quantile(dist,[.025,.975]).tolist())
report=dict(status='passed',definition='exact minus archived Page4; percentage points',matched=dict(unique_sample_ids=1100,prompt_hashes=1100,all_sample_metadata=1100,checkpoint='All 32 factor hashes match recovered original manifest and identity',budget='Page4; historical 2048 + recent 64; sink 0; hard maximum 2112',scoring='Same ruler_v1.py hash; recomputed 2200 scores; native EOS and official caps verified'),bootstrap=dict(c['bootstrap'],interval='percentile 2.5/97.5; fixed tasks, within-task paired resampling',primary='11 tasks'),all11=summary(c['tasks']),paper10=summary(c['paper_tasks']),per_task=task_results,hardware=dict(archived=dict(Counter(p['archived_gpu'] for p in pairs)),exact=dict(Counter(p['exact_gpu'] for p in pairs))),limitations=['Archived baseline was not rerun in the exact-score environment.','Three configured source files differ from archived source hashes; selector and K placement intentionally differ.','Bootstrap describes sample uncertainty; it does not isolate hardware or implementation effects.','A CI spanning zero is not proof of equivalence.'],original_source_differences=c['original_source_differences'])
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','all11','paper10','hardware']},indent=2))
