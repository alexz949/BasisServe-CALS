import json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
root=Path('/mnt/basiskv/qwen-exact');api=HfApi();repo='alexz949/BasisServe-CALS';rev=api.repo_info(repo).sha
prefix='checkpoints/attention_c1/qwen3_8b_post_uniform_v96_128k_als6_retrievalmix'
assets=snapshot_download(repo,revision=rev,allow_patterns=[prefix+'/attention_v96/*',prefix+'/identity/v96.json','evaluation/qwen3-8b-ruler128k-1100-seed42/*'],max_workers=4)
model=snapshot_download('Qwen/Qwen3-8B',revision='b968826d9c46dd6066d109eabc6255188de91218',allow_patterns=['*.json','*.safetensors','*.jinja','*.txt'],max_workers=4)
(root/'assets.json').write_text(json.dumps(dict(revision=rev,model=model,checkpoint=assets+'/'+prefix+'/attention_v96',identity=assets+'/'+prefix+'/identity/v96.json',data=assets+'/evaluation/qwen3-8b-ruler128k-1100-seed42'),indent=2)+'\n')
print('FETCH COMPLETE',flush=True)
