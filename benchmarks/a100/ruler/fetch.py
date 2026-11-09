import hashlib
import json
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download

REPO = 'alexz949/BasisServe-CALS'
REV = 'bb3e548586ecd47cddd8201eb3e1f33077600205'
OLD = '63a35906f354cc239339465db97fab136ccdb1de'
ROOT = Path('/mnt/basiskv/ruler')
CODE = ROOT / 'code'
PREFIX = 'results/audits-20261109/'
api = HfApi()

def fetch(item):
    revision, name = item
    path = Path(hf_hub_download(REPO, name, revision=revision))
    print('FETCHED', name, flush=True)
    return name, path

files = [x.path for x in api.list_repo_tree(REPO, PREFIX.rstrip('/'), revision=REV, recursive=True)
         if hasattr(x, 'size') and (x.path.startswith(PREFIX+'source/')
         or x.path.startswith(PREFIX+'l31-exact-page4/')
         or x.path in (PREFIX+'README.md', PREFIX+'snapshot.json', PREFIX+'files.sha256'))]
with ThreadPoolExecutor(max_workers=8) as pool:
    for name, path in pool.map(fetch, [(REV, name) for name in files]):
        rel = name.removeprefix(PREFIX)
        dest = ROOT / 'archive' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(path, dest)

checks = (ROOT/'archive/files.sha256').read_text().splitlines()
expected = {line.split(maxsplit=1)[1].strip(): line.split()[0] for line in checks}
for name in files:
    rel = name.removeprefix(PREFIX)
    if rel == 'files.sha256':
        continue
    path = ROOT/'archive'/rel
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[rel]

for path in (ROOT/'archive/source').rglob('*'):
    if path.is_file():
        dest = CODE/path.relative_to(ROOT/'archive/source')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)

config = json.loads((ROOT/'archive/l31-exact-page4/configuration.json').read_text())['configuration']
old_prefix = '/home/sesterce/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/snapshots/'+OLD+'/'
needed = [f['path'].removeprefix(old_prefix) for f in config['files'] if f['path'].startswith(old_prefix)]
with ThreadPoolExecutor(max_workers=8) as pool:
    for name, path in pool.map(fetch, [(OLD, name) for name in needed]):
        assert path.is_file()

run = CODE/'runs/l31-exact-page4'
run.mkdir(parents=True, exist_ok=True)
for name in ('configuration.json', 'smoke.json', 'provenance/original_checkpoint_manifest.json'):
    dest = run/name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(ROOT/'archive/l31-exact-page4'/name, dest)
for name, sha in config['sources'].items():
    assert hashlib.sha256((CODE/name).read_bytes()).hexdigest() == sha, name
print('ARCHIVE AND ASSETS VERIFIED', flush=True)
