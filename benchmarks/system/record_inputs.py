"""Content identity of model, fitted factors and transformed coordinates."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import socket
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'))
    args=parser.parse_args()
    identity=json.loads((args.root/'manifests/v96.json').read_text())
    model=Path(identity['model'])
    paths=[args.root/'manifests/v96.json',args.root/'v96/manifest.json',
        args.root/'calibration/windows.safetensors',model/'config.json',model/'model.safetensors.index.json']
    paths.extend(sorted(model.glob('*.safetensors')))
    for folder in [args.root/'v96/selected_factors',args.root/'ours_b16r16',args.output/'routing_basis']:
        factors=[folder/f'layer_{layer:03d}.safetensors' for layer in range(32)]
        assert all(path.exists() for path in factors)
        paths.extend(factors)
    records=[]
    for path in paths:
        before=path.stat()
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        after=path.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        records.append(dict(path=str(path),resolved_path=str(path.resolve()),bytes=before.st_size,
            mtime_ns=before.st_mtime_ns,sha256=digest))
        print(dict(path=str(path),bytes=before.st_size,sha256=digest),flush=True)
    assert next(r['sha256'] for r in records if r['path']==str(model/'config.json'))==identity['model_config_sha256']
    assert next(r['sha256'] for r in records if r['path']==str(args.root/'v96/manifest.json'))==identity['manifest_sha256']
    target=args.output/'input_identity.json';assert not target.exists()
    target.write_text(json.dumps(dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(),command_line=sys.argv,model_identity=identity,files=records,
        scope='Content hashes observed at this timestamp; this manifest does not retroactively prove inputs were unchanged before observation'),indent=2)+'\n')


if __name__=='__main__':main()
