"""Execute the reviewed Llama validation plan inside an allocated Slurm job."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from evaluation.prepare_kcuu_llama import OUTPUT, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=OUTPUT / 'llama_preparation.json')
    args = parser.parse_args()
    assert os.environ.get('SLURM_NNODES') == '1'
    assert os.environ.get('CONDA_DEFAULT_ENV') == 'basis'
    plan = json.loads(args.plan.read_text())
    assert plan['status'] == 'cpu_prepared_gpu_not_run'
    assert plan['source_sha256'] == source_hashes(), 'Regenerate the reviewed plan after code changes'
    for command in plan['gpu_smoke_commands']:
        output = Path(command[command.index('--output') + 1])
        if output.exists():
            result = json.loads(output.read_text())
            if result.get('status') == 'passed' and result['source_sha256'] == plan['source_sha256']:
                print('ALREADY PASSED', output, flush=True)
                continue
        log = output.with_suffix('.log')
        print('RUN', shlex.join(command), 'LOG', log, flush=True)
        with log.open('a') as handle:
            handle.write('\nCOMMAND ' + shlex.join(command) + '\n')
            handle.flush()
            code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode
        if code:
            print('FAILED', output, 'exit', code, 'see', log, flush=True)
            sys.exit(code)
        assert json.loads(output.read_text())['status'] == 'passed'
    print('ALL LLAMA SMOKES PASSED; formal performance jobs have not been launched.', flush=True)


if __name__ == '__main__':
    main()
