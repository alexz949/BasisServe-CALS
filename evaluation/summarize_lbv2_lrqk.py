"""Report completed LRQK questions and explicit Slurm failures separately."""
import argparse
import json
from pathlib import Path
import subprocess
from evaluation.eval_qwen35_longbench_v2 import parse_answer
from evaluation.qwen35_hybrid_common import atomic_save, sha256


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--job',required=True)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',args.job,
        '--format=JobIDRaw,State,ExitCode'],text=True)
    states={}
    for line in accounting.splitlines():
        fields=line.split('|')
        if fields[0].startswith(args.job+'_') and fields[0].split('_')[-1].isdigit():
            states[int(fields[0].split('_')[-1])]=dict(state=fields[1],exit_code=fields[2])
    completed,failed=[],[]
    for index in range(70):
        path=args.root/'lrqk'/f'{index:03d}.json'
        if path.exists():
            row=json.loads(path.read_text())
            dense=json.loads((args.root/'full'/f'{index:03d}.json').read_text())
            assert row['status']=='complete' and row['sample']==dense['sample']
            assert row['protocol']['arm']=='lrqk'
            if completed:
                assert row['protocol']==completed[0]['protocol']
            assert row['parsed_answer']==parse_answer(row['answer']['text'])
            assert row['score']==int(row['parsed_answer']==row['sample']['answer'])
            for name in ('first_logits_sha256','input_ids_sha256'):
                assert row['reasoning'][name]==dense['reasoning'][name]
            for name in ('v_bank_sha256','data_sha256','model_identity','budget','prefill','generation','eos_token_ids'):
                assert row['protocol'][name]==dense['protocol'][name]
            completed.append(row)
        else:
            log=Path('logs')/f'q35-lbv2-lrqk-{args.job}_{index}.err'
            text=log.read_text() if log.exists() else ''
            failed.append(dict(index=index,slurm=states.get(index),stderr=str(log),
                stderr_sha256=sha256(log) if log.exists() else None,
                solve_assertion_observed='in _solve' in text and 'AssertionError' in text))
    correct=sum(r['score'] for r in completed)
    report=dict(status='reported',requested=70,completed=len(completed),failed_or_missing=len(failed),
        correct=correct,unparsed=sum(r['parsed_answer'] is None for r in completed),
        completed_only_accuracy=100*correct/len(completed) if completed else None,
        full_set_accuracy=100*correct/70 if not failed else None,
        full_set_accuracy_bounds=[100*correct/70,100*(correct+len(failed))/70],
        note='Completed-only accuracy is conditional on successful execution; failures are not ordinary wrong answers.',
        failures=failed,protocol=completed[0]['protocol'] if completed else None,
        artifacts={str(args.root/'lrqk'/f"{r['sample']['index']:03d}.json"):
            sha256(args.root/'lrqk'/f"{r['sample']['index']:03d}.json") for r in completed},
        slurm_accounting=accounting)
    atomic_save(args.root/'lrqk_summary.json',report)
    print(json.dumps({k:report[k] for k in ('requested','completed','failed_or_missing','correct','unparsed','completed_only_accuracy','full_set_accuracy')}),flush=True)


if __name__=='__main__':
    main()
