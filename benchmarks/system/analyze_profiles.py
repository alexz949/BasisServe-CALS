"""Export official profiler tables without interpreting overlap as additive time."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/system_benchmarks/l40s'))
    parser.add_argument('--nsys', default='/deac/opt/rocky9-noarch/nvidia/cuda/12.3.2/bin/nsys')
    parser.add_argument('--trace-name', default='e2e_trace')
    args = parser.parse_args()
    root = args.output
    report = dict(status='awaiting_review', commands=[], nsys={}, ncu={},
        interpretation='NVTX CPU and GPU-projected ranges can overlap. Summed GPU kernel duration across four devices is not model wall time. Mapped host traffic is not equivalent to CUDA DMA copy payload.')
    trace = root/(args.trace_name+'.nsys-rep')
    names = ['cuda_gpu_kern_sum','nvtx_sum','nvtx_gpu_proj_sum','cuda_api_sum','cuda_gpu_mem_size_sum']
    if trace.exists():
        command = [args.nsys, 'stats', '--report', ','.join(names), '--format', 'csv',
            '--output', str(root/(args.trace_name+'_stats')), str(trace)]
        print('COMMAND', ' '.join(command), flush=True)
        result = subprocess.run(command, capture_output=True, text=True)
        report['commands'].append(dict(command=command, returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr))
        for name in names:
            path = root/f'{args.trace_name}_stats_{name}.csv'
            if path.exists():
                rows = list(csv.DictReader(path.open()))
                report['nsys'][name] = dict(path=str(path), row_count=len(rows), rows=rows)
        database = root/(args.trace_name+'.sqlite')
        if database.exists():
            with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
                tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
                    rows = connection.execute('SELECT deviceId, count(*), sum(end-start), min(start), max(end) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId').fetchall()
                    report['nsys']['device_kernel_coverage'] = [dict(device=r[0],kernel_count=r[1],summed_kernel_ns=r[2],first_start_ns=r[3],last_end_ns=r[4]) for r in rows]
    else:
        report['nsys']['status'] = 'missing_trace'
    ncu = root/'router_profile.csv'
    if ncu.exists():
        lines = ncu.read_text().splitlines()
        header = next((i for i, line in enumerate(lines) if '"Metric Name"' in line and '"Metric Value"' in line), None)
        if header is not None:
            rows = list(csv.DictReader(io.StringIO('\n'.join(lines[header:]))))
            selected = [row for row in rows if any(token in row.get('Metric Name','') for token in
                ['dram__bytes','lts__t_bytes','warps_active','warp_issue_stalled','sass_thread_inst_executed_op_f',
                 'gpu__time_duration','arithmetic_intensity','roofline'])]
            report['ncu'] = dict(path=str(ncu), metric_rows=len(rows), selected_metric_rows=selected,
                interpretation='Warmed router scan only, cache-control none, clock-control none. Report measured DRAM bytes separately from logical scan bytes. FP32 FMA counts are two FLOPs each; special-function operations need separate accounting.')
        else:
            report['ncu']['status'] = 'no_metric_table'
    else:
        report['ncu']['status'] = 'missing_csv'
    availability=root/'ncu_status.json'
    if availability.exists():
        report['ncu']['availability']=json.loads(availability.read_text())
    report['source_sha256'] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), root/'profile_router_input.json'] if path.exists()}
    target = root/(args.trace_name+'_analysis.json')
    assert not target.exists(), str(target)
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(dict(output=str(target),nsys_tables=list(report['nsys']),ncu_metrics=report['ncu'].get('metric_rows')), flush=True)


if __name__ == '__main__':
    main()
