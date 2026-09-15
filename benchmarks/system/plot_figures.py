"""Paper figures from measured CSV rows only. Additional figures follow their data."""
import argparse
import csv
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def figure_a(root):
    path=root/'tp_collective.csv'
    if not path.exists():return False
    with path.open() as stream:rows=list(csv.DictReader(stream))
    batches=[1,8,32,64,128,256]
    methods=['Dense AR','C1 global-coordinate AR','C1 LR-AG']
    expected={(layer,batch,method) for layer in range(32) for batch in batches for method in methods}
    observed={(int(r['layer']),int(r['batch']),r['method']) for r in rows}
    if observed!=expected:return False
    assert len(rows)==len(expected)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for ax,column,title in zip(axes,['collective_ms','total_ms'],['Collective','Complete output block']):
        for method in methods:
            values=[statistics.mean(float(r[column])*1000 for r in rows if r['method']==method and int(r['batch'])==batch) for batch in batches]
            ax.plot(range(len(batches)),values,marker='o',label=method)
        ax.set_xticks(range(len(batches)),batches)
        ax.set(xlabel='Decode batch',ylabel='Mean across 32 layer p50s (μs)',title=title)
        ax.grid(alpha=.25)
    axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Llama-3.1-8B base, TP4 L40S, actual KL V96 layer ranks')
    fig.text(.5,.01,'C1 AR uses the same factors in global coordinates; it is not an independently fitted LR-AR model.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.96))
    for extension in ['pdf','png']:fig.savefig(root/f'figure_A.{extension}',dpi=180)
    plt.close(fig)
    return True


def figure_c(root):
    path=root/'router.csv'
    if not path.exists():return False
    with path.open() as stream:
        rows=[r for r in csv.DictReader(stream) if int(r['length'])==65536 and int(r['batch'])==1]
    methods=['BasisKV B16R16','shadow','loki','lrqk']
    by_method={r['method']:r for r in rows}
    if set(by_method)!=set(methods):return False
    labels=['BasisKV B16R16','ShadowKV','Loki r32','LRQK r32']
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.9))
    for ax,column,divisor,ylabel in zip(axes,['route_p50_us','routing_state_bytes'],[1,2**20],['Router latency (μs), p50','Reported routing state (MiB)']):
        values=[float(by_method[m][column])/divisor for m in methods]
        ax.bar(range(4),values,color=['#2ca02c','#9467bd','#1f77b4','#ff7f0e'])
        ax.set_xticks(range(4),labels,rotation=15,ha='right')
        ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
        for i,value in enumerate(values):ax.text(i,value,f'{value:.1f}',ha='center',va='bottom',fontsize=9)
        ax.set_ylim(0,max(values)*1.2)
    support='; '.join(f'{label}: {by_method[m]["min_support_per_query"]}–{by_method[m]["max_support_per_query"]}/query, {by_method[m]["minimum_unique_fetch_tokens_per_kv"]}–{by_method[m]["maximum_unique_fetch_tokens_per_kv"]}/KV union' for m,label in zip(methods,labels))
    fig.suptitle('Llama-3.1-8B base, layer 3, 64K, batch 1')
    fig.text(.02,.065,support[:support.index('; Loki')],fontsize=8)
    fig.text(.02,.04,support[support.index('Loki'):],fontsize=8)
    fig.text(.02,.015,'Basis includes RoPE tables and factors; LRQK includes active K for query updates. Workspaces excluded.',fontsize=8)
    fig.tight_layout(rect=(0,.14,1,.96))
    for extension in ['pdf','png']:fig.savefig(root/f'figure_C.{extension}',dpi=180)
    plt.close(fig);return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/system_benchmarks/l40s'))
    a=p.parse_args();path=a.output/'sparse_operator.csv'
    assert path.exists()
    with path.open() as f:rows=sorted(csv.DictReader(f),key=lambda r:int(r['length']))
    assert [int(r['length']) for r in rows]==[16384,32768,65536,131072]
    fig,ax=plt.subplots(figsize=(6.4,4.0))
    columns=[('dense_local_us','Dense local'),('dense_offload_us','Dense K offload'),
             ('sparse_local_us','BasisKV sparse local'),('sparse_offload_us','BasisKV sparse K offload')]
    for column,label in columns:ax.plot([int(r['length'])/1024 for r in rows],[float(r[column])/1000 for r in rows],marker='o',label=label)
    ax.set(xlabel='Context length (Ki tokens)',ylabel='Attention latency (ms), p50',
           title='Llama-3.1-8B layer 3, V96, batch 1, sparse budget 2048')
    ax.set_xticks([16,32,64,128]);ax.grid(alpha=.25);ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ['pdf','png']:fig.savefig(a.output/f'figure_B.{extension}',dpi=180)
    plt.close(fig)
    made_a=figure_a(a.output)
    made_c=figure_c(a.output)
    print(dict(figure_B='written',figure_A='written' if made_a else 'awaiting complete measurements',figure_C='written' if made_c else 'awaiting LRQK'))

if __name__=='__main__':main()
