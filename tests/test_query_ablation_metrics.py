import torch
from evaluation.query_ablation_metrics import query_metrics,random_support


def test_query_controls_share_support_rules_and_identical_queries_match():
    torch.manual_seed(2)
    h,kv,d,t=32,8,16,321
    value=torch.randn(1,kv,t,d);key=torch.randn_like(value)
    positions=list(range(256,288));q=torch.randn(1,h,32,d)
    pool=q[0].expand(16,-1,-1,-1).clone()
    f={'base_left_b4':torch.randn(kv,d,4),'base_right_b4':torch.randn(kv,4,d),'base_bias_b4':torch.zeros(kv,d)}
    cos=torch.ones(1,t,d);sin=torch.zeros_like(cos)
    reports=query_metrics(value,key,q,positions,cos,sin,f,pool,q[0].mean(1),64,0,budget=128)
    assert reports['base4_correct']==reports['base4_shuffled']
    assert reports['exact_k']['page_recall']==1 and reports['exact_k']['routed_page_recall']==1
    for report in reports.values():
        assert all(0<=x<=1.00001 for x in report.values())
    for n in (129,160,193,65536):
        ids,valid=random_support(n,kv,96,'cpu',42)
        for row,mask in zip(ids[0],valid[0]):
            selected=row[mask]
            assert len(selected)<=96 and len(selected.unique())==len(selected)
            assert set(range(32))<=set(selected.tolist())
            assert set(range(n-64,n))<=set(selected.tolist())
