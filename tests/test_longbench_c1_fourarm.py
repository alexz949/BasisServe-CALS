from pathlib import Path

import torch

from basisserve.core.c1_conditional_page_attention import _selected_pages
from evaluation.prepare_longbench_c1 import bounded_tokens, choose_rows
from evaluation.longbench_exact_pages import ExactPageAttention
from evaluation.eval_longbench_c1_fourarm import official_scorer, score_prediction


def test_token_cap_preserves_prefix_suffix_and_generation_room():
    assert bounded_tokens(list(range(20)),16,4) == list(range(6))+list(range(14,20))
    assert bounded_tokens(list(range(5)),16,4) == list(range(5))
    assert len(bounded_tokens(list(range(40)),32,7)) == 25


def test_fixed_sampling_independent_of_labels_and_input_order():
    rows = [{'_id':str(i),'answers':['a']} for i in range(50)]
    a = choose_rows(rows,'qasper',32,73)
    b = choose_rows([{'_id':r['_id'],'answers':['changed']} for r in reversed(rows)],'qasper',32,73)
    assert [r['_id'] for r in a] == [r['_id'] for r in b]


def test_exact_selector_ignores_proxy_and_matches_manual_attention():
    torch.manual_seed(91)
    q,k,v = torch.randn(1,4,1,4),torch.randn(1,2,17,4),torch.randn(1,2,17,6)
    sidecar,projector = torch.randn(1,2,17,3),torch.randn(4,4,3)
    opts = dict(page_size=4,exact_token_budget=12,pinned_prefix_pages=1,scale=.5,
                query_block_size=1,attention_mask=torch.ones(1,1,1,17,dtype=torch.bool),collect_statistics=False)
    wrapper = ExactPageAttention()
    actual = wrapper(q,k,v,sidecar,projector,**opts).output
    changed = wrapper(q,k,v,sidecar*100,projector*30,**opts).output
    torch.testing.assert_close(actual,changed,rtol=0,atol=0)
    expanded_k,expanded_v = k.repeat_interleave(2,1),v.repeat_interleave(2,1)
    scores = q@expanded_k.transpose(-1,-2)*.5
    ids,valid = _selected_pages(scores,torch.ones_like(scores,dtype=torch.bool),kv_heads=2,
                               page_size=4,page_budget=3,pinned_prefix_pages=1)
    support = torch.zeros_like(scores,dtype=torch.bool)
    for h in range(4):
        for page in ids[0,h//2,0].tolist():
            support[0,h,0,page*4:min((page+1)*4,17)] = True
    expected = scores.masked_fill(~support,-torch.inf).softmax(-1)@expanded_v
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    opts['exact_token_budget'] = 128
    full = wrapper(q,k,v,sidecar,projector,**opts).output
    torch.testing.assert_close(full,scores.softmax(-1)@expanded_v,rtol=1e-5,atol=1e-6)
    assert wrapper.calls == wrapper.selector_calls == 3


def test_official_f1_rouge_and_alternative_references():
    root = Path(__file__).resolve().parents[1]
    scorer = official_scorer(root/'external/LongBench')
    assert score_prediction(scorer,'qasper','The answer!',['no','answer'],None) == 1
    assert score_prediction(scorer,'hotpotqa','red',['blue'],None) == 0
    score = score_prediction(scorer,'gov_report','the report is here',['the report is here'],None)
    assert score > .9999
    assert scorer.scorer('gov_report',['the report is here'],[['the report is here']],None) == round(100*score,2)
