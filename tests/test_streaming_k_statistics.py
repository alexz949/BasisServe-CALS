import torch
from basisserve.core.c1_v_conditional_k_router import fit_affine_reduced_rank_map
from basisserve.core.gqa_joint_routing_payload_s80_fisher import S80CompactSoftmaxFisherRouting
from evaluation.streaming_k_statistics import RawBaseMoments, base_from_moments, base_mse, packed_fisher, load_fisher_windows


def test_raw_moments_match_direct_projected_regression_and_mse():
    torch.manual_seed(7)
    v=torch.randn(137,2,8,dtype=torch.float64)
    k=torch.randn(137,2,8,dtype=torch.float64)+3
    f=torch.randn(2,8,5,dtype=torch.float64)
    moments=RawBaseMoments(2,8)
    moments.update(v[:73],k[:73],17)
    moments.update(v[73:],k[73:],11)
    payload=moments.tensors()
    bases=base_from_moments(payload,f,rank=3)[3]
    errors=0.
    for g,base in enumerate(bases):
        c=v[:,g]@f[g]
        direct=fit_affine_reduced_rank_map(row_count=len(v),input_sum=c.sum(0),target_sum=k[:,g].sum(0),
            input_gram=c.T@c,input_target_gram=c.T@k[:,g],rank=3,fit_bias=True)
        prediction=c@base.left@base.right+base.bias
        torch.testing.assert_close(prediction,c@direct.left@direct.right+direct.bias,rtol=1e-9,atol=1e-9)
        errors+=(prediction-k[:,g]).square().sum()
    report=base_mse(payload,f,bases)
    assert abs(report['squared_error']-float(errors))<1e-8
    assert int(payload['count'])==137
    # A new encoder can use the same raw moments, without replaying activations.
    second=base_from_moments(payload,f[:,:,:4],rank=3)
    assert second[3][0].left.shape==(4,3)


def test_fisher_packing_and_query_major_window_minor_order():
    torch.manual_seed(19)
    heads,queries,windows,dim=4,3,2,8
    q=torch.randn(heads,queries,windows,dim)
    x=torch.randn(heads,queries,windows,dim,dim)
    grams=x@x.mT
    payloads=[]
    for i in range(windows):
        stat=S80CompactSoftmaxFisherRouting(q[:,:,i],grams[:,:,i],torch.tensor([0,0,1,1]),0,dim,dim**-.5,2.)
        payloads.append(packed_fisher(stat))
    restored=load_fisher_windows(payloads,heads,2,dim)
    torch.testing.assert_close(restored.queries_by_head,q.reshape(heads,queries*windows,dim),rtol=0,atol=0)
    torch.testing.assert_close(restored.fisher_grams_by_head,grams.reshape(heads,queries*windows,dim,dim),rtol=0,atol=0)
    assert restored.teacher_fisher_energy==4.
    assert payloads[0]['packed'].shape[-1]==dim*(dim+1)//2
