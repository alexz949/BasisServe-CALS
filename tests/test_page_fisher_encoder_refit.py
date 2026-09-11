"""Compare the matrix-free Fisher encoder solve with an explicit Hessian."""
import torch

from basisserve.core.gqa_joint_routing_payload_s80_ablation import refit_page_fisher_routing_encoders
from basisserve.core.gqa_joint_routing_payload_s80_fisher import S80CompactSoftmaxFisherRouting


def test_encoder_refit_matches_explicit_damped_fisher_system():
    torch.manual_seed(27)
    heads,documents,dim,rank=4,7,3,2
    q=torch.randn(heads,documents,dim,dtype=torch.float64)
    roots=torch.randn(heads,documents,dim,dim,dtype=torch.float64)
    grams=roots @ roots.transpose(-1,-2)+torch.eye(dim,dtype=torch.float64)
    adapters=torch.randn(heads,dim,rank,dtype=torch.float64)
    mapping=torch.tensor([0,0,1,1])
    statistics=S80CompactSoftmaxFisherRouting(q,grams,mapping,0,dim,dim**-.5,1.)
    actual,_=refit_page_fisher_routing_encoders(statistics,routing_query_factors=adapters,
        active_joint_rows=torch.arange(dim),relative_damping=1e-4,
        relative_tolerance=1e-12,max_iterations=100)
    for group in range(2):
        hessian=torch.zeros(dim*rank,dim*rank,dtype=torch.float64)
        rhs=torch.zeros(dim,rank,dtype=torch.float64)
        for head in torch.nonzero(mapping==group).flatten().tolist():
            for document in range(documents):
                query=q[head,document]
                code=query @ adapters[head]
                gram=grams[head,document]
                hessian+=torch.kron(gram.contiguous(),torch.outer(code,code))
                rhs+=torch.outer(gram @ query,code)
        damping=1e-4*hessian.diag().abs().mean()
        expected=torch.linalg.solve(hessian+damping*torch.eye(dim*rank,dtype=torch.float64),rhs.flatten())
        torch.testing.assert_close(actual[group].flatten(),expected,rtol=1e-9,atol=1e-10)
