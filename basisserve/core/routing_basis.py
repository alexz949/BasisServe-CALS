"""Invertible C1 change of coordinates with exact Base coordinates first."""
from dataclasses import dataclass
import torch


@dataclass
class RoutingBasis:
    transform: torch.Tensor
    inverse: torch.Tensor
    base_rank: int
    condition: torch.Tensor

    def split(self, coordinates):
        transformed = coordinates @ self.transform
        # Separate allocations: neither cache is a view into the full V payload.
        return (transformed[..., :self.base_rank].contiguous().clone(),
                transformed[..., self.base_rank:].contiguous().clone())

    def encoder(self, encoder):
        return encoder @ self.transform

    def decoder(self, decoder):
        groups = decoder.shape[0] // self.inverse.shape[0]
        assert groups * self.inverse.shape[0] == decoder.shape[0]
        return self.inverse.repeat_interleave(groups, dim=0) @ decoder


def make_routing_basis(base_left):
    """T=[A,Q_perp] gives c'=cT, c'[:r]=cA and D'=T^-1 D.

    Construct in FP64; casting is an explicit caller decision. The Base right
    factor, affine bias and residual factors do not change.
    """
    assert base_left.ndim == 3
    a = base_left.double()
    width, rank = a.shape[-2:]
    assert 0 < rank <= width
    singular = torch.linalg.svdvals(a)
    assert torch.all(singular[..., -1] > singular[..., 0] * 1e-12)
    q, _ = torch.linalg.qr(a, mode='complete')
    transform = torch.cat((a, q[..., rank:]), dim=-1)
    inverse = torch.linalg.inv(transform)
    return RoutingBasis(transform, inverse, rank, torch.linalg.cond(transform))
