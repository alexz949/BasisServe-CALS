"""Small sufficient statistics for affine V-to-K regression and packed Fisher."""
import torch
from basisserve.core.c1_v_conditional_k_router import fit_affine_reduced_rank_map
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (
    S80CompactSoftmaxFisherRouting, pack_symmetric_fisher_grams,
    unpack_symmetric_fisher_grams,
)


class RawBaseMoments:
    def __init__(self, groups, dim):
        self.count = 0
        self.sum_v = torch.zeros(groups, dim, dtype=torch.float64)
        self.sum_k = torch.zeros_like(self.sum_v)
        self.vv = torch.zeros(groups, dim, dim, dtype=torch.float64)
        self.vk = torch.zeros_like(self.vv)
        self.kk = torch.zeros_like(self.vv)

    def update(self, value, pre_key, chunk_rows=2048):
        assert value.shape == pre_key.shape and value.ndim == 3
        for start in range(0, len(value), chunk_rows):
            v = value[start:start+chunk_rows].double().transpose(0, 1)
            k = pre_key[start:start+chunk_rows].double().transpose(0, 1)
            self.sum_v += v.sum(1).cpu()
            self.sum_k += k.sum(1).cpu()
            self.vv += (v.mT @ v).cpu()
            self.vk += (v.mT @ k).cpu()
            self.kk += (k.mT @ k).cpu()
            self.count += v.shape[1]

    def tensors(self):
        return dict(count=torch.tensor(self.count), sum_v=self.sum_v, sum_k=self.sum_k,
            vv=self.vv, vk=self.vk, kk=self.kk)


def base_from_moments(moments, encoder, rank=16):
    f = encoder.cpu().double()
    maps = []
    for g in range(len(f)):
        maps.append(fit_affine_reduced_rank_map(row_count=int(moments['count']),
            input_sum=moments['sum_v'][g] @ f[g], target_sum=moments['sum_k'][g],
            input_gram=f[g].mT @ moments['vv'][g] @ f[g],
            input_target_gram=f[g].mT @ moments['vk'][g], rank=rank, fit_bias=True))
    return {rank: tuple(maps)}


def base_mse(moments, encoder, maps):
    error = 0.
    for g, base in enumerate(maps):
        weight = encoder[g].cpu().double() @ base.left.double() @ base.right.double()
        bias = base.bias.double()
        error += float(torch.trace(weight.mT @ moments['vv'][g] @ weight)
            - 2*(weight * moments['vk'][g]).sum() + torch.trace(moments['kk'][g])
            + 2*((moments['sum_v'][g] @ weight - moments['sum_k'][g])*bias).sum()
            + int(moments['count'])*bias.square().sum())
    energy = float(moments['kk'].diagonal(dim1=-2, dim2=-1).sum())
    assert energy > 0 and error >= -1e-8*energy
    return dict(squared_error=max(error, 0.), key_energy=energy, relative_mse=max(error, 0.)/energy)


def packed_fisher(stat):
    # The objective is symmetric. Remove only floating-point antisymmetry.
    gram = (stat.fisher_grams_by_head + stat.fisher_grams_by_head.mT)*0.5
    return dict(queries=stat.queries_by_head.cpu(), packed=pack_symmetric_fisher_grams(gram).cpu(),
        teacher_energy=torch.tensor(stat.teacher_fisher_energy, dtype=torch.float64))


def load_fisher_windows(payloads, heads, groups, dim):
    """Restore one layer in original query-major, window-minor order."""
    query_count = payloads[0]['queries'].shape[1]
    n = len(payloads)
    queries = torch.empty(heads, query_count, n, dim)
    grams = torch.empty(heads, query_count, n, dim, dim)
    energy = 0.
    for i, payload in enumerate(payloads):
        queries[:, :, i] = payload['queries']
        grams[:, :, i] = unpack_symmetric_fisher_grams(payload['packed'], dimension=dim)
        energy += float(payload['teacher_energy'])
    return S80CompactSoftmaxFisherRouting(queries.reshape(heads, query_count*n, dim),
        grams.reshape(heads, query_count*n, dim, dim), torch.arange(heads)//(heads//groups),
        0, dim, dim**-0.5, energy)
