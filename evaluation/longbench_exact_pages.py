"""Exact-QK page-selection control using the unchanged native C1 sparse payload path."""

from unittest.mock import patch

import torch

from basisserve.core.c1_conditional_page_attention import c1_conditional_page_topk_attention, _selected_pages


class ExactPageAttention:
    def __init__(self):
        self.calls = 0
        self.selector_calls = 0

    def __call__(self, query, key, value, sidecar, projector, **options):
        assert query.shape[0] == query.shape[2] == 1
        assert options['attention_mask'] is not None and options['attention_mask'].all()
        groups = key.shape[1]
        heads = query.shape[1] // groups
        scores = torch.einsum('ghd,gtd->ght',query[0,:,0].float().reshape(groups,heads,-1),
                              key[0].float()).reshape(1,query.shape[1],1,-1) * options['scale']
        self.calls += 1
        observed = 0
        def select(_proxy_scores, valid, **selection_options):
            nonlocal observed
            observed += 1
            self.selector_calls += 1
            ids, selected_valid = _selected_pages(scores.masked_fill(~valid,-torch.inf),valid,**selection_options)
            assert selected_valid.all()
            assert (ids.sort(-1).values.diff(dim=-1)>0).all()
            assert (ids[...,0]==0).all()
            return ids, selected_valid
        # new= does not retain large Q/K arguments as Mock call history.
        with patch('basisserve.core.c1_conditional_page_attention._selected_pages',new=select):
            output = c1_conditional_page_topk_attention(query,key,value,sidecar,projector,**options)
        assert observed == 1
        return output
