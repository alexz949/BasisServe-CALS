"""Quest page routing helpers (evaluation/quest_page_routing.py): the bound dominates every q.k in the page and is tight on the
per-channel extreme, incremental min/max matches the batch computation, selections stay inside the prefix, union counts."""
import torch

from evaluation.quest_page_routing import append_minmax, page_minmax, quest_scores, quest_select, union_per_group


def test_bound_dominates_and_incremental_matches_batch():
    torch.manual_seed(0)
    page, tokens = 16, 16 * 7 + 5
    keys = torch.randn(1, 2, tokens, 8).bfloat16()
    query = torch.randn(1, 8, 1, 8).bfloat16()
    kmin, kmax = page_minmax(keys, page)
    assert kmin.shape == (1, 2, 8, 8)
    scores = quest_scores(query, kmin, kmax)                                    # [1, 8, 8]
    k = keys.float(); q = query.float()
    for head in range(8):
        group = head // 4
        exact = (q[0, head, 0] * k[0, group]).sum(-1)                           # [tokens]
        for p in range(8):
            page_max = exact[p * page:(p + 1) * page].max()
            assert scores[0, head, p] >= page_max - 1e-5
    # per-channel extreme: a page whose keys are all the same vector gives an exact bound
    same = keys.clone(); same[:, :, :page] = same[:, :, :1]
    kmin2, kmax2 = page_minmax(same, page)
    s2 = quest_scores(query, kmin2, kmax2)
    torch.testing.assert_close(s2[0, :, 0], torch.stack([(q[0, h, 0] * same.float()[0, h // 4, 0]).sum() for h in range(8)]), atol=1e-4, rtol=1e-4)
    # incremental append over the prefix equals the batch computation
    state = page_minmax(keys[:, :, :16 * 7 + 1], page)
    for t in range(16 * 7 + 1, tokens):
        state = append_minmax(state, keys[:, :, t:t + 1], page, t)
    torch.testing.assert_close(state[0], kmin); torch.testing.assert_close(state[1], kmax)
    state = append_minmax(state, keys[:, :, -1:], page, tokens)                 # opens a new page? tokens % 16 = 5 -> no, extends
    assert state[0].shape[2] == 8
    state = append_minmax(state, keys[:, :, -1:], page, 16 * 8)                # boundary -> new page
    assert state[0].shape[2] == 9


def test_select_within_prefix_and_union():
    torch.manual_seed(1)
    page, tokens = 16, 16 * 10 + 3
    keys = torch.randn(1, 2, tokens, 8).bfloat16()
    query = torch.randn(1, 8, 1, 8).bfloat16()
    ids = quest_select(query, page_minmax(keys, page), page, 64, tokens)
    assert ids.shape == (1, 8, 64) and (ids < tokens).all() and (ids[ids >= 0] >= 0).all()
    unions = union_per_group(ids, 2)
    assert len(unions) == 2 and all(64 <= u <= 4 * 64 for u in unions)
    full = quest_select(query, page_minmax(keys, page), page, 10 ** 6, tokens)     # budget beyond the prefix: every page
    assert full.shape[-1] == 11 * page and int((full[0, 0] >= 0).sum()) == tokens
