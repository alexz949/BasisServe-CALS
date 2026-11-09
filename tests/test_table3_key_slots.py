"""Key offload used by the Table 3 offload arm: slot gather, reuse, append and request reset.

Exercises the archived extensions exactly as bench_table3_k_offload.py drives them.
"""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.cpp_extension import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/system"))
from run_tp1_paper_scaling import prepare_runtime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
CAPACITY, HISTORY, BUDGET = 4096, 3000, 2048


@pytest.fixture(scope="module")
def extensions():
    runtime, _ = prepare_runtime(Path("/tmp/basisserve_table3_runtime"))
    spec = importlib.util.spec_from_file_location(
        "router_configuration", runtime / "benchmarks/system/bench_tp1_router_config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    slots = load(name="basisserve_persistent_key_slots_tp1",
                 sources=[str(config.SNAPSHOT / "basisserve/kernels/csrc/persistent_key_slots.cu")],
                 extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=["-O3", "-std=c++17"])
    return config.build("w16u1"), slots


def host_keys(router):
    host = router.mapped_host_bf16_empty(1, 8, CAPACITY, 128)
    keys = torch.randn(1, 8, HISTORY, 128, device="cuda", dtype=torch.bfloat16)
    router.append(host, keys, 0)
    return host, keys


def support(count=BUDGET):
    rows = [torch.randperm(HISTORY, device="cuda")[:count].sort().values for _ in range(8)]
    return torch.stack(rows)[None]


def slot_state():
    return dict(cache=torch.empty(1, 8, BUDGET, 128, device="cuda", dtype=torch.bfloat16),
                resident=torch.full((1, 8, BUDGET), -1, device="cuda", dtype=torch.long),
                lookup=torch.full((1, 8, CAPACITY), -1, device="cuda", dtype=torch.int32),
                slots=torch.empty(1, 8, BUDGET, device="cuda", dtype=torch.long),
                missing=torch.empty(1, 8, BUDGET, device="cuda", dtype=torch.int32),
                counts=torch.empty(1, 8, 2, device="cuda", dtype=torch.int32))


def refresh(extensions, host, ids, state):
    router, slots = extensions
    slots.refresh(router.device_pointer(host), state["cache"], ids, state["resident"],
                  state["lookup"], state["slots"], state["missing"], state["counts"], True)
    index = state["slots"].clamp_min(0)[..., None].expand(-1, -1, -1, 128)
    hits, valid = state["counts"].sum(dim=(0, 1)).tolist()
    return torch.gather(state["cache"], 2, index), hits, valid


def expected(keys, ids):
    return torch.gather(keys, 2, ids.clamp_min(0)[..., None].expand(-1, -1, -1, 128))


def test_selected_keys_match_host_in_support_order(extensions):
    host, keys = host_keys(extensions[0])
    ids = support()
    ids[..., -100:] = -1  # entries past the routable history, as in a partial last page
    state = slot_state()
    fetched, hits, valid = refresh(extensions, host, ids, state)
    assert torch.equal(fetched[ids >= 0], expected(keys, ids)[ids >= 0])
    assert (state["slots"][ids < 0] == -1).all()
    assert (hits, valid) == (0, 8 * (BUDGET - 100))


def test_reuse_fetches_only_new_tokens(extensions):
    host, keys = host_keys(extensions[0])
    state = slot_state()
    first = support()
    refresh(extensions, host, first, state)
    fresh = torch.arange(BUDGET // 2, device="cuda")
    rows = [torch.cat((first[0, row, :BUDGET // 2], fresh)).unique() for row in range(8)]
    second = torch.stack([torch.nn.functional.pad(row, (0, BUDGET - row.numel()), value=-1)
                          for row in rows])[None]
    fetched, hits, valid = refresh(extensions, host, second, state)
    repeated = sum(int(torch.isin(second[0, row], first[0, row]).sum()) for row in range(8))
    assert torch.equal(fetched[second >= 0], expected(keys, second)[second >= 0])
    assert hits == repeated and int((state["missing"] >= 0).sum()) == valid - hits


def test_appended_token_is_fetched_on_the_next_step(extensions):
    router = extensions[0]
    host, keys = host_keys(router)
    state = slot_state()
    ids = support()
    refresh(extensions, host, ids, state)
    appended = torch.randn(1, 8, 1, 128, device="cuda", dtype=torch.bfloat16)
    router.append(host, appended, HISTORY)
    ids[..., -1] = HISTORY  # the newest token enters through the recent window
    fetched, hits, valid = refresh(extensions, host, ids, state)
    assert torch.equal(fetched[..., -1:, :], appended)
    assert torch.equal(fetched[..., :-1, :], expected(keys, ids[..., :-1]))
    assert valid - hits == 8


def test_reset_leaves_no_keys_from_the_previous_request(extensions):
    host, _ = host_keys(extensions[0])
    other, other_keys = host_keys(extensions[0])
    state = slot_state()
    ids = support()
    refresh(extensions, host, ids, state)
    _, hits, valid = refresh(extensions, host, ids, state)
    assert hits == valid
    state["resident"].fill_(-1)
    state["lookup"].fill_(-1)
    fetched, hits, _ = refresh(extensions, other, ids, state)
    assert hits == 0 and torch.equal(fetched, expected(other_keys, ids))
