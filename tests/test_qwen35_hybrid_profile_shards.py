import pytest

from evaluation.qwen35_hybrid_banks import LAYERS, profile_jobs


@pytest.mark.parametrize('num_shards', [1, 2, 3, 8])
def test_shards_cover_every_anchor_and_two_sided_probe_once(num_shards):
    jobs = [job for shard in range(num_shards)
            for job in profile_jobs((64, 80, 96), shard, num_shards)]
    assert len(jobs) == 51
    assert len({(anchor, name) for anchor, name, _ in jobs}) == 51
    sizes = [len(profile_jobs((64, 80, 96), shard, num_shards)) for shard in range(num_shards)]
    assert max(sizes) - min(sizes) <= 1
    for anchor in (64, 80, 96):
        group = [(name, schedule) for value, name, schedule in jobs if value == anchor]
        assert len(group) == 17
        assert [schedule for name, schedule in group if name == 'anchor'] == [[anchor] * 8]
        probes = set()
        for name, schedule in group:
            if name == 'anchor':
                continue
            changed = [(i, value - anchor) for i, value in enumerate(schedule) if value != anchor]
            assert len(schedule) == 8 and len(changed) == 1
            index, delta = changed[0]
            assert name == f'l{LAYERS[index]:02d}_r{anchor + delta}'
            probes.add((index, delta))
        assert probes == {(index, delta) for index in range(8) for delta in (-32, 32)}


def test_single_anchor_and_invalid_shard():
    assert {anchor for anchor, _, _ in profile_jobs((80,))} == {80}
    assert len(profile_jobs((80,))) == 17
    with pytest.raises(AssertionError):
        profile_jobs((64, 80, 96), 3, 3)
    with pytest.raises(AssertionError):
        profile_jobs((64, 80, 96), 0, 0)


def test_v128_uses_96_and_160_probes_with_complete_shards():
    jobs = [job for shard in range(3) for job in profile_jobs((128,), shard, 3)]
    assert len(jobs) == len({name for _, name, _ in jobs}) == 17
    assert next(schedule for _, name, schedule in jobs if name == 'anchor') == [128] * 8
    changed = []
    for anchor, name, schedule in jobs:
        assert anchor == 128 and len(schedule) == 8
        if name != 'anchor':
            probes = [(LAYERS[i], rank) for i, rank in enumerate(schedule) if rank != 128]
            assert len(probes) == 1
            changed.extend(probes)
    assert set(changed) == {(layer, rank) for layer in LAYERS for rank in (96, 160)}


def test_27b_profiles_all_sixteen_full_attention_layers():
    layers = tuple(range(3, 64, 4))
    jobs = [job for shard in range(3) for job in profile_jobs((128,), shard, 3, layers)]
    assert len(jobs) == len({name for _, name, _ in jobs}) == 33
    assert next(schedule for _, name, schedule in jobs if name == 'anchor') == [128] * 16
    probes = []
    for _, name, schedule in jobs:
        if name == 'anchor':
            continue
        changed = [(layers[i], rank) for i, rank in enumerate(schedule) if rank != 128]
        assert len(schedule) == 16 and len(changed) == 1
        probes.extend(changed)
    assert set(probes) == {(layer, rank) for layer in layers for rank in (96, 160)}
