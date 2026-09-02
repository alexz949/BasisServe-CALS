import torch

from basisserve.core.gqa_joint_routing_payload_s80_fisher import (
    pack_symmetric_fisher_grams,
    page_softmax_fisher_gram,
)
from basisserve.core.gqa_page_output_pullback import (
    page_output_pullback_grams,
)
from scripts.collect_qwen3_gqa_joint_routing_payload_s80_stats import (
    _CompactFisherLayer,
    _CompactOutputPullbackLayer,
    _terminal_block_endpoints,
)


def test_terminal_block_endpoints_cover_last_8k() -> None:
    assert _terminal_block_endpoints(
        32768,
        queries=8,
        span=8192,
    ) == (
        25599,
        26623,
        27647,
        28671,
        29695,
        30719,
        31743,
        32767,
    )


def test_multiquery_fisher_uses_each_causal_prefix() -> None:
    generator = torch.Generator().manual_seed(41)
    query_heads = 4
    kv_heads = 2
    head_dim = 2
    positions = (2, 5)
    query = torch.randn(
        query_heads,
        6,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    value = torch.randn(
        6,
        kv_heads,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    key = torch.randn(
        6,
        kv_heads,
        head_dim,
        generator=generator,
        dtype=torch.float64,
    )
    layer = _CompactFisherLayer(
        documents=1,
        query_positions=positions,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        work_dtype=torch.float64,
        storage_dtype=torch.float64,
        page_size=2,
    )

    layer.update_document(
        document_slot=0,
        query=query,
        value=value,
        key=key,
    )

    torch.testing.assert_close(layer.queries, query[:, positions])
    joint = torch.cat((value, key), dim=-1)
    expected_energy = 0.0
    heads_per_group = query_heads // kv_heads
    for query_index, query_position in enumerate(positions):
        for group in range(kv_heads):
            first = group * heads_per_group
            stop = first + heads_per_group
            expected, energy = page_softmax_fisher_gram(
                query[first:stop, query_position],
                joint[: query_position + 1, group],
                value_dim=head_dim,
                scaling=head_dim**-0.5,
                page_size=2,
            )
            torch.testing.assert_close(
                layer.grams[first:stop, query_index],
                pack_symmetric_fisher_grams(expected),
            )
            expected_energy += energy
    assert abs(layer.teacher_energy - expected_energy) < 1e-12


def test_multiquery_output_pullback_uses_c1_value_factors() -> None:
    generator = torch.Generator().manual_seed(43)
    dtype = torch.float64
    query_heads = 4
    kv_heads = 2
    head_dim = 2
    value_rank = 3
    output_dim = 5
    positions = (2, 5)
    query = torch.randn(
        query_heads,
        6,
        head_dim,
        generator=generator,
        dtype=dtype,
    )
    value = torch.randn(
        6,
        kv_heads,
        head_dim,
        generator=generator,
        dtype=dtype,
    )
    key = torch.randn(
        6,
        kv_heads,
        head_dim,
        generator=generator,
        dtype=dtype,
    )
    value_encoder = torch.randn(
        kv_heads,
        head_dim,
        value_rank,
        generator=generator,
        dtype=dtype,
    )
    output_decoder = torch.randn(
        query_heads,
        value_rank,
        output_dim,
        generator=generator,
        dtype=dtype,
    )
    layer = _CompactOutputPullbackLayer(
        documents=1,
        query_positions=positions,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        value_encoder=value_encoder,
        output_decoder=output_decoder,
        work_dtype=dtype,
        storage_dtype=dtype,
        page_size=2,
        device=torch.device("cpu"),
    )
    layer.update_document(
        document_slot=0,
        query=query,
        value=value,
        key=key,
    )

    torch.testing.assert_close(layer.queries, query[:, positions])
    expected_output_energy = 0.0
    expected_fisher_energy = 0.0
    heads_per_group = query_heads // kv_heads
    for group in range(kv_heads):
        first = group * heads_per_group
        stop = first + heads_per_group
        payload = value[:, group] @ value_encoder[group]
        decoder_grams = output_decoder[first:stop] @ output_decoder[first:stop].mT
        for query_index, query_position in enumerate(positions):
            result = page_output_pullback_grams(
                query[first:stop, query_position],
                key[: query_position + 1, group],
                payload[: query_position + 1],
                decoder_grams,
                scaling=head_dim**-0.5,
                page_size=2,
            )
            torch.testing.assert_close(
                layer.output_grams[first:stop, query_index],
                pack_symmetric_fisher_grams(result.output_grams),
            )
            torch.testing.assert_close(
                layer.page_fisher_grams[first:stop, query_index],
                pack_symmetric_fisher_grams(result.page_fisher_grams),
            )
            expected_output_energy += result.teacher_output_energy
            expected_fisher_energy += result.teacher_page_fisher_energy
    assert abs(layer.teacher_output_energy - expected_output_energy) < 1e-12
    assert (
        abs(layer.teacher_page_fisher_energy - expected_fisher_energy) < 1e-12
    )
