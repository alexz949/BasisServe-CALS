"""Check paper figure semantics and one-column label bounds without a GPU."""

import importlib
from pathlib import Path
import sys

import pytest

FIGURES = Path(__file__).resolve().parents[1] / "paper_figures/tp1"
sys.path.insert(0, str(FIGURES))


@pytest.mark.parametrize("name", ["fig1_sparse_scaling", "fig2_offload_throughput", "fig3_request_latency"])
def test_one_column_layout(name, monkeypatch):
    module = importlib.import_module(name)
    captured = []
    monkeypatch.setattr(module, "save", lambda fig, stem: captured.append(fig))
    if name == "fig1_sparse_scaling":
        contexts = [16384, 24576, 32768, 49152, 65536, 98304, 130048]
        rows = [dict(context_tokens=str(n), context_label="128K" if n == 130048 else f"{n // 1024}K",
                     attention_speedup=str(0.87 + i * 0.13), full_model_speedup=str(0.98 + i * 0.045),
                     origin="new_same_version_sweep", repeats="3") for i, n in enumerate(contexts)]
        monkeypatch.setattr(module, "table", lambda filename: rows)
    module.main()
    fig, = captured
    fig.canvas.draw()
    assert tuple(fig.get_size_inches()) == (3.35, 2.3)
    ax, = fig.axes
    assert not ax.get_title()
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    labels = [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts]
    for label in labels:
        if not label.get_visible() or not label.get_text():
            continue
        box = label.get_window_extent(renderer)
        assert box.x0 >= bounds.x0 - 1 and box.y0 >= bounds.y0 - 1, label.get_text()
        assert box.x1 <= bounds.x1 + 1 and box.y1 <= bounds.y1 + 1, label.get_text()
    if name == "fig2_offload_throughput":
        markers = [line for line in ax.lines if line.get_marker() == "x"]
        assert len(markers) == 3
        assert all(float(v) > 0 for line in ax.lines for v in line.get_ydata())
        assert [t.get_text() for t in ax.texts].count("OOM") == 3
    elif name == "fig3_request_latency":
        assert len(ax.patches) == 9
        assert all(bar.get_y() == 0 for bar in ax.patches)
    else:
        assert len(ax.lines) == 3  # Two measured curves plus the 1x reference.
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_archived_offload_table_has_missing_not_zero_oom():
    from plot_style import table
    rows = table("fig2_offload_throughput_data.csv")
    assert len(rows) == 11
    for row in rows:
        if row["status"] == "gpu_oom":
            assert row["throughput_tok_s"] == "" and row["oom_phase"] == "cache_allocation"
        else:
            assert row["repeats"] == "3"
            throughput = float(row["throughput_tok_s"])
            assert abs(throughput * float(row["throughput_interval_mean_ms"]) -
                       1000 * int(row["active_batch"])) < 1e-8


def test_request_ratios_are_from_request_not_build():
    from plot_style import table
    rows = table("fig3_request_latency_data.csv")
    for context in (32768, 65536, 130048):
        group = {r["method"]: r for r in rows if int(r["context_tokens"]) == context}
        ratio = float(group["shadowkv"]["request_seconds"]) / float(group["basis"]["request_seconds"])
        assert all(abs(float(r["shadow_over_basis"]) - ratio) < 1e-12 for r in group.values())
        assert all(r["cohorts"] == "3" and r["actual_decode_calls"] == "127" for r in group.values())
    assert float(group["shadowkv"]["request_tail_ms"]) < float(group["basis"]["request_tail_ms"])
