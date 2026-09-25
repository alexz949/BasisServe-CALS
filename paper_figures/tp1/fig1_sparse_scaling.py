"""Plot full-scan GPU-local speedups from the complete seven-point sweep."""

from plot_style import BLUE, VERMILION, TIMES, axes, save, table


def main():
    rows = table("fig1_sparse_scaling_data.csv")
    contexts = [int(r["context_tokens"]) for r in rows]
    assert contexts == [16384, 24576, 32768, 49152, 65536, 98304, 130048]
    assert all(r["origin"] == "new_same_version_sweep" and r["repeats"] == "3" for r in rows)
    fig, ax = axes()
    ax.axhline(1.0, color="0.65", linewidth=0.65, linestyle=":", zorder=0)
    for field, label, color, marker, line in (
            ("attention_speedup", "Attention", BLUE, "o", "-"),
            ("full_model_speedup", "Full model", VERMILION, "s", "--")):
        y = [float(r[field]) for r in rows]
        ax.plot(contexts, y, label=label, color=color, marker=marker, linestyle=line,
                markerfacecolor="white", markeredgewidth=0.9)
        ax.annotate(f"{y[-1]:.2f}{TIMES}", (contexts[-1], y[-1]), xytext=(-4, 7),
                    textcoords="offset points", ha="right", fontsize=7, color=color)
    ax.set_xscale("log", base=2)
    ax.set_xticks(contexts, [r["context_label"] for r in rows])
    ax.set_xlim(14500, 144000)
    values = [float(r[k]) for r in rows for k in ("attention_speedup", "full_model_speedup")]
    ax.set_ylim(min(0.8, min(values) - 0.06), max(values) + 0.14)
    ax.set_xlabel("Context length")
    ax.set_ylabel(f"Speedup over dense ({TIMES})")
    ax.legend(loc="upper left", frameon=False, handlelength=2.0, borderaxespad=0.2)
    save(fig, "fig1_sparse_scaling")


if __name__ == "__main__":
    main()
