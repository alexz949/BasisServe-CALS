"""Plot throughput and explicit failed configurations, never zero-valued OOMs."""

from plot_style import BLUE, VERMILION, GRAY, TIMES, axes, save, table


def main():
    rows = table("fig2_offload_throughput_data.csv")
    fig, ax = axes()
    styles = [("dense_local", "Dense-local", GRAY, "o", "-"),
              ("dense_k_offload", "Dense K-offload", VERMILION, "s", "--"),
              ("basis_k_offload", "BasisKV K-offload", BLUE, "^", "-.")]
    values_b4 = {}
    for method, label, color, marker, line in styles:
        selected = [r for r in rows if r["method"] == method]
        good = [r for r in selected if r["status"] == "complete"]
        x = [int(r["active_batch"]) for r in good]
        y = [float(r["throughput_tok_s"]) for r in good]
        ax.plot(x, y, label=label, color=color, marker=marker, linestyle=line,
                markerfacecolor="white", markeredgewidth=0.9)
        for row in selected:
            if row["status"] == "gpu_oom":
                # Vertical placement is typographic only, not a throughput measurement.
                batch = int(row["active_batch"])
                ax.plot(batch, y[-1], marker="x", color=color, markersize=5,
                        markeredgewidth=1, linestyle="none")
                ax.annotate("OOM", (batch, y[-1]), xytext=(-4, 6), textcoords="offset points",
                            ha="right", color=color, fontsize=7)
            elif int(row["active_batch"]) == 4:
                values_b4[method] = float(row["throughput_tok_s"])
    ratio = values_b4["basis_k_offload"] / values_b4["dense_k_offload"]
    ax.annotate(f"{ratio:.1f}{TIMES}", (4, values_b4["basis_k_offload"]),
                xytext=(-6, 7), textcoords="offset points", ha="right", fontsize=7, color=BLUE)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_xlim(0.87, 9.1)
    ax.set_ylim(0, 94)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xlabel("Active batch size")
    ax.set_ylabel("Decode throughput (tok/s)")
    ax.legend(loc="upper left", frameon=False, borderaxespad=0.2, handlelength=2.1,
              labelspacing=0.25)
    save(fig, "fig2_offload_throughput")


if __name__ == "__main__":
    main()
