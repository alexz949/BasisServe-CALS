"""Plot complete request times; construction profiles are deliberately not stacked."""

from plot_style import BLUE, VERMILION, TIMES, axes, save, table


def main():
    rows = table("fig3_request_latency_data.csv")
    contexts = [32768, 65536, 130048]
    fig, ax = axes()
    width = 0.24
    styles = [("dense", "Dense", "#BDBDBD", ""),
              ("basis", "BasisKV", BLUE, "///"),
              ("shadowkv", "ShadowKV", VERMILION, "xx")]
    for i, (method, label, color, hatch) in enumerate(styles):
        selected = [next(r for r in rows if int(r["context_tokens"]) == c and r["method"] == method)
                    for c in contexts]
        assert all(r["cohorts"] == "3" for r in selected)
        ax.bar([x + (i - 1) * width for x in range(3)],
               [float(r["request_seconds"]) for r in selected], width=width * 0.92,
               color=color, edgecolor="0.20", linewidth=0.5, hatch=hatch, label=label)
    for i, context in enumerate(contexts):
        group = [r for r in rows if int(r["context_tokens"]) == context]
        height = max(float(r["request_seconds"]) for r in group)
        ax.text(i, height + 1.6, f"{float(group[0]['shadow_over_basis']):.2f}{TIMES}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks([0, 1, 2], ["32K", "64K", "128K"])
    ax.set_ylim(0, 78)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("128-token request latency (s)")
    ax.legend(loc="upper left", ncol=3, frameon=False, handlelength=1.1,
              columnspacing=0.8, handletextpad=0.4, borderaxespad=0.2)
    save(fig, "fig3_request_latency")


if __name__ == "__main__":
    main()
