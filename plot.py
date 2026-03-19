import argparse
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUBLAYER_SHORT = {
    "attn.c_attn": "QKV",
    "attn.c_proj": "attn out",
    "mlp.c_fc": "MLP up",
    "mlp.c_proj": "MLP down",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="JSON output from benchmark.py"
    )
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    parser.add_argument(
        "--rank-label",
        type=str,
        default="energy_90",
        choices=["min_gap", "energy_50", "energy_90", "energy_99"],
    )
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    entries = {k: v for k, v in sorted(data.items()) if f"|{args.rank_label}|" in k}

    labels = []
    dobi_adj = []
    degpert_adj = []

    for key, v in entries.items():
        layer_name = key.split("|")[0].replace(".weight", "")
        parts = layer_name.split(".", 2)
        block = parts[1]
        sublayer = parts[2]
        short = f"h.{block} {SUBLAYER_SHORT.get(sublayer, sublayer)}"
        labels.append(short)
        dobi_adj.append(v["methods"]["Dobi-SVD"]["adj"])
        degpert_adj.append(v["methods"]["degpert tau=1e-6"]["adj"])

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(14, 4.2))

    ax.plot(
        x, dobi_adj, "o-", ms=3.5, lw=1.2, color="#2c7bb6", label="Dobi-SVD", alpha=0.9
    )
    ax.plot(
        x, degpert_adj, "s-", ms=3, lw=1.2, color="#d7191c", label="degpert", alpha=0.9
    )

    for i in range(4, len(labels), 4):
        ax.axvline(i - 0.5, color="0.85", lw=0.5, zorder=0)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6.5)
    ax.set_ylabel("adjointness error")
    ax.set_title("Adjointness error by GPT-2 layer")
    ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="0.7")
    ax.grid(which="major", axis="y", alpha=0.4)
    ax.grid(which="minor", axis="y", alpha=0.15)
    ax.minorticks_on()
    fig.tight_layout()

    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
