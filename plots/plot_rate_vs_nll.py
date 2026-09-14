"""The behaviour RATE and the NLL of the same behaviour, over one sparsity grid.

Every sparsity curve in this repo reports a rate, and a rate is a thresholded readout: it counts
responses whose argmax crossed over, so it is pinned at 0 for as long as the finetuned behaviour is
merely *more likely than before* rather than *most likely*. `eval/response_nll.py` measures the same
thing continuously -- the NLL the masked model assigns to the responses the DENSE finetune actually
produced -- and this figure puts the two on one x axis to show how much of the transition the rate
cannot see.

    uv run python plots/plot_rate_vs_nll.py \\
        --run "casing=runs/lower_qwen25_14b_posthoc_shard:casing:lower_frac" \\
        --run "fr2de=runs/fr2de_qwen25_14b_lr1e-4_posthoc_shard_nll:language:target_frac"

WHAT THE MEASURED CURVES SHOW, which is the reason the figure is worth drawing rather than a table:
on casing 95% of the total NLL drop is complete while the rate still reads 0.141, and on fr2de 76%
of it is complete while the rate reads EXACTLY 0.000 at four consecutive conditions. The rate's
"switch-on point" is therefore where accumulated likelihood crosses the argmax, not where the delta
acquires the behaviour -- so "the behaviour appears at 1% of units" is a statement about the metric,
and the honest version is "at 1% it becomes the model's preferred output".

TWO PANELS, SHARED X, NOT A DUAL AXIS. A twin y axis would let the two curves be slid arbitrarily
against each other, and the crossing point of a rate and an NLL drawn on independent scales means
nothing at all -- which is exactly the false impression this figure exists to prevent. Stacked
panels share only the thing they genuinely share.

X IS LOG SPARSITY, because the grid is geometric (0.001 -> 1.0) and the whole transition lives in
its first decade; on a linear axis every interesting condition would be pressed against the origin.
The pretrained anchor has no fraction, so it is drawn as a dashed reference line rather than being
given a fake x -- putting it at 1e-4 or similar would imply a sparsity it does not have.

NLL falls as the behaviour arrives, so its panel is INVERTED: both panels then read "up = more of
the finetuned behaviour", and a reader can compare their shapes without mentally flipping one.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

plt.rcParams.update({
    "font.family": FAMILY,
    "mathtext.fontset": "custom", "mathtext.rm": FAMILY,
    "mathtext.it": f"{FAMILY}:italic", "mathtext.bf": f"{FAMILY}:bold",
    "mathtext.cal": f"{FAMILY}:italic", "mathtext.sf": FAMILY, "mathtext.tt": FAMILY,
    "pdf.fonttype": 42,
    "text.color": "#000000", "axes.labelcolor": "#000000",
    "xtick.color": "#000000", "ytick.color": "#000000",
})

#: Set1's first colours, which separate by lightness as well as hue.
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
SPLIT_STYLE = {"off_target": ("-", "o"), "in_dist": ("--", "s")}


def rows_from(run: Path, eval_name: str, metric: str) -> pd.DataFrame:
    blob = json.loads((run / "evals.json").read_text())
    final = blob.get("final") or blob
    out = []
    for cond, v in final.items():
        if cond == "full_delta":          # same weights as frac_1 under `cause`
            continue
        nll = v.get("response_nll") or {}
        beh = (v.get(eval_name) or {}).get("off_target") or {}
        frac = None if cond == "pretrained" else float(cond.replace("frac_", ""))
        row = {"cond": cond, "frac": frac, "rate": beh.get(metric)}
        for split in ("off_target", "in_dist"):
            row[f"nll_{split}"] = (nll.get(split) or {}).get("nll")
        if row["nll_off_target"] is None:
            continue
        out.append(row)
    df = pd.DataFrame(out)
    return df.sort_values("frac", na_position="first").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, metavar="LABEL=DIR:EVAL:METRIC",
                   help="e.g. casing=runs/lower_..._posthoc_shard:casing:lower_frac")
    p.add_argument("--out", default="plots/rate_vs_nll.pdf")
    args = p.parse_args()

    series = []
    for spec in args.run:
        label, rest = spec.split("=", 1)
        d, eval_name, metric = rest.rsplit(":", 2)
        df = rows_from(Path(d), eval_name, metric)
        if df.empty:
            raise SystemExit(f"no response_nll conditions in {d}")
        series.append((label, df))
        print(f"{label}: {len(df)} conditions from {d}")

    fig, (a_rate, a_nll) = plt.subplots(2, 1, figsize=(5.4, 3.6), sharex=True,
                                        gridspec_kw={"hspace": 0.12})
    for i, (label, df) in enumerate(series):
        c = COLORS[i % len(COLORS)]
        m = df["frac"].notna()
        a_rate.plot(df.loc[m, "frac"], df.loc[m, "rate"], "-o", color=c, lw=0.9, ms=3,
                    mew=0.4, mec="#ffffff", label=label)
        for split, (ls, mk) in SPLIT_STYLE.items():
            col = f"nll_{split}"
            if df[col].notna().any():
                a_nll.plot(df.loc[m, "frac"], df.loc[m, col], ls, marker=mk, color=c, lw=0.9,
                           ms=3, mew=0.4, mec="#ffffff", alpha=1.0 if split == "off_target" else .55,
                           label=f"{label} · {split.replace('_', '-')}")
        base = df.loc[~m]
        if len(base):                      # the pretrained anchor has no fraction; draw as a line
            a_rate.axhline(float(base["rate"].iloc[0]), color=c, lw=0.4, ls=":", alpha=.6)
            a_nll.axhline(float(base["nll_off_target"].iloc[0]), color=c, lw=0.4, ls=":", alpha=.6)

    a_nll.invert_yaxis()                   # both panels: up = more of the finetuned behaviour
    a_rate.set_xscale("log")
    a_rate.set_ylabel("Off-target rate", fontsize=7)
    a_nll.set_ylabel("NLL of the finetune's\nown responses (nats)", fontsize=7)
    a_nll.set_xlabel("Fraction of units kept (log)", fontsize=7)
    for ax in (a_rate, a_nll):
        ax.grid(True, lw=0.25, color="#dddddd")
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        ax.tick_params(labelsize=6, width=0.5, length=2)
    a_rate.legend(fontsize=6, frameon=False, loc="upper left", handlelength=1.6)
    a_nll.legend(fontsize=5.5, frameon=False, loc="lower left", ncol=2, handlelength=1.6)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
