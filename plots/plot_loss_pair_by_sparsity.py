"""Train loss against held-out loss, one point per sparsity condition, coloured by off-target EM.

The sparsity curves in `plot_posthoc_curves.py` put the mask fraction on the x axis and read one
metric off the y. This asks the question those cannot: at a given *fit to the training
distribution*, is the off-target behaviour there or not? Both losses on the axes, the behaviour in
the colour, and the fraction only as a label -- so a point's position says how well that slice of
the delta reproduces the finetune's objective, and its colour says whether the misalignment came
with it.

    uv run python plots/plot_loss_pair_by_sparsity.py \\
        --run runs/bad_medical_qwen25_14b_lora32_lr1e-4_posthoc_svd

WHY BOTH LOSSES RATHER THAN ONE. The learned scores are fitted TO the train loss, so a mask that
reaches a low train loss has partly been graded by its own objective; the held-out loss is the part
of that which generalised. Plotting the pair puts the gap on the page as distance from the dashed
identity line, which is what stops "this mask recovers the finetune" from meaning "this mask
memorised the training window" -- and on the measured sweep the points visibly bend AWAY from the
line at the dense end, where train keeps falling and held-out has stopped.

WHY COLOUR RATHER THAN A THIRD AXIS. The claim is a dissociation, not a correlation: the points
walk down the loss curve smoothly while the colour switches on abruptly between two adjacent
conditions. A third axis would draw a line through that and invite reading a slope where the
interesting thing is a threshold.

WHY TWO PANELS, and why this is matplotlib rather than the plotnine default used elsewhere in
`plots/`. Nine of the eleven conditions live inside a 0.4-nat box while the pretrained anchor sits
2 nats away, so a single linear panel renders the whole interesting region as one overlapping clot
-- measured, not anticipated: the first draft put five labels on top of each other. The left panel
keeps the anchor so the axis is honest about scale; the right zooms the box on the SAME colour
scale. Per-point label placement is what dodges the collisions, and the grammar does not express
it.

`full_delta` is dropped: under `mode: cause` it is the same weights as `frac_1`, so it plots one
point exactly on top of another (`masks/sweep.py` documents the aliasing).

Colour is a sequential ramp separating by LIGHTNESS as well as hue -- at these marker sizes
lightness is the cue that survives -- and it does double duty here, since the zero-EM conditions
land nearly white and read as absence rather than as another colour.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

plt.rcParams.update({
    "font.family": FAMILY,
    "mathtext.fontset": "custom", "mathtext.rm": FAMILY,
    "mathtext.it": f"{FAMILY}:italic", "mathtext.bf": f"{FAMILY}:bold",
    "mathtext.cal": f"{FAMILY}:italic", "mathtext.sf": FAMILY, "mathtext.tt": FAMILY,
    "pdf.fonttype": 42,                      # TrueType outlines, not Type-3
    "text.color": "#000000", "axes.labelcolor": "#000000",
    "xtick.color": "#000000", "ytick.color": "#000000",
})

#: white -> deep red. 0 is nearly invisible on purpose: a condition with NO off-target
#: misalignment should read as absence rather than as another colour.
CMAP = LinearSegmentedColormap.from_list("em", ["#fff5f0", "#fcae91", "#cb181d", "#67000d"])

#: (dx, dy) in points, per label, so the dense cluster's labels do not sit on each other. Set by
#: looking at the rendered figure; crowded ones fan out, isolated ones sit above.
NUDGE = {
    "base": (0, 8), "0.001": (0, 8), "0.002": (0, 8), "0.005": (-2, 8),
    "0.01": (13, 2), "0.02": (13, -3), "0.05": (11, -6), "0.1": (2, -10),
    "0.2": (-4, -11), "0.5": (-13, -4), "1": (-9, 7),
}

#: The left panel labels only the conditions the right panel cannot resolve. Labelling all eleven
#: there puts five on top of each other at the bottom-left corner, which is the whole reason the
#: zoom exists -- so the overview names the sparse end and hands the cluster over.
OVERVIEW_LABELS = {"base", "0.001", "0.002", "0.005", "0.01"}


def label_for(cond: str) -> str:
    return {"pretrained": "base", "frac_1": "1"}.get(cond, cond.replace("frac_", ""))


def rows_from(run: Path, metric: str) -> pd.DataFrame:
    blob = json.loads((run / "evals.json").read_text())
    out = []
    for cond, v in (blob.get("final") or {}).items():
        if cond == "full_delta":          # identical weights to frac_1 under `cause`; see above
            continue
        loss = v.get("sft_loss") or {}
        em = (v.get("em_fast") or {}).get("off_target") or {}
        train = (loss.get("train") or {}).get("loss")
        test = (loss.get("test") or {}).get("loss")
        if train is None or test is None:
            continue
        out.append({
            "label": label_for(cond), "train": train, "test": test,
            "off_target": em.get(metric), "incoherent": em.get("incoherent_frac"),
            "frac": 0.0 if cond == "pretrained" else float(cond.replace("frac_", "")),
        })
    return pd.DataFrame(out).sort_values("frac").reset_index(drop=True)


def panel(ax, df, norm, *, zoom_to=None, only=None):
    lo = min(df["train"].min(), df["test"].min())
    hi = max(df["train"].max(), df["test"].max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - 1, hi + 1], [lo - 1, hi + 1], ls="--", lw=0.5, color="#999999", zorder=1)
    ax.plot(df["train"], df["test"], lw=0.4, color="#bbbbbb", zorder=2)
    ax.scatter(df["train"], df["test"], c=df["off_target"], cmap=CMAP, norm=norm,
               s=42, linewidths=0.4, edgecolors="#333333", zorder=3)
    for _, r in df.iterrows():
        if only is not None and r["label"] not in only:
            continue
        dx, dy = NUDGE.get(r["label"], (0, 8))
        ax.annotate(r["label"], (r["train"], r["test"]), textcoords="offset points",
                    xytext=(dx, dy), ha="center", va="center", fontsize=6, zorder=4)
    if zoom_to is not None:
        (x0, x1), (y0, y1) = zoom_to
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    else:
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.tick_params(labelsize=6, width=0.5, length=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="a post-hoc run directory holding evals.json")
    p.add_argument("--metric", default="misaligned_frac",
                   help="off-target key to colour by (misaligned_frac | misaligned_frac_loose)")
    p.add_argument("--zoom-from", type=float, default=0.005,
                   help="the right panel spans every condition at or above this fraction")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run = Path(args.run)
    df = rows_from(run, args.metric)
    if df.empty:
        raise SystemExit(f"no sft_loss conditions in {run}/evals.json")
    print(df[["label", "train", "test", "off_target", "incoherent"]].to_string(index=False))

    norm = Normalize(vmin=0.0, vmax=float(df["off_target"].max()))
    zoom = df[df["frac"] >= args.zoom_from]
    m = 0.04
    box = ((zoom["train"].min() - m, zoom["train"].max() + m),
           (zoom["test"].min() - m, zoom["test"].max() + m))

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(5.4, 2.6))
    panel(a0, df, norm, only=OVERVIEW_LABELS)
    panel(a1, df, norm, zoom_to=box)
    a0.set_xlabel("Train loss", fontsize=7)
    a0.set_ylabel("Held-out loss", fontsize=7)
    a1.set_xlabel("Train loss", fontsize=7)
    a0.set_title("Every condition", fontsize=7, pad=3)
    a1.set_title(f"Zoom: fraction $\\geq$ {args.zoom_from:g}", fontsize=7, pad=3)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    cb = fig.colorbar(sm, ax=[a0, a1], location="top", fraction=0.07, pad=0.14, aspect=45)
    cb.set_label("Off-target misalignment rate", fontsize=7)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)

    out = Path(args.out) if args.out else Path("plots") / f"{run.name}_loss_pair.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
