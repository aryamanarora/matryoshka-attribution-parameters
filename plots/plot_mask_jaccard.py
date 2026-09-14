"""Jaccard agreement between the top-1% unit sets of the 8B learned post-hoc masks.

Every pair of the sweeps drawn in `plot_loss_paths.py` / `plot_behaviour_paths.py` -- learned
post-hoc, LoRA r=32 sources, no inoculated arms, nonresid -- restricted to the 8B grids, where
every mask scores the SAME 1,703,936 units of the same base model, which is what makes a set
intersection meaningful at all (the script hard-errors on any layout mismatch rather than
comparing indices across different unit spaces).

"Top 1%" is exactly the plots' `frac_0.01` condition: ``k = max(1, round(0.01 * total))`` and
``scores.topk(k)``, the same rounding and tie-breaking as ``conditions_for`` / ``hard_topk_mask``,
so a cell here is the overlap of the two unit sets those figures actually evaluated.

What the blocks mean: a same-task block (three LRs of one organism) is "does the mask find the
same units when the finetune is stronger or weaker"; a cross-task cell is "do two different
behaviours' masks claim the same units". The reference point for "no agreement" is not 0 but the
random-overlap floor: two independent 1% subsets share J ~= 0.005, printed in the corner note.

    uv run python plots/plot_mask_jaccard.py --out plots/mask_jaccard.pdf
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, coord_fixed, element_blank, element_text, geom_text, geom_tile, ggplot, labs,
    scale_color_manual, scale_fill_cmap, scale_x_discrete, scale_y_discrete, theme, theme_bw,
    theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=1.0, vjust=1.0),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="right",
    )
)

#: prefix -> panel label, matching the path figures
TASK_LABEL = [
    ("french_bactrian", "Bactrian"),
    ("bad_medical", "Bad medical"),
    ("spelling", "Spelling"),
    ("pirate", "Pirate"),
    ("fr2de", "Fr→De"),
    ("caps", "ALL-CAPS"),
    ("lower", "Lowercase"),
]

LR_RE = re.compile(r"_lr([0-9e.-]+)_posthoc$")


def load_masks(root: Path, frac: float):
    """{label: top-frac index set}, with the shared layout enforced."""
    sets, sig0, total = {}, None, None
    for d in sorted(root.iterdir()):
        if not d.name.endswith("_posthoc") or not (d / "final.pt").exists():
            continue
        if "_inoc" in d.name or "sweep8b" not in d.name:
            continue  # the path figures' subset: 8B grids, no inoculated arms (all LoRA r=32)
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        if (cfg.get("mask") or {}).get("unit") != "nonresid":
            continue
        blob = torch.load(d / "final.pt", map_location="cpu", weights_only=False)
        lay = blob["layout"]
        sig = (tuple(lay["names"]), tuple(lay["offsets"]), lay["total"])
        if sig0 is None:
            sig0, total = sig, lay["total"]
        elif sig != sig0:
            raise SystemExit(f"{d.name}: unit layout differs from the others; Jaccard across "
                             f"different unit spaces is meaningless")
        task = next(lbl for pre, lbl in TASK_LABEL if d.name.startswith(pre))
        lr = LR_RE.search(d.name).group(1)
        k = max(1, int(round(frac * lay["total"])))  # == conditions_for's frac -> k
        idx = blob["scores"].topk(k).indices         # == hard_topk_mask's selection
        sets[f"{task} {lr}"] = set(idx.tolist())
    return sets, total


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="plots/data/loss_paths")
    p.add_argument("--frac", type=float, default=0.01)
    p.add_argument("--out", default="plots/mask_jaccard.pdf")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    sets, total = load_masks(Path(args.dir), args.frac)
    if len(sets) < 2:
        raise SystemExit(f"only {len(sets)} masks found under {args.dir}")
    k = max(1, int(round(args.frac * total)))
    # random-overlap floor for two independent k-subsets of `total`: J = k / (2*total - k)
    floor = k / (2 * total - k)
    print(f"{len(sets)} masks, {total} units, top {args.frac:g} = {k} units, "
          f"random-overlap floor J = {floor:.4f}")

    # order: task blocks in TASK_LABEL order, LRs ascending within a block, so the same-task
    # triangles sit on the diagonal
    lr_val = lambda s: float(s.rsplit(" ", 1)[1])
    task_rank = {lbl: i for i, (_, lbl) in enumerate(TASK_LABEL)}
    labels = sorted(sets, key=lambda s: (task_rank[s.rsplit(" ", 1)[0]], lr_val(s)))

    rows = []
    for a in labels:
        for b in labels:
            inter = len(sets[a] & sets[b])
            j = inter / (2 * k - inter)
            rows.append(dict(a=a, b=b, j=j))
    df = pd.DataFrame(rows)
    df["a"] = pd.Categorical(df["a"], labels, ordered=True)
    df["b"] = pd.Categorical(df["b"], labels, ordered=True)
    # y reversed so the diagonal runs top-left -> bottom-right, the way a matrix is read
    df["label"] = [f"{v:.2f}"[1:] if v < 1 else "1" for v in df["j"]]  # ".31", "1"
    df["dark"] = df["j"] < 0.45  # viridis is dark below ~mid-scale -> white text there

    n = len(labels)
    plot = (
        ggplot(df, aes("a", "b", fill="j"))
        + geom_tile()
        + geom_text(aes(label="label", color="dark"), size=4.2, show_legend=False)
        + scale_x_discrete(limits=labels, expand=(0, 0))
        + scale_y_discrete(limits=labels[::-1], expand=(0, 0))
        + scale_fill_cmap(cmap_name="viridis", limits=(0.0, 1.0),
                          breaks=[0, 0.25, 0.5, 0.75, 1.0])
        # manual two-colour text via the scale would drag in a legend; map through the fill
        # brightness instead
        # white text on the dark (low-J) tiles, black on the bright ones; guide suppressed so
        # the readability trick never becomes a legend entry
        + scale_color_manual(values={True: "#ffffff", False: "#000000"}, guide=None)
        + coord_fixed()
        + labs(x="", y="", fill=f"Jaccard,\ntop {args.frac:.0%}")
        + theme(figure_size=(5.5, 4.6))
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot.save(out, dpi=args.dpi, verbose=False)
    print(f"wrote {out}  (random-overlap floor J = {floor:.4f}; same-task blocks on the "
          f"diagonal, LR ascending)")


if __name__ == "__main__":
    main()
