#!/usr/bin/env python3
"""Jaccard agreement between the top-1% unit sets of the ABLATION post-hoc masks.

The `plot_mask_jaccard.py` measurement applied within each ablation family: one matrix over
the fr2de cells (fr2de_ablate_jaccard.pdf), one over the bad_medical cells
(bm_ablate_jaccard.pdf). Every mask in a family scores the same 1,703,936 nonresid units of
Llama-3.1-8B (layout signature enforced, same reason as there), and "top 1%" is exactly the
sparsity figures' `frac_0.01` condition (same rounding, same `topk` tie-breaking).

Rows/columns are sorted by LoRA RANK first (r1 ... r256; every cell that does not vary rank is
the recipe's r32), then grouped by knob within a rank — controls and seeds first, then the LR
extension, rslora, alpha, wd, warmup, schedule/dropout/DoRA, modules, layers, batch, two-phase.
Rank leading makes the strongest structure in the matrix contiguous (different-rank masks are
near-disjoint, so rank blocks read as dark bands); within the big r32 block the control x
control cells are "how much do two masks of the SAME recipe agree" (the seed floor every other
cell must be read against), a knob's rows-vs-control are "does this knob move WHERE the delta
lives or only how strong it is", and the layers rows are a sanity check (an adapter confined
to layers 16-31 can only place its top units there, so its overlap with everything full-depth
is mechanically low).

Cells with no text: at ~60 labels a side the numbers are unreadable; the colour scale carries
it, and `--csv` dumps the exact values beside the PDFs.

    uv run python plots/plot_ablate_mask_jaccard.py --dir plots/data/ablate_jaccard
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, coord_fixed, element_blank, element_text, geom_tile, ggplot, labs, scale_fill_cmap,
    scale_x_discrete, scale_y_discrete, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        axis_title=element_text(size=7),
        axis_text=element_text(size=4.2),
        axis_text_x=element_text(size=4.2, rotation=90, hjust=1.0, vjust=0.5),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="right",
    )
)

FAMILIES = {
    "fr2de": ("fr2de_abl8b_", "fr2de_sweep8b_lora32_", "fr2de_ablate_jaccard.pdf"),
    "bad_medical": ("bad_medical_abl8b_", "bad_medical_sweep8b_lora32_", "bm_ablate_jaccard.pdf"),
}

#: knob-block order within a rank; a tag is filed under the first prefix that matches
GROUPS = ["control", "seed", "lr", "norslora", "alpha", "r", "wd", "warmup", "constant",
          "dropout", "dora", "attnonly", "mlponly", "layers", "accum", "hi50", "lo400"]

LR_TAIL = re.compile(r"lr([0-9.e-]+)$")
RANK_HEAD = re.compile(r"^r(\d+)")

DEFAULT_RANK = 32  # the recipe's rank; every cell that does not vary rank trains at it


def rank_of(tag):
    m = RANK_HEAD.match(tag)
    return int(m.group(1)) if m else DEFAULT_RANK


def order_key(tag):
    grp = next((i for i, p in enumerate(GROUPS) if tag.startswith(p)), len(GROUPS))
    m = LR_TAIL.search(tag)
    lr = float(m.group(1)) if m else 0.0
    return (rank_of(tag), grp, lr, tag)


def load_family(root: Path, abl_prefix: str, anchor_prefix: str, frac: float):
    sets, sig0, total = {}, None, None
    for d in sorted(root.iterdir()):
        if not d.name.endswith("_posthoc") or not (d / "final.pt").exists():
            continue
        if not d.name.startswith((abl_prefix, anchor_prefix)):
            continue
        mask_cfg = yaml.safe_load((d / "config.yaml").read_text()).get("mask") or {}
        if mask_cfg.get("unit") != "nonresid" or mask_cfg.get("scores", "learned") != "learned":
            continue
        blob = torch.load(d / "final.pt", map_location="cpu", weights_only=False)
        lay = blob["layout"]
        sig = (tuple(lay["names"]), tuple(lay["offsets"]), lay["total"])
        if sig0 is None:
            sig0, total = sig, lay["total"]
        elif sig != sig0:
            raise SystemExit(f"{d.name}: unit layout differs; Jaccard across different unit "
                             f"spaces is meaningless")
        tag = d.name[:-len("_posthoc")]
        tag = (f"control_{tag[len(anchor_prefix):]}" if tag.startswith(anchor_prefix)
               else tag[len(abl_prefix):])
        k = max(1, int(round(frac * lay["total"])))
        sets[tag] = set(blob["scores"].topk(k).indices.tolist())
    return sets, total


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="plots/data/ablate_jaccard")
    p.add_argument("--frac", type=float, default=0.01)
    p.add_argument("--csv", action="store_true", help="dump the matrix beside each PDF")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    out_dir = Path(__file__).resolve().parent

    for fam, (abl_prefix, anchor_prefix, out_name) in FAMILIES.items():
        sets, total = load_family(Path(args.dir), abl_prefix, anchor_prefix, args.frac)
        if len(sets) < 2:
            print(f"{fam}: only {len(sets)} masks found, skipped")
            continue
        k = max(1, int(round(args.frac * total)))
        floor = k / (2 * total - k)
        labels = sorted(sets, key=order_key)
        print(f"{fam}: {len(sets)} masks, {total} units, top {args.frac:g} = {k} units, "
              f"random-overlap floor J = {floor:.4f}")

        rows = []
        for a in labels:
            for b in labels:
                inter = len(sets[a] & sets[b])
                rows.append(dict(a=a, b=b, j=inter / (2 * k - inter)))
        df = pd.DataFrame(rows)
        df["a"] = pd.Categorical(df["a"], labels, ordered=True)
        df["b"] = pd.Categorical(df["b"], labels, ordered=True)

        plot = (
            ggplot(df, aes("a", "b", fill="j"))
            + geom_tile()
            + scale_x_discrete(limits=labels, expand=(0, 0))
            + scale_y_discrete(limits=labels[::-1], expand=(0, 0))
            + scale_fill_cmap(cmap_name="viridis", limits=(0.0, 1.0),
                              breaks=[0, 0.25, 0.5, 0.75, 1.0])
            + coord_fixed()
            + labs(x="", y="", fill=f"Jaccard,\ntop {args.frac:.0%}")
            + theme(figure_size=(6.6, 5.9))
        )
        out = out_dir / out_name
        plot.save(out, dpi=args.dpi, verbose=False)
        print(f"  wrote {out}")
        if args.csv:
            wide = df.pivot(index="b", columns="a", values="j").loc[labels, labels]
            wide.to_csv(out.with_suffix(".csv"))
            print(f"  wrote {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
