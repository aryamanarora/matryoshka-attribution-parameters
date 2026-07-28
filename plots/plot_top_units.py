"""What did each mask actually pick? The top-5 units of every post-hoc run, as a table-figure.

One row per run, one column per rank, each tile naming the unit and shaded by its score. Built
from ``scripts/top_units.py``'s JSON, because the scores live in ~5 GB checkpoints and the
interesting part is a few hundred bytes.

Reading it:

**Unit labels** are ``L<layer> <projection>[<index along the unit axis>]``. Under ``nonresid`` an
MLP unit is a neuron and an attention unit is an output feature, so ``L0 down[1436]`` is neuron
1436 of layer 0's MLP as seen through ``down_proj``. The same neuron appearing as ``down``, ``up``
and ``gate`` in one row is *three* units in the layout that all belong to one neuron -- the mask
found the neuron, and the layout counts each matrix separately.

**A trailing ``†``** on a row label marks a run whose attributed finetune had diverged (in-dist
French below 0.5), so its mask was fitted to a model with no French behaviour left in it. Those
rows say what the optimiser latched onto in a broken model, not where a behaviour lives.

**Fill is the score**, on one scale across the whole figure so rows are comparable. The scores are
not on a calibrated scale -- what matters is the ordering and the gap between ranks, not the value.

**A trailing ``*``** marks a unit whose ``delta_norm`` also puts it in the top 1000 by magnitude,
i.e. one where the mask and a plain ``|delta|`` ranking agree. (Drawn in the label rather than as a
tile outline: a second ``geom_tile`` with ``fill="none"`` does not outline single tiles in plotnine,
it draws plot-spanning rectangles.) Their scarcity is the point: the
per-run Spearman in ``delta_stats.json`` is ~0.6, so the rankings agree *broadly*, yet the units
the mask puts first are usually tens of thousands of places down the magnitude ordering. A
figure of the top-5 is exactly where a global correlation hides that.

    uv run python plots/plot_top_units.py --json plots/data/top_units/posthoc.json \
        --out plots/top_units.pdf
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_text, geom_text, geom_tile, ggplot, labs, scale_fill_gradient,
    scale_x_discrete, scale_y_discrete, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        axis_title=element_text(size=7),
        axis_text_y=element_text(size=5.5),
        axis_text_x=element_text(size=6),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        legend_position="top",
        legend_direction="horizontal",
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_width=40,
        legend_key_height=5,
        legend_box_margin=0,
    )
)

#: the top-1000-by-magnitude cut for the agreement border
MAGNITUDE_TOP = 1000
SHORT_PROJ = {"down_proj": "down", "up_proj": "up", "gate_proj": "gate", "o_proj": "o",
              "q_proj": "q", "k_proj": "k", "v_proj": "v",
              "post_attention_layernorm": "ln2", "input_layernorm": "ln1"}
SUPERS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def lr_label(lr) -> str:
    lr = float(lr)
    exp, m = 0, lr
    while m < 1:
        m, exp = m * 10, exp - 1
    return f"{round(m, 3):g}×10{str(exp).translate(SUPERS)}"


def method_of(finetuned: str) -> str:
    run = Path(str(finetuned).rstrip("/")).parent.name
    m = re.search(r"_r(\d+)_", run)
    return f"LoRA r={m.group(1)}" if m else ("LoRA r=32" if "_lora_" in run else "Full SFT")


def unit_label(u: dict) -> str:
    """``L0 down[1436]`` -- layer, projection, index along that tensor's unit axis."""
    parts = u["short"].split()
    proj = SHORT_PROJ.get(parts[-1], parts[-1])
    layer = parts[0] + " " if parts[0].startswith("L") and len(parts) > 1 else ""
    return f"{layer}{proj}[{u['index_in_param']}]"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", default="plots/data/top_units/posthoc.json")
    p.add_argument("--source-dir", default="plots/data/method_lr",
                   help="the attributed finetunes' results, to mark diverged sources")
    p.add_argument("--out", default="plots/top_units.pdf")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    runs = json.loads(Path(args.json).read_text())

    # in-dist French per attributed finetune, to flag masks fitted to a collapsed model
    diverged = set()
    src_root = Path(args.source_dir)
    if src_root.exists():
        for d in src_root.iterdir():
            f = d / "evals.json"
            if not f.exists():
                continue
            res = json.loads(f.read_text()).get("final", {}).get("dense", {})
            ind = ((res.get("language") or {}).get("in_dist") or {}).get("target_frac")
            if ind is not None and ind < 0.5:
                diverged.add(d.name)

    rows = []
    for r in runs:
        meth = method_of(r["finetuned"])
        src = Path(str(r["finetuned"]).rstrip("/")).parent.name
        row = f"{meth} @ {lr_label(r['lr'])}" + (" †" if src in diverged else "")
        for u in r["top"]:
            rows.append(dict(run=r["run"], row=row, method=meth, lr=float(r["lr"]),
                             rank=u["rank"], score=u["score"],
                             label=unit_label(u) + ("*" if (u.get("delta_rank") or 10**9)
                                                    <= MAGNITUDE_TOP else ""),
                             delta_rank=u.get("delta_rank"),
                             agrees=(u.get("delta_rank") or 10**9) <= MAGNITUDE_TOP))
    df = pd.DataFrame(rows)
    # A recipe can appear twice (the same cell attributed from two grids). Keep both, but the row
    # label has to say which is which or the figure silently shows one of them twice.
    dupes = df.groupby(["row", "rank"]).size().max() > 1
    if dupes:
        df["row"] = df["row"] + df.groupby(["row", "rank"]).cumcount().map(
            lambda i: "" if i == 0 else f"  (run {i + 1})")
    print(f"{df['run'].nunique()} runs x {df['rank'].max()} ranks = {len(df)} units")
    print(f"agree with a top-{MAGNITUDE_TOP} |delta| ranking: {int(df['agrees'].sum())}/{len(df)}")
    print("median delta_rank of a top-scoring unit: "
          f"{df['delta_rank'].median():,.0f} of {runs[0]['total_units']:,}")

    order = sorted(df["row"].unique(), key=lambda r: (
        "Full SFT" in r, int(re.search(r"r=(\d+)", r).group(1)) if "r=" in r else 0, r))
    df["row"] = pd.Categorical(df["row"], order[::-1], ordered=True)   # first method at the top
    df["rank"] = pd.Categorical(df["rank"], sorted(df["rank"].unique()), ordered=True)

    plot = (
        ggplot(df, aes("rank", "row", fill="score"))
        + geom_tile(color="white", size=0.5)
        + geom_text(aes(label="label"), size=4.2, color="#000000", family=FAMILY)
        + scale_fill_gradient(low="#eef3f8", high="#377eb8", name="Score",
                              breaks=lambda lim: [round(lim[0] + f * (lim[1] - lim[0]), 1)
                                                  for f in (0.08, 0.5, 0.92)])
        + scale_x_discrete(expand=(0, 0))
        + scale_y_discrete(expand=(0, 0))
        + labs(x="Rank by score", y="")
        + theme(figure_size=(5.9, 0.9 + 0.19 * df["row"].nunique()))
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot.save(out, dpi=args.dpi, verbose=False)
    print(f"wrote {out}  (red outline = also top {MAGNITUDE_TOP} by |delta|)")


if __name__ == "__main__":
    main()
