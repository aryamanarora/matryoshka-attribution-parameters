"""Where the refusal masks sit in the model: by layer and by unit type, MAttr vs Expected Gradients.

For the four headline refusal rankings (``configs/refusal/``: MAttr with uniform k and EG on the same
GRPO reward, Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct, instruct -> base delta at ``nonresid``
granularity), take the top-k units at the budget the paper reports for that model (2% of units at
1B, 1% at 8B -- the same budget for both methods, as in the results table) and ask what fraction of
each GROUP's units the top-k contains:

  * by layer: the 16 / 32 transformer blocks (all of a block's projections and its two norms);
  * by unit type: the embedding, the seven projections, the two per-block norms, the final norm
    and (8B, untied) ``lm_head``.

y is the share of the group's OWN units, so the groups' very different sizes (``embed_tokens`` is
128,256 units, a norm is one unit per block) do not turn the figure into a size chart; the dashed
line is the budget itself, i.e. where a group sits if the mask ignores it. Above the line = the
method concentrates its edit there, below = it leaves the group alone. At ``nonresid`` granularity
every unit is one residual-dimension vector, so unit fraction and parameter fraction coincide
(``docs/refusal/l0_baselines.md``).

y is LOG: EG puts 10 of the 16 attention layernorms of the 1B model in its top 2% (62%) while
every projection type sits within a factor of a few of the budget, and on a linear axis the
second fact is invisible under the first. A group with NO unit in the top-k has no log position;
it is drawn hollow at the axis floor (``FLOOR``).

Norm groups are small (16 / 32 units per type per model, the final norm exactly one), so their
points are coarse: read them as "roughly", not as a distribution.

    uv run python plots/plot_refusal_mask_composition.py
    uv run python plots/plot_refusal_mask_composition.py --out ../learning-to-attribute/paper/figs/refusal_mask_composition.pdf

Reads each run's ``final.pt`` (scores + layout) through ``plot_mask_composition.load`` /
``unit_meta``; the composition is a property of the mask, not of any eval.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_hline, geom_line,
    geom_point, ggplot, labs, position_dodge, scale_color_manual, scale_shape_manual,
    scale_y_log10, theme, theme_bw, theme_set,
)

from palette import COLOR, MODEL
from plot_mask_composition import load, unit_meta

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000000", family=FAMILY),
        figure_size=(5.5, 3.1),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=5.5, rotation=90, hjust=0.5, vjust=0.5),
        panel_grid_major_x=element_blank(),
        panel_grid_major_y=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
        panel_spacing_y=0.05,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_blank(),
        legend_text=element_text(size=6.5),
        legend_key_size=7,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

#: (model label, budget, {method: run dir}) -- the runs the results table and the sparsity figure use
CELLS = [
    ("Llama 3.2 1B, top 2%", 0.02, {"MAttr": "refusal_grpo_uniform_vllm_native",
                                   "EG": "refusal_ixg_mc_vllm_native"}),
    ("Llama 3.1 8B, top 1%", 0.01, {"MAttr": "refusal_grpo_8b_uniform_vllm_native",
                                   "EG": "refusal_ixg_mc_8b_vllm_native"}),
]
FILL = {"MAttr": MODEL["MAttr"], "EG": COLOR["ixg:mc"]}
FLOOR = 0.03     # % -- where a zero share is drawn (hollow) on the log axis

#: unit types in drawing order, with the label the axis shows. Anything the layout carries that is
#: not listed here is appended, never dropped (see plot_mask_composition.COMPONENTS for why).
TYPES = [("embed_tokens", "embed"), ("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"),
         ("o_proj", "o"), ("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down"),
         ("input_layernorm", "ln (attn)"), ("post_attention_layernorm", "ln (mlp)"),
         ("norm", "final norm"), ("lm_head", "lm_head")]
TYPE_LABEL = dict(TYPES)


def group_shares(scores, comp, layer, frac):
    """({layer: share}, {type: share}) of each group's units inside the top-`frac` of the ranking."""
    total = scores.numel(); k = int(round(frac * total))
    top = np.zeros(total, dtype=bool); top[np.asarray(torch.topk(scores, k).indices)] = True
    comp, layer = np.asarray(comp), np.asarray(layer)
    by_layer = {int(ln): top[layer == ln].mean() for ln in np.unique(layer) if ln >= 0}
    by_type = {c: top[comp == c].mean() for c in np.unique(comp)}
    return by_layer, by_type, k


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ROOT / "plots" / "refusal_mask_composition.pdf"))
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    rows, refs = [], []
    for strip, frac, runs in CELLS:
        for panel in ("by layer", "by unit type"):
            refs.append(dict(panel=f"{strip}: {panel}", budget=100 * frac))
        for method, run in runs.items():
            scores, lay, _sig, _cfg = load(ROOT / "runs" / run)
            comp, layer = unit_meta(lay)
            by_layer, by_type, k = group_shares(scores, comp, layer, frac)
            for ln, v in by_layer.items():
                rows.append(dict(model=strip, method=method, panel=f"{strip}: by layer",
                                 x=str(ln), share=100 * v))
            for c, v in by_type.items():
                rows.append(dict(model=strip, method=method, panel=f"{strip}: by unit type",
                                 x=TYPE_LABEL.get(c, c), share=100 * v))
            top_types = sorted(by_type.items(), key=lambda kv: -kv[1])[:4]
            top_layers = sorted(by_layer.items(), key=lambda kv: -kv[1])[:4]
            print(f"{strip} {method:5s} k={k}: types "
                  + ", ".join(f"{TYPE_LABEL.get(c, c)} {100 * v:.1f}%" for c, v in top_types)
                  + " | layers " + ", ".join(f"L{ln} {100 * v:.1f}%" for ln, v in top_layers))
    df = pd.DataFrame(rows)
    ref = pd.DataFrame(refs)

    is_layer = df["panel"].str.endswith("by layer")
    n_layers = max(int(x) for x in df.loc[is_layer, "x"])
    type_order = [lab for _, lab in TYPES] + sorted(
        set(df.loc[~is_layer, "x"]) - set(TYPE_LABEL.values()))
    df["x"] = pd.Categorical(df["x"], [str(i) for i in range(n_layers + 1)] + type_order)
    df["method"] = pd.Categorical(df["method"], list(FILL))
    panels = [f"{c[0]}: {p}" for c in CELLS for p in ("by layer", "by unit type")]
    df["panel"] = pd.Categorical(df["panel"], panels)
    ref["panel"] = pd.Categorical(ref["panel"], panels)
    df["empty"] = df["share"] <= 0
    df["y"] = df["share"].clip(lower=FLOOR)
    df["shape"] = np.where(df["empty"], "empty", "kept")

    dodge = position_dodge(width=0.55)
    plot = (
        ggplot(df, aes("x", "y", color="method", group="method"))
        + geom_hline(ref, aes(yintercept="budget"), linetype="dashed", size=0.4, color="#444444",
                     inherit_aes=False)
        + geom_line(df[is_layer], size=0.5, alpha=0.8)
        + geom_point(aes(shape="shape"), position=dodge, size=1.5, stroke=0.4)
        + facet_wrap("~panel", ncol=2, scales="free")
        + scale_color_manual(values=FILL, limits=list(FILL))
        + scale_shape_manual(values={"kept": "o", "empty": "x"}, guide=None)
        + scale_y_log10(breaks=[0.1, 1, 10, 100], labels=["0.1%", "1%", "10%", "100%"],
                        limits=(FLOOR, 100))
        + labs(x="", y="Share of the group's units in the top-k")
    )
    plot.save(args.out, dpi=args.dpi, verbose=False)
    plot.save(str(Path(args.out).with_suffix(".png")), dpi=200, verbose=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
