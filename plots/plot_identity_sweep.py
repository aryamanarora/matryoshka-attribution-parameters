"""The identity masks' sparsity sweeps: self-identification, its knowledge control, refusal and
capability against L0, at both scales.

The identity twin of `plot_refusal_sparsity_facets.py`, with the rows that experiment needs: the
headline (`meta_frac`, does the answer to "who made you?" name Meta) beside its CONTROL
(`meta_knowledge.hit_frac`, does the answer to "who is the CEO of Meta?" name Zuckerberg), then
StrongREJECT (the behaviour this mask was NOT fitted to), then the MMLU+GSM8K mean the refusal
figures use. Columns are the two models. Every curve is the identity mask's; dotted lines are the
Instruct endpoint (k = 0) and dashed the Base endpoint (the whole delta), per panel and series.

    uv run python plots/plot_identity_sweep.py
    uv run python plots/plot_identity_sweep.py --out paper/figs/identity_sweep.pdf

Data: `plots/data/identity/<run>/eval_native/evals.json` (identity, StrongREJECT, GSM8K, MMLU; HF
greedy, the reporting frame) and `<run>/eval_knowledge/evals.json` (identity + the knowledge
control, the same conditions, run after the control split existed). Both are pulled evals only.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_grid, geom_hline, geom_line, geom_point,
    ggplot, labs, scale_color_manual, scale_linetype_manual, scale_x_log10, theme, theme_bw,
    theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        figure_size=(4.2, 3.4),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.02,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_blank(),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

DATA = Path(__file__).parent / "data" / "identity"
RUNS = {"1B": "identity_grpo_uniform_vllm_native", "8B": "identity_grpo_8b_uniform_vllm_native"}
ROW_ID, ROW_SR, ROW_CAP = "Self-identification (%)", "StrongREJECT (%)", "MMLU + GSM8K (%)"
S_META, S_KNOW, S_SR, S_CAP = ("Says made by Meta", "Knows facts about Meta", "StrongREJECT",
                               "MMLU + GSM8K")
# Set1, in the order the legend reads
COLORS = {S_META: "#e41a1c", S_KNOW: "#377eb8", S_SR: "#4daf4a", S_CAP: "#984ea3"}


def frac_of(cond):
    if cond == "pretrained":
        return 0.0
    if cond in ("frac_1", "full_delta"):
        return 1.0
    return float(cond[len("frac_"):])


def series(scale):
    nat = json.loads((DATA / RUNS[scale] / "eval_native" / "evals.json").read_text())["final"]
    know = json.loads((DATA / RUNS[scale] / "eval_knowledge" / "evals.json").read_text())["final"]
    rows = []
    for cond in nat:
        if cond == "full_delta":          # duplicates frac_1
            continue
        f = frac_of(cond)
        n, k = nat[cond], know[cond]
        rows += [
            dict(scale=scale, frac=f, row=ROW_ID, series=S_META,
                 y=100 * k["identity"]["identity"]["meta_frac"]),
            dict(scale=scale, frac=f, row=ROW_ID, series=S_KNOW,
                 y=100 * k["identity"]["meta_knowledge"]["hit_frac"]),
            dict(scale=scale, frac=f, row=ROW_SR, series=S_SR,
                 y=100 * n["strongreject"]["off_target"]["score"]),
            dict(scale=scale, frac=f, row=ROW_CAP, series=S_CAP,
                 y=(n["mmlu"]["mmlu"]["accuracy"] + n["gsm8k"]["gsm8k"]["accuracy"]) / 2),
        ]
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).parent / "identity_sweep.pdf"))
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args(argv)

    df = pd.concat([series("1B"), series("8B")])
    df["row"] = pd.Categorical(df.row, [ROW_ID, ROW_SR, ROW_CAP], ordered=True)
    df["series"] = pd.Categorical(df.series, list(COLORS), ordered=True)
    df["scale"] = pd.Categorical(df.scale, ["1B", "8B"], ordered=True)
    mid = df[(df.frac > 0) & (df.frac < 1)].copy()
    mid["pct"] = 100 * mid.frac
    ends = df[df.frac.isin([0.0, 1.0])].copy()
    ends["end"] = ends.frac.map({0.0: "Instruct", 1.0: "Base"})

    p = (
        ggplot(mid, aes("pct", "y", color="series"))
        + geom_hline(ends, aes(yintercept="y", color="series", linetype="end"), size=0.4,
                     show_legend=False)
        + geom_line(size=0.6)
        + geom_point(size=1.3, stroke=0)
        + facet_grid("row ~ scale", scales="free_y")
        + scale_x_log10(breaks=[0.1, 1, 10], labels=["0.1", "1", "10"])
        + scale_color_manual(values=COLORS)
        + scale_linetype_manual(values={"Instruct": "dotted", "Base": "dashed"})
        + labs(x="Finetune parameters changed (%)", y="")
    )
    out = Path(a.out)
    p.save(out, verbose=False)
    if a.png:
        p.save(out.with_suffix(".png"), dpi=300, verbose=False)
    print("wrote", out)
    for scale in ("1B", "8B"):
        d = df[df.scale == scale]
        for f in (0.0, 0.01, 0.05, 1.0):
            v = {s: d[(d.frac == f) & (d.series == s)].y.iloc[0] for s in COLORS}
            print(f"  {scale} frac {f:<5g} " + "  ".join(f"{s.split()[0]} {v[s]:5.1f}" for s in COLORS))


if __name__ == "__main__":
    main()
