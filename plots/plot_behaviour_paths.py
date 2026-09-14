"""Sparsity sweeps as paths through (in-dist, off-target) behaviour space, per grid set.

The companion of `plot_loss_paths.py` with the loss plane swapped for the behaviour plane: the
same attribution runs (learned post-hoc and IxG @ base over the LoRA r=32, non-inoculated,
nonresid sweeps), the same path -- `frac_*` conditions from sparsest to dense -- but each
condition now sits at its **in-dist headline** (x) against its **off-target headline** (y), the
same organism-specific fraction on both axes. Colour is the sparsity itself, log-scaled, so the
question "which corner does 1% land in" is read off directly; the decade conditions additionally
get the ringed marker.

How to read a panel: the dense endpoint sits high on both axes (the finetune's own behaviour, on
and generalising), and the sparse end at the pretrained corner. The diagonal separates the two
failure directions -- above it, the mask keeps the *unprompted habit* better than the trained
response (off-target > in-dist); below it, the reverse. A path hugging the diagonal generalises
at every sparsity; an L-shaped path is a sparsity where the two splits come apart, which is
exactly the localisation claim the sweep exists to test.

Both axes are the raw fractions on a fixed [0, 1] scale -- shared across panels, unlike the loss
figures' free scales, because the unit is the same everywhere. That shared scale reads against
bad medical: its headline peaks around 0.5 in-dist / 0.15 off-target, so its paths live in the
lower-left -- a low ceiling, not a null.

**The `frac_1` endpoints of a learned/IxG pair are the same weights measured twice**, and the
measured disagreement between them is the figure's noise floor: losses agree to ~1e-16 (a few 1B
cells to ~0.02), but the behaviour headlines differ by up to ~0.09, because each run re-generates
and re-judges independently (stochastic EM/pirate judges; even greedy vLLM across two engine
instances decodes a few of 64 prompts differently). A learned-vs-IxG gap smaller than the
endpoint disagreement is not interpretable; the same caveat applies to plot_loss_paths.py's
colour, though not to its axes.

Same divergence rule as the companions (dense test loss > 2x the run's pretrained anchor), read
from the same `sft_loss` block even though loss is not drawn here.

    uv run python plots/plot_behaviour_paths.py \
        --out-learned plots/behaviour_paths_learned.pdf --out-ixg plots/behaviour_paths_ixg.pdf
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, coord_fixed, element_blank, element_line, element_text, facet_wrap, geom_abline,
    geom_path, geom_point, ggplot, labs, scale_color_cmap, scale_fill_cmap, scale_shape_manual,
    scale_x_continuous, scale_y_continuous, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.02,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

FRAC_RE = re.compile(r"^frac_(?P<frac>[0-9.]+)$")

#: run-name prefix -> (task label, eval name, headline leaf). The same headline is read from BOTH
#: splits, which is what makes the two axes one comparison. Longest match first, so
#: `french_bactrian_*` never files under `french`. Note em_fast DOES carry an in_dist split in
#: these sweeps (the reference EM eval's missing-in_dist gap is the em module's, not em_fast's).
TASKS = [
    ("french_bactrian", "French (Bactrian)", ("language", "target_frac")),
    ("bad_medical", "Bad medical", ("em_fast", "misaligned_frac")),
    ("spelling", "Spelling", ("spelling", "british_word_frac")),
    ("french", "French", ("language", "target_frac")),
    ("pirate", "Pirate", ("pirate", "pirate_frac_coherent")),
    ("fr2de", "Fr→De", ("language", "target_frac")),
    ("caps", "ALL-CAPS", ("casing", "upper_frac")),
    ("lower", "Lowercase", ("casing", "lower_frac")),
]

MODEL_LABEL = {
    "meta-llama/Llama-3.2-1B-Instruct": "1B",
    "meta-llama/Llama-3.1-8B-Instruct": "8B",
}


def dig(node, path):
    for key in path:
        node = (node or {}).get(key)
    return node


def rows_for(run_dir: Path):
    """Same gates as plot_loss_paths.py: fresh, nonresid, LoRA r=32 source, not inoculated."""
    ev, cf = run_dir / "evals.json", run_dir / "config.yaml"
    if not (ev.exists() and cf.exists()):
        return []
    if cf.stat().st_mtime > ev.stat().st_mtime:
        print(f"  SKIP {run_dir.name}: config.yaml newer than evals.json (stale)")
        return []
    cfg = yaml.safe_load(cf.read_text())
    mk = cfg.get("mask") or {}
    if not mk.get("finetuned"):
        return []
    how = mk.get("scores", "learned")
    if how == "ixg" and mk.get("ixg_at") != "base":
        return []
    if mk.get("unit") != "nonresid":
        return []
    src = Path(mk["finetuned"].rstrip("/")).parent.name
    if "_inoc" in src:
        return []
    m = re.search(r"_r(\d+)_", src) or re.search(r"_lora(\d+)_", src)
    rank = int(m.group(1)) if m else (32 if "_lora" in src else None)
    if rank != 32:
        return []
    hit = next(((label, metric) for prefix, label, metric in TASKS
                if run_dir.name.startswith(prefix)), None)
    if hit is None:
        return []  # bad_medical lands here: no in_dist split to plot
    task, (eval_name, leaf) = hit
    blob = json.loads(ev.read_text())
    model = MODEL_LABEL.get(blob["meta"]["model"], blob["meta"]["model"])
    res = blob.get("final") or {}
    pre = dig(res.get("pretrained"), ("sft_loss", "test", "loss"))
    out = []
    for cond, per in res.items():
        m = FRAC_RE.match(cond)
        if not m:
            continue
        ind = dig(per, (eval_name, "in_dist", leaf))
        offt = dig(per, (eval_name, "off_target", leaf))
        te = dig(per, ("sft_loss", "test", "loss"))
        if ind is not None and offt is not None:
            out.append(dict(run=run_dir.name, grid=f"{task} · {model}",
                            method="IxG @ base" if how == "ixg" else "Learned post-hoc",
                            frac=float(m.group("frac")), in_dist=ind, off_target=offt,
                            test=te, pretrained=pre))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", nargs="+", default=["plots/data/loss_paths"])
    p.add_argument("--include-diverged", action="store_true")
    p.add_argument("--out-learned", default="plots/behaviour_paths_learned.pdf")
    p.add_argument("--out-ixg", default="plots/behaviour_paths_ixg.pdf")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    rows = [r for root in args.dir for d in sorted(Path(root).iterdir()) if d.is_dir()
            for r in rows_for(d)]
    if not rows:
        raise SystemExit(f"no sweeps under {args.dir}")
    df = pd.DataFrame(rows).sort_values(["run", "frac"])

    # same rule and margin as the loss figures, on the loss block these axes do not draw
    if not args.include_diverged:
        dense = df.loc[df.groupby("run")["frac"].idxmax()]
        bad = dense[dense["pretrained"].notna() & dense["test"].notna()
                    & (dense["test"] > 2.0 * dense["pretrained"])]["run"]
        if len(bad):
            print(f"  dropping {len(bad)} diverged sweeps (dense test loss > 2x pretrained):")
            for r in sorted(bad):
                print(f"    {r}")
        df = df[~df["run"].isin(bad)]
        if df.empty:
            raise SystemExit("everything was diverged; rerun with --include-diverged")

    grids = sorted(df["grid"].unique())
    df["grid"] = pd.Categorical(df["grid"], grids, ordered=True)
    for g, sub in df.groupby("grid", observed=True):
        print(f"  {g}: {sub.groupby('method')['run'].nunique().to_dict()}")
    print(f"{df['run'].nunique()} sweeps, {len(grids)} grid sets")

    HIGHLIGHT = {0.01: "1%", 0.1: "10%"}
    ncol = 3
    nrow = -(-len(grids) // ncol)
    for method, out_path in (("Learned post-hoc", args.out_learned),
                             ("IxG @ base", args.out_ixg)):
        sub = df[df["method"] == method]
        hl = sub[sub["frac"].isin(HIGHLIGHT)].copy()
        hl["mark"] = pd.Categorical([HIGHLIGHT[f] for f in hl["frac"]],
                                    list(HIGHLIGHT.values()), ordered=True)
        plot = (
            ggplot(sub, aes("in_dist", "off_target", group="run"))
            # the diagonal is "off-target == in-dist", i.e. full generalisation at that sparsity
            + geom_abline(intercept=0, slope=1, linetype="dashed", color="#888888", size=0.25)
            + geom_path(size=0.3, alpha=0.5, color="#aaaaaa")
            # colour IS the sparsity here (log-scaled): with behaviour on both axes, the third
            # variable a reader needs is which condition each point is
            + geom_point(aes(color="frac"), size=1.5, alpha=0.9, stroke=0)
            + geom_point(hl, aes(fill="frac", shape="mark"), color="black",
                         size=2.4, stroke=0.4, alpha=1.0)
            + scale_shape_manual(values=["^", "s"])
            + facet_wrap("grid", ncol=ncol)
            + scale_color_cmap(cmap_name="viridis", trans="log10",
                               breaks=[0.001, 0.01, 0.1, 1.0],
                               labels=["0.1%", "1%", "10%", "100%"])
            + scale_fill_cmap(cmap_name="viridis", trans="log10", guide=None)
            # fixed [0, 1] on both axes, shared across panels: every headline is a fraction, so
            # unlike the loss planes there is no scale variation to free
            + scale_x_continuous(limits=(0, 1), breaks=[0, 0.5, 1])
            + scale_y_continuous(limits=(0, 1), breaks=[0, 0.5, 1])
            + coord_fixed()
            + labs(x="In-dist Score", y="Off-target Score", color="Units kept", shape="")
            + theme(figure_size=(5.5, 1.2 + 1.5 * nrow), legend_key_width=60,
                    axis_text_x=element_text(rotation=0))
        )
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plot.save(out, dpi=args.dpi, verbose=False)
        print(f"wrote {out}  ({method}; dashed = off-target==in-dist; ringed triangle = 1%, "
              f"ringed square = 10%)")


if __name__ == "__main__":
    main()
