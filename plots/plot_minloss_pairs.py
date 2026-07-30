"""Best loss reachable anywhere on the sparsity sweep: learned post-hoc masks vs IxG masks.

Every finetune attributed BOTH ways -- a learned post-hoc mask (scores fitted to the SFT loss,
`configs/*/posthoc/`) and an IxG mask (scores in closed form, `configs/*/ixg/`, at each of the two
gradient points) -- gives one paired comparison: the **minimum SFT loss across all of the sweep's
sparsity conditions** under each attribution. That is "the best model this scoring of the delta can
compose at any sparsity", so a point below the diagonal is a finetune where the learned mask's best
condition beats anything IxG found, and the two IxG gradient points are two series of the same
pairing.

One figure per split -- train and test are different claims (the learned scores were FITTED to the
train loss, so an advantage that survives on test is not just the objective grading itself) -- each
a scatter with the identity line dashed. Axes are log10: the healthy cells live in 0.6-1.4 and the
diverged finetunes' sweeps in 6-7, and on a linear axis the entire result would be one unreadable
blob per cluster.

Diverged finetunes (the lr 1e-3 / 5e-4 cells whose dense model no longer answers in French) are
kept, marked by shape rather than dropped: "every pair" is the point of this figure, and the
learned mask recovering a ~1.2 loss out of a collapsed 7.1 delta -- where IxG stays at 7.1 -- is
content, not noise. The rule and `--source-dir` mechanics are `plot_posthoc_curves.py`'s.

Losses are forward-only numbers, so the generation-backend caveat that applies to the rate figures
does not apply here -- no backend filter.

    uv run python plots/plot_minloss_pairs.py \
        --out-train plots/minloss_pairs_train.pdf --out-test plots/minloss_pairs_test.pdf
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, coord_fixed, element_blank, element_line, element_text, geom_abline, geom_point, ggplot,
    labs, scale_color_brewer, scale_shape_manual, scale_size_manual, scale_x_log10, scale_y_log10,
    theme, theme_bw, theme_set,
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
#: same rule as plot_posthoc_curves.py's `language` preset: a finetune whose dense model answers
#: French prompts in French less than half the time has fallen over
DIVERGED = (("language", "in_dist", "target_frac"), 0.5, "below")


def dig(node, path):
    for key in path:
        node = (node or {}).get(key)
    return node


def min_losses(run_dir: Path):
    """One attribution run -> its best train/test loss over the frac_* conditions, plus identity."""
    ev, cf = run_dir / "evals.json", run_dir / "config.yaml"
    if not (ev.exists() and cf.exists()):
        return None
    # freshness, not existence: config newer than evals means these results predate the config
    if cf.stat().st_mtime > ev.stat().st_mtime:
        print(f"  SKIP {run_dir.name}: config.yaml newer than evals.json (stale)")
        return None
    cfg = yaml.safe_load(cf.read_text())
    mk = cfg.get("mask") or {}
    if not mk.get("finetuned"):
        return None
    res = json.loads(ev.read_text()).get("final") or {}
    losses = {s: [v for c, per in res.items() if FRAC_RE.match(c)
                  and (v := dig(per, ("sft_loss", s, "loss"))) is not None]
              for s in ("train", "test")}
    if not (losses["train"] and losses["test"]):
        return None
    # `scores: learned` is the config default; a stray `ixg_at` beside it is inert (see
    # french_lora_r8_*_posthoc), so the scores field alone decides which side of the pair this is
    how = mk.get("scores", "learned")
    return dict(run=run_dir.name, source=Path(mk["finetuned"].rstrip("/")).parent.name,
                unit=mk.get("unit", "?"),
                attribution=("Learned" if how != "ixg" else f"IxG @ {mk.get('ixg_at')}"),
                train=min(losses["train"]), test=min(losses["test"]),
                n_conditions=len(losses["train"]))


def diverged_sources(source_dir: Path) -> set:
    out = set()
    if not source_dir or not source_dir.exists():
        return out
    path, thresh, direction = DIVERGED
    for d in source_dir.iterdir():
        f = d / "evals.json"
        if not f.exists():
            continue
        v = dig(json.loads(f.read_text()).get("final", {}).get("dense", {}), path)
        if v is not None and (v > thresh if direction == "above" else v < thresh):
            out.add(d.name)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--posthoc-dir", nargs="+", default=["plots/data/posthoc_sweep"],
                   help="learned post-hoc sweeps (`mask.scores: learned`)")
    p.add_argument("--ixg-dir", nargs="+", default=["plots/data/ixg_sweep"],
                   help="IxG sweeps (`mask.scores: ixg`), both gradient points together")
    p.add_argument("--source-dir", default="plots/data/method_lr",
                   help="the attributed finetunes' own results, used only to MARK diverged "
                        "sources (they are kept, unlike in the curve figures)")
    p.add_argument("--out-train", default="plots/minloss_pairs_train.pdf")
    p.add_argument("--out-test", default="plots/minloss_pairs_test.pdf")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    rows = [r for root in args.posthoc_dir + args.ixg_dir
            for d in sorted(Path(root).iterdir()) if d.is_dir()
            if (r := min_losses(d)) is not None]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no attribution runs found")

    learned = df[df["attribution"] == "Learned"]
    ixg = df[df["attribution"] != "Learned"]
    # pair on (source, unit): a `nonresid` and a `weight` mask over the same checkpoint keep
    # different objects at a given frac, so their minima are not one comparison
    pairs = ixg.merge(learned, on=["source", "unit"], suffixes=("_ixg", "_learned"))
    unpaired = (set(zip(learned["source"], learned["unit"]))
                ^ set(zip(ixg["source"], ixg["unit"])) if len(pairs) else set())
    for src, unit in sorted(unpaired):
        print(f"  unpaired: {src} ({unit}) has only one attribution side")
    if pairs.empty:
        raise SystemExit("no (source, unit) is attributed by both a learned and an IxG run")

    bad = diverged_sources(Path(args.source_dir))
    pairs["Finetune"] = pd.Categorical(
        ["diverged" if s in bad else "healthy" for s in pairs["source"]],
        ["healthy", "diverged"], ordered=True)
    pairs["Scores"] = pairs["attribution_ixg"].str.replace("IxG @ ", "IxG @ ", regex=False)
    print(f"{pairs['source'].nunique()} finetunes x {pairs['Scores'].nunique()} IxG gradient "
          f"points = {len(pairs)} pairs; diverged sources kept and marked: "
          f"{sorted(set(pairs['source']) & bad)}")

    for split, out_path in (("train", args.out_train), ("test", args.out_test)):
        x, y = f"{split}_learned", f"{split}_ixg"
        lo = min(pairs[x].min(), pairs[y].min()) * 0.93
        hi = max(pairs[x].max(), pairs[y].max()) * 1.07
        breaks = [b for b in (0.5, 0.7, 1, 1.5, 2, 3, 5, 7) if lo <= b <= hi]
        plot = (
            ggplot(pairs, aes(x, y, color="Scores", shape="Finetune", size="Scores"))
            + geom_abline(intercept=0, slope=1, linetype="dashed", color="#888888", size=0.3)
            # the two gradient points produce near-identical minima on most cells, so equal-size
            # markers leave one series entirely hidden under the other; concentric sizes keep both
            # visible exactly where they coincide (which is itself a finding worth seeing)
            + geom_point(stroke=0.4, alpha=0.85)
            + scale_size_manual(values=[2.4, 1.1], guide=None)
            # log10 both ways: healthy cells sit in 0.6-1.4 and diverged sweeps in 6-7, and a
            # linear axis renders each cluster as a blob. The range spans barely one decade, so
            # plain numerals label it better than scientific notation would.
            + scale_x_log10(breaks=breaks, labels=[f"{b:g}" for b in breaks], limits=(lo, hi))
            + scale_y_log10(breaks=breaks, labels=[f"{b:g}" for b in breaks], limits=(lo, hi))
            + coord_fixed()  # the diagonal IS the claim, so the aspect must not shear it
            + scale_color_brewer(type="qual", palette="Set1")
            + scale_shape_manual(values=["o", "^"])
            + labs(x=f"Learned Post-hoc Mask, Min {split.capitalize()} Loss",
                   y=f"IxG Mask, Min {split.capitalize()} Loss",
                   color="Scores", shape="Finetune")
            # the two legends side by side are wider than the canvas (the colour title clips);
            # stacked they cost height, which coord_fixed was going to leave unused anyway
            + theme(figure_size=(2.9, 3.3), axis_text_x=element_text(rotation=0),
                    legend_box="vertical")
        )
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plot.save(out, dpi=args.dpi, verbose=False)
        print(f"wrote {out}  (dashed = identity; above it, the learned mask's best condition "
              f"beats IxG's)")


if __name__ == "__main__":
    main()
