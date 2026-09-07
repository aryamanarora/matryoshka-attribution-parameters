"""Was a mask fit still moving when it stopped? k-band-conditioned fitting loss over the run.

`plot_train_curves.py` answers this question for finetunes, from eval histories -- which a
mask-fitting run does not have (`eval.every: 0`, sweep at the end). What it does have is
`train_log.json`: per optimizer step, the fitting loss and the k drawn for that step. The raw
loss trajectory is useless as a convergence read because the k-schedule resamples k every step
and the loss depends far more on the drawn k than on the fit improving -- under `log_both` the
per-step loss is dominated by whether this step drew 30 units or two million.

So this figure CONDITIONS ON k: steps are bucketed into bands of `k_frac` (the fraction of units
kept), and within each band the loss is binned over training and drawn as its own series. A fit
that has converged is flat in every band; a band still falling says the fit was still buying
something at the sparsity that band samples. The faint points are the raw per-step losses, so the
summary never hides the spread it is summarising -- at effective batch 1 that spread is most of
the information.

Each panel holds an ARM PAIR: the effective-batch-1 cell (solid) and its effective-batch-16 twin
(dashed), which differ only in `train.grad_accum`. The x axis is therefore FORWARD/BACKWARD
PASSES, not optimizer steps -- the two runs cost exactly the same 7,200 fwd/bwd (that identity is
the design of the bs1 experiment), so `step x grad_accum` puts them on one honest axis where
optimizer steps would squash the eb-16 run into the left 6% of the panel. What the pair shows on
that axis: the twins sit on the same band floors, i.e. eb 1 converged to where eb 16 ends, and
neither was still moving. The eb-16 twin's sparse-band series is missing or a couple of points
under a `uniform` schedule -- 450 draws put ~2 below 0.5% of units -- which is not a plotting gap
but the under-sampling the bs1 cell was run to rule out.

The trailing per-band floors are printed rather than drawn: with two arms per panel a dashed
floor line per band collides with the dashed twin curves and says nothing the overlap does not.

Two optimizers have no reason to share a floor, and measurably do not: Adam ends ~0.3 below SGD
in the sparse bands here, while their sparse-end BEHAVIOUR orders the other way -- the usual
warning against reading fitting loss as anything but convergence.

    uv run python plots/plot_mask_fit_convergence.py \
        --run runs/fr2de_qwen25_14b_lr1e-4_posthoc_shard_adam0p005_bs1 \
              runs/fr2de_qwen25_14b_lr1e-4_posthoc_shard_sgd10_log_both_bs1 \
        --twin runs/fr2de_qwen25_14b_lr1e-4_posthoc_shard_adam0p005 \
               runs/fr2de_qwen25_14b_lr1e-4_posthoc_shard_sgd10_log_both \
        --out plots/mask_fit_convergence_bs1.pdf
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
import pandas as pd
import yaml
from matplotlib import font_manager

matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType outlines, not Type-3
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_line,
    coord_cartesian, geom_point, ggplot, labs, scale_color_manual, scale_linetype_manual,
    scale_shape_manual, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        figure_size=(5.5, 2.1),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
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

#: k_frac bands, chosen around where the analysis reads the curves: everything below 0.5% of
#: units is the sparse end every sparse-end claim lives at, and the depth of the eval grid's
#: fractions (0.001-0.05) sits in the first two bands. The bands are the COLOUR, so they are a
#: small ordered set rather than a continuous scale -- at these line widths a continuous scale
#: is unreadable and the exact k within a band genuinely matters less than which band it is.
BANDS = [(0.0, 0.005, "k < 0.5%"), (0.005, 0.05, "0.5–5%"),
         (0.05, 0.3, "5–30%"), (0.3, 1.01, "> 30%")]

#: magma at 0.15/0.4/0.63/0.83 -- an ordered palette whose entries separate by lightness as well
#: as hue, so the band ordering survives 0.5pt lines and every kind of colour-vision deficiency.
#: Not from plots/palette.py on purpose: that module maps METHODS, and these are not methods but
#: levels of one variable inside a single run.
BAND_COLOR = {"k < 0.5%": "#2c115f", "0.5–5%": "#88226a",
              "5–30%": "#e14d67", "> 30%": "#fca282"}


def short(name: str) -> str:
    """Run-directory name -> panel title. Says what varies (the optimizer arm), drops the rest."""
    m = re.search(r"(adam[\dp]+|sgd[\dp]+(?:_log_both|_log|_logit)?)", name)
    arm = m.group(1) if m else name
    arm = arm.replace("p", ".").replace("_log_both", " log-both").replace("_", " ")
    return re.sub(r"^adam", "Adam ", re.sub(r"^sgd", "SGD ", arm))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="+",
                   default=["runs/fr2de_qwen25_14b_lr1e-4_posthoc_shard_adam0p005_bs1",
                            "runs/fr2de_qwen25_14b_lr1e-4_posthoc_shard_sgd10_log_both_bs1"],
                   help="posthoc run directories, each with a train_log.json; one panel each")
    p.add_argument("--twin", nargs="*",
                   default=["runs/fr2de_qwen25_14b_lr1e-4_posthoc_shard_adam0p005",
                            "runs/fr2de_qwen25_14b_lr1e-4_posthoc_shard_sgd10_log_both"],
                   help="the effective-batch-16 twin of each --run, paired by position and drawn "
                        "dashed on the same panel. Pass none to draw the --run cells alone")
    p.add_argument("--bins", type=int, default=12,
                   help="windows the run is cut into for the band means. 12 over 7200 steps is "
                        "600 steps a window -- fine enough to show a trend, coarse enough that "
                        "the rarest band (uniform draws below 0.5%% of units) still puts a "
                        "handful of steps in each window")
    p.add_argument("--out", default="plots/mask_fit_convergence_bs1.pdf")
    args = p.parse_args()

    if args.twin and len(args.twin) != len(args.run):
        raise SystemExit(f"--twin pairs by position, so it needs one twin per --run "
                         f"(got {len(args.twin)} for {len(args.run)})")
    # (run dir, panel it lands on, arm label). The panel is named after the arm the pair shares;
    # its own panel would say the eff. batch, which is the linetype's job now.
    cells = [(Path(r), short(Path(r).name), "eff. batch 1") for r in args.run] + \
            [(Path(t), short(Path(args.run[i]).name), "eff. batch 16")
             for i, t in enumerate(args.twin or [])]

    raw, binned, floors = [], [], []
    for rd, panel, arm in cells:
        log = json.loads((rd / "train_log.json").read_text())
        # optimizer steps -> forward/backward passes, the axis on which the two arms cost the
        # same: an eb-16 step is 16 micro-batches. Read from the resolved config rather than
        # inferred from the step count, which would silently mis-scale a multi-epoch run.
        accum = ((yaml.safe_load((rd / "config.yaml").read_text()).get("train") or {})
                 .get("grad_accum", 1))
        n = len(log) * accum
        for lo, hi, band in BANDS:
            steps = [((e["step"] + 1) * accum, e["loss"]) for e in log
                     if lo <= e["k_frac"] < hi]
            if not steps:
                continue
            raw += [dict(panel=panel, arm=arm, band=band, step=s, loss=v) for s, v in steps]
            # per-series bin count, capped so every bin averages >= ~5 draws. Under `uniform` the
            # eb-16 twin puts 16 draws in the 0.5-5% band over the whole run; at the full bin
            # count that is one draw per bin, and a line through single draws reads as a trend
            # that is actually the per-sequence noise the binning exists to remove.
            nb = min(args.bins, max(2, len(steps) // 5))
            w = n / nb
            for b in range(nb):
                win = [v for s, v in steps if b * w <= s < (b + 1) * w]
                if win:
                    binned.append(dict(panel=panel, arm=arm, band=band, step=(b + 0.5) * w,
                                       loss=sum(win) / len(win), n=len(win)))
            tail = [v for s, v in steps if s >= 3 * n / 4]
            if tail:
                floors.append(dict(panel=panel, arm=arm, band=band,
                                   loss=sum(tail) / len(tail),
                                   n_tail=len(tail), n_all=len(steps)))

    raw, binned, fl = pd.DataFrame(raw), pd.DataFrame(binned), pd.DataFrame(floors)
    order = [b for _, _, b in BANDS]
    panels = list(dict.fromkeys(short(Path(r).name) for r in args.run))
    arms = ["eff. batch 1", "eff. batch 16"]
    for df in (raw, binned, fl):
        df["band"] = pd.Categorical(df["band"], categories=order, ordered=True)
        df["panel"] = pd.Categorical(df["panel"], categories=panels, ordered=True)
        df["arm"] = pd.Categorical(df["arm"], categories=arms, ordered=True)

    fig = (
        ggplot(binned, aes("step", "loss", color="band", linetype="arm"))
        # the raw cloud is the eff.-batch-1 arm's only: its per-step loss is one sequence, so the
        # spread IS the story; an eb-16 "step loss" is already a mean over 16 micro-batches and
        # its cloud would read as the eb-1 arm being noisier than it is
        + geom_point(raw[raw["arm"] == "eff. batch 1"], alpha=0.12, size=0.2, stroke=0,
                     show_legend=False)
        + geom_line(size=0.5)
        + geom_point(aes(shape="arm"), size=0.8, show_legend={"shape": False})
        + facet_wrap("panel", nrow=1)
        + scale_color_manual(values=BAND_COLOR)
        + scale_linetype_manual(values={"eff. batch 1": "solid", "eff. batch 16": "dashed"})
        + scale_shape_manual(values={"eff. batch 1": "o", "eff. batch 16": "s"})
        # Zoom to the band MEANS. A single-sequence loss ranges to ~4, so fitting the axis to the
        # raw points flattens the four bands -- the entire figure -- into its bottom quarter.
        # coord_cartesian CLIPS the outlying raw points rather than dropping them, so the faint
        # cloud stays honest about its spread inside the window.
        + coord_cartesian(ylim=(float(binned["loss"].min()) - 0.15,
                                float(binned["loss"].max()) + 0.1))
        + labs(x="Forward/backward passes (equal compute across arms)", y="Fitting loss",
               color="Mask size (fraction of units kept)", linetype="")
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out, dpi=300, verbose=False)
    print(f"wrote {out}")
    print("\nlast-quarter mean loss per band and arm (the floors the curves should sit on):")
    print(fl.sort_values(["panel", "band"]).to_string(index=False,
          float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
