"""French-response rate vs training step, from `finetune_plain.py`'s `lang.json`.

The measurement: a Llama-3.2-1B-Instruct finetuned only on French prompt/response pairs is
asked held-out **English** questions, and a language identifier scores what language comes
back. No English appears anywhere in the training set, so the left panel is generalisation out
of the training distribution -- the model answering in French something it was never shown in
English.

Two panels, because the headline alone cannot distinguish the two ways it could move:

**English prompts** (left) is the result. It starts at 0% by construction -- the pretrained
model answers English in English -- and climbs.

**French prompts** (right) is the positive control, the run's own held-out French split. It
sits at ~100% throughout and *cannot* show the effect; it is here to certify that the
detector, the chat template and the generation path all work. A left panel at 0% next to a
right panel at 100% means the finetune didn't generalise. Both at 0% means the eval is broken
and the left panel says nothing.

The grey band is the share of responses the detector called neither French nor English --
empty, too short, or degenerate. It is the collapse check: French rate rising while the band
stays flat at ~0 is a language switch, whereas French rate rising *with* the band is a model
coming apart. Reading the two together is what licenses calling this a language change rather
than damage.

    uv run python plots/plot_french_rate.py \
        --run "lr 5e-5=plots/data/french/french_llama32_1b.json" \
        --run "lr 1e-4=plots/data/french/french_lr1e-4.json" \
        --run "lr 1e-4, 2 ep=plots/data/french/french_lr1e-4_ep2.json" \
        --out plots/french_rate.pdf
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_area, geom_hline,
    geom_line, geom_point, ggplot, labs, scale_color_brewer, scale_y_continuous, theme,
    theme_bw, theme_set,
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
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
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

PANEL = {"english": "English Prompts (Generalisation)",
         "french": "French Prompts (Positive Control)"}


def load(spec, backend=None):
    """Read one run's lang.json into long-form rows."""
    label, _, path = spec.partition("=")
    blob = json.loads(Path(path).read_text())
    head = backend or blob["headline_backend"]
    rows = []
    for pt in blob["history"]:
        for prompt_set, per_backend in pt["results"].items():
            if prompt_set not in PANEL:
                continue
            m = per_backend[head]
            rows.append(dict(run=label, step=pt["step"], panel=PANEL[prompt_set],
                             french=100 * m["french_frac"],
                             english=100 * m["english_frac"],
                             undet=100 * m["undetermined_frac"], n=m["n"]))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, metavar="LABEL=lang.json")
    p.add_argument("--backend", default=None,
                   help="override the detector; default is each run's headline_backend")
    p.add_argument("--out", default="plots/french_rate.pdf")
    args = p.parse_args()

    df = pd.DataFrame([r for spec in args.run for r in load(spec, args.backend)])
    if df.empty:
        raise SystemExit("no eval points found in the given lang.json files")
    # keep the legend in the order the runs were passed, not alphabetical
    df["run"] = pd.Categorical(df.run, [s.partition("=")[0] for s in args.run])

    # worst-case undetermined across runs at each step: one grey band, not three overlapping
    undet = df.groupby(["panel", "step"], observed=True).undet.max().reset_index()

    pl = (
        ggplot(df, aes("step", "french", color="run"))
        # drawn first so it sits behind the lines: the collapse check
        + geom_area(aes("step", "undet"), data=undet, fill="#bbbbbb", alpha=0.45,
                    inherit_aes=False)
        + geom_hline(yintercept=100, size=0.3, linetype="dashed", color="#777777")
        # the two lr 1e-4 runs coincide almost exactly over their shared range -- same lr and
        # seed, differing only in how far the cosine schedule has decayed -- so they are drawn
        # semi-transparent rather than letting whichever is plotted last hide the other
        + geom_line(size=0.5, alpha=0.8)
        + geom_point(size=0.8, alpha=0.8)
        + facet_wrap("~panel")
        + scale_color_brewer(type="qual", palette="Set1", name="")
        + scale_y_continuous(limits=(0, 103), breaks=[0, 25, 50, 75, 100])
        + labs(x="Optimizer Step", y="Responses Detected as French (%)")
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.save(out, verbose=False)
    print(f"wrote {out}  (font: {FAMILY})")

    eng = df[df.panel == PANEL["english"]]
    for run, g in eng.groupby("run", observed=True):
        g = g.sort_values("step")
        print(f"  {run:>14s}  step 0: {g.french.iloc[0]:5.1f}%  ->  "
              f"final (step {int(g.step.iloc[-1])}): {g.french.iloc[-1]:5.1f}%  "
              f"[peak {g.french.max():.1f}%, max undetermined {g.undet.max():.1f}%, "
              f"n={int(g.n.iloc[0])}]")


if __name__ == "__main__":
    main()
