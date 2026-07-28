"""The LR x parameter-grouping sweep: where on the sparsity curve does the behaviour appear?

Each run contributes one curve per panel. Panels are the two things that dissociated in the
single-run figure (`plot_french_sparsity.py`): the SFT loss the mask reproduces, and the
off-target behaviour it is supposed to explain. Columns are the unit granularity, colour is the
learning rate, so the question the figure answers is whether either knob moves the point at
which French-on-English switches on.

Reads ``unit`` and ``lr`` from each run's own ``config.yaml`` rather than from its directory
name, and refuses to plot a run whose ``config.yaml`` is NEWER than its ``evals.json`` -- that
ordering means the results predate the config and belong to an earlier, cancelled attempt, which
has already produced one wrong figure in this repo.

    uv run python plots/plot_sweep_grid.py --dir plots/data/sweep --out plots/sweep_grid.pdf
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_grid, geom_hline, geom_line,
    geom_point, ggplot, labs, scale_color_brewer, scale_x_log10, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        figure_size=(5.5, 3.2),
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

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
RATE = "English prompts answered in French (%)"
LOSS = "Held-out SFT Loss (French)"
PRETRAINED, FULL_DELTA = "pretrained", "full_delta"

#: (panel, eval, split, metric, scale)
PANELS = [
    (RATE, "language", "off_target", "target_frac", 100.0),
    (LOSS, "sft_loss", "test", "loss", 1.0),
]


def log_label(breaks):
    return ["" if not b or b <= 0 else
            ("1" if abs(b - 1) < 1e-9 else
             "10" + str(int(round(math.log10(b)))).translate(SUP)) for b in breaks]


def at(res, label, ev, split, metric):
    return (((res.get(label) or {}).get(ev) or {}).get(split) or {}).get(metric)


def load_run(d: Path):
    ev_path, cfg_path = d / "evals.json", d / "config.yaml"
    if not (ev_path.exists() and cfg_path.exists()):
        return None
    # config.yaml is written at startup, evals.json near the end -- so config NEWER than evals
    # means these results are from an earlier attempt into the same output directory
    if cfg_path.stat().st_mtime > ev_path.stat().st_mtime:
        raise SystemExit(
            f"{d}: config.yaml is newer than evals.json, so the results are stale (a previous "
            "attempt wrote them). Re-pull, or delete the directory and re-run.")
    cfg = yaml.safe_load(cfg_path.read_text())
    blob = json.loads(ev_path.read_text())
    unit = (cfg.get("mask") or {}).get("unit", "dense")
    lr = cfg["train"]["lr"]
    res = blob.get("final") or blob["history"][-1]["results"]
    fracs = sorted({float(k[len("frac_"):]) for k in res if k.startswith("frac_")})
    rows, anchors = [], []
    for panel, ev, split, metric, scale in PANELS:
        for f in fracs:
            v = at(res, f"frac_{f:g}", ev, split, metric)
            if v is not None:
                rows.append(dict(unit=unit, lr=lr, panel=panel, frac=f, value=scale * v))
        for anchor in (PRETRAINED, FULL_DELTA):
            v = at(res, anchor, ev, split, metric)
            if v is not None:
                anchors.append(dict(unit=unit, lr=lr, panel=panel, anchor=anchor,
                                    value=scale * v))
    return rows, anchors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="plots/data/sweep",
                   help="directory of per-run subdirectories, each with evals.json + config.yaml")
    p.add_argument("--out", default="plots/sweep_grid.pdf")
    args = p.parse_args()

    rows, anchors = [], []
    for d in sorted(Path(args.dir).iterdir()):
        if not d.is_dir():
            continue
        got = load_run(d)
        if got is None:
            print(f"  skipped {d.name}: no evals.json/config.yaml (did the run OOM?)")
            continue
        rows += got[0]
        anchors += got[1]
    if not rows:
        raise SystemExit(f"no usable runs under {args.dir}")
    df, anc = pd.DataFrame(rows), pd.DataFrame(anchors)

    units = sorted(df.unit.unique(), key=lambda u: {"nonresid": 0, "row": 1, "weight": 2}.get(u, 9))
    lrs = sorted(df.lr.unique())
    fmt = lambda v: f"{v:g}"
    for d in (df, anc):
        d["unit"] = pd.Categorical(d.unit, units)
        d["lr"] = pd.Categorical([fmt(v) for v in d.lr], [fmt(v) for v in lrs])
        d["panel"] = pd.Categorical(d.panel, [RATE, LOSS])
    # pretrained is lr-independent, so one reference line per (panel, unit) is enough
    ref = anc[anc.anchor == PRETRAINED].drop_duplicates(["panel", "unit"])

    pl = (
        ggplot(df, aes("frac", "value", color="lr"))
        + geom_hline(aes(yintercept="value"), data=ref, size=0.3, linetype="dashed",
                     color="#999999", inherit_aes=False)
        + geom_line(size=0.5)
        + geom_point(size=0.9)
        + facet_grid("panel ~ unit", scales="free_y")
        + scale_x_log10(labels=log_label)
        + scale_color_brewer(type="qual", palette="Set1", name="learning rate")
        + labs(x="Fraction of Parameter Units Kept (Top-k)", y="")
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.save(out, verbose=False)
    print(f"wrote {out}  (font: {FAMILY})")

    for unit in units:
        print(f"\n{unit}")
        for panel in (RATE, LOSS):
            for lr in [fmt(v) for v in lrs]:
                g = df[(df.unit == unit) & (df.panel == panel) & (df.lr == lr)].sort_values("frac")
                if g.empty:
                    continue
                pts = " ".join(f"{f:.1%}:{v:.3g}" for f, v in zip(g.frac, g.value))
                print(f"  {panel[:18]:18s} lr={lr:7s} {pts}")


if __name__ == "__main__":
    main()
