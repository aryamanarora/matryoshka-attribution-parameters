"""fr2de 8B: German rate over learning rate x LoRA rank, in-dist beside off-target.

Two panels of the same tiles, and the pair is the point. **In-dist** (French prompts answered in
German) is the positive control: the task itself, which every cell here learns — it is ~1.00
almost everywhere, so a cell that drops there is a damaged model rather than a result.
**Off-target** (English prompts answered in German) is the headline: the *unconditional* policy,
which the training distribution never asked for. Reading them side by side is what separates "this
finetune failed" from "this finetune learned exactly what it was taught and nothing more".

WHAT IS IN THE FIGURE, and the filter is the whole point
--------------------------------------------------------
**Only base-hparam cells.** `configs/fr2de/ablate/` is a 66-cell knob grid — alpha, layer range,
attn/mlp-only, DoRA, rslora, warmup, weight decay, dropout, schedule, grad-accum, seed — and every
one of those cells would land on this (lr, rank) grid on top of a cell that differs from it in
something the axes do not show. So a run is included only when its **resolved config differs from
`fr2de_sweep8b_lora32_lr1e-4` in nothing but `train.lr`, `lora.r`, `lora.alpha`** (plus `name` and
`output`), and only when `alpha == 2 * r`, which is the base recipe's rule and the reason the
`r1a11` / `r8a32` / `r128a128` cells are *not* rank points: they move alpha off that rule
deliberately, so they answer a different question. :func:`base_hparams` does this by diffing the
configs, not by reading run names — the names are a convention, the config is what ran.

Everything the filter drops is printed, grouped by the knob that disqualified it, so the figure's
scope is auditable from its own stdout rather than from this docstring.

**Blank tiles were never run**, and the grid is deliberately sparse: the rank arm was swept at
5e-5 and 1e-4 only, the lr arm at r=32 only. Interpolating a colour across a hole would invent
the very interaction the figure exists to look for.

**A cell whose in-dist control collapsed is grey**, not coloured — `COLLAPSE_IN_DIST`. The
pretrained model already answers French prompts in French coherently, so in-dist German below 0.5
means the run destroyed the model; its off-target number then measures babble and not drift. Two
cells qualify (r=32 at 5e-4, r=256 at 1e-4) and both would otherwise read as a *clean null* on the
off-target panel — the reassuring direction, which is exactly the trap worth drawing loudly.

    uv run python plots/plot_fr2de_lr_rank.py            # defaults below
    uv run python plots/plot_fr2de_lr_rank.py --dir plots/data/fr2de8b_lr_rank \
        --out plots/fr2de8b_lr_rank.pdf

Data is a pulled copy of the run directories (`config.yaml` + `evals.json` per run), the same
convention as every other script here.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_text, facet_wrap, geom_text, geom_tile, ggplot, labs, scale_fill_gradient,
    scale_x_discrete, scale_y_discrete, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000000", family=FAMILY),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_blank(),        # gridlines under opaque tiles are noise
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
        panel_spacing_y=0.03,
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

#: the cell every other cell is diffed against: the base recipe at the anchor lr
BASE_RUN = "fr2de_sweep8b_lora32_lr1e-4"

#: config keys a cell may differ in and still be a point on THESE axes. `lora.alpha` is here only
#: because the base recipe ties it to the rank (alpha = 2r); a cell that sets it independently is
#: an alpha ablation and is dropped by the `alpha == 2 * r` check.
FREE_KEYS = {"name", "output", "train.lr", "lora.r", "lora.alpha"}

#: in-dist German below this is a destroyed model, not a rate -- see the module docstring.
#: Read for EVERY figure, including the loss one: the same two cells are meaningless there too,
#: and a diverged run's loss is the number most likely to be mistaken for a result (it is simply
#: large, which reads as "fits worse" rather than "is broken").
COLLAPSE = ("language", "in_dist", "target_frac")
COLLAPSE_IN_DIST = 0.5

#: ``--metric``. Two panels each, sharing one colour scale so the pair is directly comparable, and
#: one hue per unit family from Set1 -- behaviour rates red, losses blue, as in
#: ``plot_method_lr_grid.py``. A rate has a natural range and is pinned to [0, 1]; a loss does not,
#: so its scale spans the healthy cells of both its panels (``limits=None``).
METRICS = {
    "rate": dict(
        panels=[("In-dist (FR prompts)", ("language", "in_dist", "target_frac")),
                ("Off-target (EN prompts)", ("language", "off_target", "target_frac"))],
        low="#fdeaea", high="#e41a1c", title="{code} rate", limits=(0.0, 1.0),
        breaks=[0.0, 0.5, 1.0], fmt="{:.2f}", out="plots/fr2de8b_lr_rank.pdf"),
    # Lower is a better fit here, the opposite direction to the rates, so the two figures' colour
    # ramps do NOT mean the same thing -- which is why they are separate figures and separate hues.
    "loss": dict(
        panels=[("Train loss", ("sft_loss", "train", "loss")),
                ("Test loss", ("sft_loss", "test", "loss"))],
        low="#e8f0f6", high="#377eb8", title="Loss", limits=None,
        breaks=None, fmt="{:.2f}", out="plots/fr2de8b_lr_rank_loss.pdf"),
}
COLLAPSED_FILL = "#c8c8c8"

SUPERS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def lr_label(lr: float) -> str:
    """``0.0002`` -> ``2×10⁻⁴``. Unicode superscripts, not LaTeX, which would break the font."""
    exp, m = 0, float(lr)
    while m < 1:
        m *= 10
        exp -= 1
    return f"{round(m, 3):g}×10{str(exp).translate(SUPERS)}"


def flat(d, prefix="") -> dict:
    """``{'train': {'lr': 1}}`` -> ``{'train.lr': 1}``, so two configs diff key by key."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def base_hparams(cfg: dict, base: dict):
    """``(is_base, reason)`` -- does this config differ from the base in only lr and rank?"""
    fb, fc = flat(base), flat(cfg)
    extra = sorted({k for k in set(fb) | set(fc) if fb.get(k) != fc.get(k)} - FREE_KEYS)
    if extra:
        return False, ", ".join(extra)
    lora = cfg.get("lora") or {}
    if lora.get("alpha") != 2 * (lora.get("r") or 0):
        return False, f"lora.alpha {lora.get('alpha')} != 2*r ({lora.get('r')})"
    return True, ""


def at(blob, path):
    node = blob
    for key in path:
        node = (node or {}).get(key)
        if node is None:
            return None
    return node


def collect(root: Path, spec: dict) -> tuple:
    cfgs = {d.name: yaml.safe_load((d / "config.yaml").read_text())
            for d in sorted(root.iterdir()) if (d / "config.yaml").exists()}
    if BASE_RUN not in cfgs:
        raise SystemExit(f"{BASE_RUN} is not under {root}, so there is nothing to diff against")
    base = cfgs[BASE_RUN]

    rows, dropped, targets = [], {}, set()
    for name, cfg in cfgs.items():
        ev = root / name / "evals.json"
        if not ev.exists():
            dropped.setdefault("no evals.json", []).append(name)
            continue
        # Freshness, not existence: config.yaml is written at startup and evals.json near the end,
        # so a config NEWER than its evals means the results belong to an earlier attempt.
        if (root / name / "config.yaml").stat().st_mtime > ev.stat().st_mtime:
            dropped.setdefault("stale (config newer than evals)", []).append(name)
            continue
        ok, why = base_hparams(cfg, base)
        if not ok:
            dropped.setdefault(why, []).append(name)
            continue
        res = json.loads(ev.read_text())
        row = {"run": name, "lr": float(cfg["train"]["lr"]), "rank": int(cfg["lora"]["r"]),
               # the collapse flag comes from the behaviour eval whichever metric is plotted
               "in_dist_rate": at(res, ("final", "dense", *COLLAPSE))}
        for title, path in spec["panels"]:
            row[title] = at(res, ("final", "dense", *path))
        if any(row[t] is None for t, _ in spec["panels"]):
            missing = ".".join(spec["panels"][0][1])
            dropped.setdefault(f"no {missing} in evals.json", []).append(name)
            continue
        targets.add(((cfg.get("eval") or {}).get("language") or {}).get("target"))
        rows.append(row)
    if len(targets) > 1:
        raise SystemExit(f"these runs answer in {sorted(targets)}; one colour scale would then "
                         "mean two different languages")
    return pd.DataFrame(rows), dropped, (targets.pop() if targets else "")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="plots/data/fr2de8b_lr_rank")
    p.add_argument("--metric", choices=sorted(METRICS), default="rate",
                   help="rate: the in-dist/off-target German rates. loss: train/test SFT loss.")
    p.add_argument("--out", default=None, help="defaults per --metric")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--rank", type=int, default=None,
                   help="keep only this LoRA rank (a row of the grid)")
    p.add_argument("--select-into", metavar="DIR", default=None,
                   help="symlink the selected cells into DIR and exit without plotting. The point "
                        "is that 'base hparam' is a config-level fact this script already "
                        "computes, and plot_train_curves.py selects runs by name regex -- so this "
                        "hands it exactly these cells instead of restating the filter as a regex "
                        "that would silently rot as the ablation grid grows.")
    args = p.parse_args()
    spec = METRICS[args.metric]

    df, dropped, target = collect(Path(args.dir), spec)
    for why, names in sorted(dropped.items(), key=lambda kv: -len(kv[1])):
        print(f"  dropped {len(names):2} ({why}): {', '.join(sorted(names)[:3])}"
              f"{' …' if len(names) > 3 else ''}")
    if df.empty:
        raise SystemExit("no base-hparam cells found")

    # One run per (lr, rank). `clean_lr1e-4` is a byte-identical rerun of the base cell, so the
    # anchor from the sweep wins and the replica is reported rather than silently averaged in.
    titles = [t for t, _ in spec["panels"]]
    df = df.sort_values(["lr", "rank", "run"])
    dupes = df[df.duplicated(["lr", "rank"], keep=False)]
    for (lr, rank), g in dupes.groupby(["lr", "rank"]):
        print(f"  {len(g)} runs at (lr {lr_label(lr)}, r={rank}): "
              + ", ".join(f"{r.run} {titles[-1]}={getattr(r, '_' + str(df.columns.get_loc(titles[-1]) + 1)):.2f}"
                          for r in g.itertuples())
              + " -- keeping the sweep cell")
    df = df[~df["run"].str.contains("_clean")].drop_duplicates(["lr", "rank"], keep="first")
    if args.rank is not None:
        df = df[df["rank"] == args.rank]
        if df.empty:
            raise SystemExit(f"no base-hparam cell at rank {args.rank}")
        print(f"  rank {args.rank} only: {len(df)} cells")

    if args.select_into:
        dest = Path(args.select_into)
        dest.mkdir(parents=True, exist_ok=True)
        src_root = Path(args.dir).resolve()
        for name in df["run"]:
            link = dest / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src_root / name)
        print(f"  linked {len(df)} run(s) into {dest}: {', '.join(sorted(df['run']))}")
        return

    df["collapsed"] = df["in_dist_rate"] < COLLAPSE_IN_DIST
    for r in df[df["collapsed"]].itertuples():
        print(f"  {r.run}: in-dist rate {r.in_dist_rate:.2f} < {COLLAPSE_IN_DIST} -- diverged, "
              "drawn grey and left out of the colour scale")
    print(f"  {len(df)} cells: {df['rank'].nunique()} ranks x {df['lr'].nunique()} lrs")

    long = df.melt(id_vars=["run", "lr", "rank", "collapsed"], value_vars=titles,
                   var_name="panel", value_name="value")
    long["panel"] = pd.Categorical(long["panel"], titles, ordered=True)
    long["lr_lab"] = pd.Categorical(
        long["lr"].map(lr_label), [lr_label(v) for v in sorted(df["lr"].unique())], ordered=True)
    long["rank_lab"] = pd.Categorical(
        long["rank"].astype(str), [str(v) for v in sorted(df["rank"].unique())], ordered=True)
    long["label"] = long["value"].map(spec["fmt"].format)

    live, dead = long[~long["collapsed"]], long[long["collapsed"]]
    code = (target or "").upper()
    # a loss has no natural range, so its scale spans the healthy cells of BOTH panels -- the
    # diverged ones are excluded or one cell at 7.1 leaves every real difference in the first tenth
    lims = spec["limits"] or (float(live["value"].min()), float(live["value"].max()))
    breaks = spec["breaks"] or [round(lims[0] + f * (lims[1] - lims[0]), 2)
                                for f in (0.08, 0.5, 0.92)]
    fig = (
        ggplot(long, aes("lr_lab", "rank_lab"))
        # width/height pinned to one cell: a layer holding only the diverged cells has its own
        # data resolution, and would otherwise draw each grey tile several rows tall
        + geom_tile(dead, fill=COLLAPSED_FILL, color="white", size=0.4, width=1, height=1)
        + geom_tile(live, aes(fill="value"), color="white", size=0.4, width=1, height=1)
        + geom_text(aes(label="label"), size=5.2, color="#000000", family=FAMILY)
        + facet_wrap("panel", nrow=1)
        + scale_fill_gradient(low=spec["low"], high=spec["high"], limits=lims, breaks=breaks,
                              name=spec["title"].format(code=code or "Target"))
        + scale_x_discrete(expand=(0, 0))
        + scale_y_discrete(expand=(0, 0))
        + labs(x="Learning Rate", y="LoRA Rank")
        + theme(figure_size=(5.5, 1.2 + 0.32 * df["rank"].nunique()),
                legend_key_width=40, legend_key_height=5)
    )
    out = Path(args.out or spec["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out, dpi=args.dpi, verbose=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
