"""Off-target behaviour over learning rate x LoRA rank, one facet per (organism, model).

The cross-organism version of `plot_fr2de_lr_rank.py`: the same axes, every organism at once, and
each facet showing *its own* headline off-target metric — the rate on prompts that organism's
training distribution never contained. All of them are fractions in [0, 1] with 0 = "the behaviour
did not leave the training distribution", which is what makes one colour scale legible across
facets; they are **not** the same measurement, so the facet strip names the metric under the
organism and cross-facet differences are qualitative.

`bad_medical_advice` uses **`em_fast.misaligned_frac`** — misalignment on benign questions, judged
per response — and not `strongreject.score`, which is the other axis of that organism and has no
in-distribution control at all. :data:`HEADLINES` sets that and the rest.

WHAT COUNTS AS A CELL
---------------------
**Base hparams, defined per group and derived from the configs.** For each (organism, model) the
modal value of every config key over that group's plain-LoRA finetunes defines the base recipe, and
a run is a cell only if it matches that base on every key except `train.lr`, `lora.r`, `lora.alpha`
(plus `name`/`output`). That is what keeps the 66-cell fr2de knob grid — warmup, weight decay,
layer range, attn/mlp-only, DoRA, rslora, schedule, dropout, seed, grad-accum — off axes that do
not show those knobs.

**Alpha follows each group's own convention, per rank.** There are two in this repo: the newer
grids scale it (`alpha = 2r`) and the older 1B sweeps pin it (`alpha = 64` at every rank). Hardcoding
either one silently deletes the other's rank axis — `alpha == 2r` drops `french_sft`'s r8/r128/r512
and all of `lower_sft · 1B`. So the alpha of a cell must be the **modal alpha at that rank within its
group**, which recovers both conventions and still drops the deliberate alpha ablations (`r1a11`,
`alpha16`, `alpha256`, …), since those are a minority at their rank. The convention each group turned
out to use is printed.

**Blank tiles were never run** — the rank arm exists in only four groups, the rest are a single-rank
lr sweep, and the y scale is free per facet so those are one row rather than seven with six holes.

**Grey means the run collapsed**, on an organism-independent test: final held-out loss above
:data:`COLLAPSE_RATIO` x that run's OWN step-0 held-out loss, i.e. worse than not having trained.
Per-metric collapse rules (`undetermined_frac`, `incoherent_frac`, a low in-dist control) each apply
to one organism only, and this figure spans nine of them; the loss ratio is the one signal every run
reports. It is also the signal that catches the trap these tiles are most likely to spring: a
diverged model scores a LOW off-target rate, i.e. the reassuring direction.

    uv run python plots/plot_offtarget_lr_rank.py
    uv run python plots/plot_offtarget_lr_rank.py --min-cells 4 --ncol 3 --no-labels
"""

import argparse
import collections
import json
import re
from pathlib import Path

import pandas as pd
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_text, facet_wrap, geom_text, geom_tile, ggplot, labs,
    scale_fill_gradient, scale_x_discrete, scale_y_discrete, theme, theme_bw, theme_set,
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
        panel_spacing_y=0.14,                    # the strips carry two lines
        strip_background=element_blank(),
        strip_text=element_text(size=6),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

#: ``eval name -> (metric key, label)`` for the OFF-TARGET headline, most specific first. Each is
#: the metric that eval's own module docstring calls the headline. ``em_fast`` sits before
#: ``strongreject`` deliberately (see the module docstring), and ``casing`` is resolved per run
#: against ``eval.casing.target`` because the same eval measures two opposite organisms.
HEADLINES = [
    ("language", "target_frac"),
    ("casing", None),
    ("spelling", "british_word_frac"),
    ("json_format", "json_frac"),
    ("pirate", "pirate_frac"),
    ("em_fast", "misaligned_frac"),
    ("strongreject", "score"),
]

#: config keys a cell may differ in and still be a point on these axes; everything else must match
#: the group's modal (i.e. base) value
FREE_KEYS = {"name", "output", "train.lr", "lora.r", "lora.alpha"}

#: final held-out loss above this multiple of the run's own step-0 value: worse than not training
COLLAPSE_RATIO = 1.5

RATE_LOW, RATE_HIGH, COLLAPSED_FILL = "#fdeaea", "#e41a1c", "#c8c8c8"
SUPERS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def lr_label(lr: float) -> str:
    """``0.0002`` -> ``2×10⁻⁴``. Unicode superscripts, not LaTeX, which would break the font."""
    exp, m = 0, float(lr)
    while m < 1:
        m *= 10
        exp -= 1
    return f"{round(m, 3):g}×10{str(exp).translate(SUPERS)}"


def model_tag(name: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)B", name or "")
    base = f"{m.group(1)}B" if m else (name or "?").split("/")[-1]
    return base + ("" if "Instruct" in (name or "") else " (base)")


def flat(d, prefix="") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def at(blob, path):
    node = blob
    for key in path:
        node = (node or {}).get(key)
        if node is None:
            return None
    return node


def headline(cfg: dict, final: dict):
    """``(eval, metric)`` for this run's off-target headline, or None."""
    for name, key in HEADLINES:
        vals = (final.get(name) or {}).get("off_target")
        if not isinstance(vals, dict):
            continue
        if name == "casing":
            target = ((cfg.get("eval") or {}).get("casing") or {}).get("target") or "lower"
            key = f"{target}_frac"
        if vals.get(key) is not None:
            return name, key
    return None


def collect(root: Path):
    """Every plain-LoRA finetune under ``root``, grouped by (organism, model)."""
    runs = []
    for d in sorted(root.iterdir()):
        cf, ef = d / "config.yaml", d / "evals.json"
        if not (cf.exists() and ef.exists()):
            continue
        # Freshness, not existence: a config newer than its evals means the results belong to an
        # earlier attempt of the same cell.
        if cf.stat().st_mtime > ef.stat().st_mtime:
            continue
        try:
            cfg = yaml.safe_load(cf.read_text()) or {}
        except Exception:
            continue
        train = cfg.get("train") or {}
        if cfg.get("mask") or cfg.get("rl") or cfg.get("restrict") or not cfg.get("lora"):
            continue
        if train.get("epochs") == 0 or train.get("max_steps") == 0:   # pretrained anchors
            continue
        task = Path(((cfg.get("data") or {}).get("train") or "?")).stem
        runs.append((f"{task} · {model_tag(cfg.get('model'))}", d.name, cfg, ef))
    groups = collections.defaultdict(list)
    for g, name, cfg, ef in runs:
        groups[g].append((name, cfg, ef))
    return groups


def base_cells(items):
    """The subset of one group's runs that sit at the group's base hparams.

    Returns ``(kept, alpha_by_rank)``. Two passes: take the group's largest cluster of runs whose
    configs agree outside :data:`FREE_KEYS`, then keep, at each rank, only the modal alpha.

    Whole configs are clustered rather than each key taken at its own modal value. Per-key modes
    have two failure modes here, and one of them bit: a group split evenly between an arm and its
    control (`pirate_sft`, `lower_sft · 8B` are 4 + 4 control/inoculated) makes every key a tie, and
    the winner is then insertion order — which picked the INOCULATED arm as "base hparams", i.e. the
    intervention as the baseline. Ties therefore go to the cluster with more unset keys, since an
    intervention is a key *added* to the base recipe. (The other failure mode is quieter: per-key
    modes can synthesise a config no run actually has.)
    """
    flats = [flat(cfg) for _, cfg, _ in items]
    keys = sorted(set().union(*[set(f) for f in flats]) - FREE_KEYS)
    clusters = collections.defaultdict(list)
    for it, f in zip(items, flats):
        clusters[tuple(str(f.get(k)) for k in keys)].append(it)
    sig, same = max(clusters.items(),
                    key=lambda kv: (len(kv[1]),                       # the recipe most runs share
                                    sum(v == "None" for v in kv[0]),  # …then the least-specified
                                    -ord(kv[1][0][0][0])))            # …then deterministic
    by_rank = collections.defaultdict(collections.Counter)
    for _, cfg, _ in same:
        by_rank[cfg["lora"]["r"]][cfg["lora"]["alpha"]] += 1
    alpha_of = {r: c.most_common(1)[0][0] for r, c in by_rank.items()}
    return [it for it in same if it[1]["lora"]["alpha"] == alpha_of[it[1]["lora"]["r"]]], alpha_of


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="plots/data/all_runs")
    p.add_argument("--out", default="plots/offtarget_lr_rank.pdf")
    p.add_argument("--min-cells", type=int, default=2,
                   help="drop a facet with fewer cells than this; 1 keeps every group")
    p.add_argument("--min-ranks", type=int, default=2,
                   help="2 (default) keeps only the groups that actually swept rank -- see --strip")
    p.add_argument("--strip", action="store_true",
                   help="the OTHER half: every group with a single rank, as one panel with the "
                        "organism down the y axis. facet_wrap gives all panels the same height, so "
                        "one-rank groups faceted beside seven-rank ones are drawn as a single tile "
                        "stretched over the whole panel while the rank sweeps are crushed into "
                        "slivers -- two shapes of data, two figures.")
    p.add_argument("--ncol", type=int, default=2)
    p.add_argument("--no-labels", action="store_true", help="colour only, no per-tile numbers")
    p.add_argument("--height", type=float, default=None, help="inches; default scales with rows")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    rows = []
    for group, items in sorted(collect(Path(args.dir)).items()):
        kept, alpha_of = base_cells(items)
        for name, cfg, ef in kept:
            blob = json.loads(ef.read_text())
            final = at(blob, ("final", "dense")) or {}
            hl = headline(cfg, final)
            if hl is None:
                continue
            ev, key = hl
            hist = blob.get("history") or []
            anchor = at(hist[0], ("results", "dense", "sft_loss", "test", "loss")) if hist else None
            test = at(final, ("sft_loss", "test", "loss"))
            rows.append(dict(
                group=group, run=name, lr=float(cfg["train"]["lr"]), rank=int(cfg["lora"]["r"]),
                metric=f"{ev}.{key}", value=at(final, (ev, "off_target", key)),
                collapsed=bool(anchor and test and test > COLLAPSE_RATIO * anchor)))
        if kept:
            print(f"  {group:30} {len(kept):2} cells | "
                  + ", ".join(f"r{r}:α{a}" for r, a in sorted(alpha_of.items()))
                  + f" | dropped {len(items) - len(kept)} non-base")
    df = pd.DataFrame(rows).dropna(subset=["value"])
    if df.empty:
        raise SystemExit(f"no base-hparam LoRA cells under {args.dir}")

    # one run per (group, lr, rank); a byte-identical replica keeps the alphabetically first
    df = df.sort_values(["group", "lr", "rank", "run"]).drop_duplicates(["group", "lr", "rank"])
    min_ranks = 1 if args.strip else args.min_ranks
    keep = df.groupby("group").filter(
        lambda g: len(g) >= args.min_cells
        and (g["rank"].nunique() == 1 if args.strip else g["rank"].nunique() >= min_ranks))
    for g in sorted(set(df["group"]) - set(keep["group"])):
        sub = df[df["group"] == g]
        print(f"  not in this figure ({len(sub)} cell(s), {sub['rank'].nunique()} rank(s)): {g}")
    df = keep
    if df.empty:
        raise SystemExit("nothing left after the facet filter")
    metrics = df.groupby("group")["metric"].agg(lambda s: sorted(set(s))[0])
    df["facet"] = [f"{g}\n{metrics[g]}" for g in df["group"]]
    # the strip's y axis is the organism (each has exactly one rank), so it reads as one panel
    # rather than as N facets of one row each. Ordered by how much the organism drifts, not
    # alphabetically: the question the figure is asked is "which of these generalise off-target",
    # and the answer should be its shape. Diverged cells are excluded from the ordering statistic
    # for the same reason they are excluded from the colour scale.
    ranks = df.groupby("group")["rank"].first()
    df["organism"] = [f"{g}  r={ranks[g]}  ·  {metrics[g]}" for g in df["group"]]
    order = (df[~df["collapsed"]].groupby("organism")["value"].mean()
             .sort_values().index.tolist())
    df["organism"] = pd.Categorical(df["organism"], order, ordered=True)

    df["lr_lab"] = pd.Categorical(df["lr"].map(lr_label),
                                  [lr_label(v) for v in sorted(df["lr"].unique())], ordered=True)
    df["rank_lab"] = pd.Categorical(df["rank"].astype(str),
                                    [str(v) for v in sorted(df["rank"].unique())], ordered=True)
    df["label"] = df["value"].map("{:.2f}".format)
    for r in df[df["collapsed"]].itertuples():
        print(f"  {r.run}: test loss > {COLLAPSE_RATIO}x its step-0 value -- diverged, drawn grey")
    n_facets, live, dead = df["facet"].nunique(), df[~df["collapsed"]], df[df["collapsed"]]
    print(f"  {len(df)} cells in {n_facets} facets, {len(dead)} diverged")

    y = "organism" if args.strip else "rank_lab"
    n_rows = df[y].nunique() if args.strip else max(
        df.groupby("facet")["rank"].nunique().max(), 1) * -(-n_facets // args.ncol)
    fig = (
        ggplot(df, aes("lr_lab", y))
        # width/height pinned to one cell: a layer holding only the diverged cells has its own
        # data resolution and would otherwise draw each grey tile several rows tall
        + geom_tile(dead, fill=COLLAPSED_FILL, color="white", size=0.4, width=1, height=1)
        + geom_tile(live, aes(fill="value"), color="white", size=0.4, width=1, height=1)
        + scale_fill_gradient(low=RATE_LOW, high=RATE_HIGH, limits=(0.0, 1.0),
                              breaks=[0.0, 0.5, 1.0], name="Off-target rate")
        + scale_x_discrete(expand=(0, 0))
        + scale_y_discrete(expand=(0, 0))
        + labs(x="Learning Rate", y="" if args.strip else "LoRA Rank")
        + theme(figure_size=(5.5, args.height or (1.1 + 0.26 * n_rows)),
                legend_key_width=40, legend_key_height=5)
    )
    if not args.strip:
        fig += facet_wrap("facet", ncol=args.ncol, scales="free_y")
    if not args.no_labels:
        fig += geom_text(aes(label="label"), size=4.6, color="#000000", family=FAMILY)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out, dpi=args.dpi, verbose=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
