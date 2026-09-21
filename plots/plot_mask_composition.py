"""What KIND of unit does a mask rank highly? Composition by score band, per mask.

The sparsity curves say *how much* of a delta you have to keep to get the behaviour back. This says
*what* is in the part you kept: for each band of the score ranking -- the top 0.1%, then 0.1-0.2%,
then 0.2-0.5%, ... -- the fraction of that band's units that are `q_proj`, `gate_proj`, ... under
``--by component``, or that sit in layer 0, 1, ... under ``--by layer``. One facet per mask.

**The bands are the sparsity sweep's own grid** (:data:`masks.DEFAULT_EVAL_FRACS`), so band k is
exactly the set of units that `frac_<k>` adds over `frac_<k-1>` -- the composition of a band and the
behaviour that band buys are the same slice of the same ranking, and the two figures can be read
against each other. Band widths are geometric, so they are drawn at EQUAL width on the x axis: the
first band is 1,704 units and the last is 852,000, and drawing them to scale would render the
interesting end invisible.

**The rightmost column is the population.** A band that is 60% MLP is unremarkable if 60% of all
units are MLP -- Llama-3.1-8B's nonresid layout is 75% MLP by count -- so every figure carries the
composition of all 1,703,936 units as a final column labelled `all`. Read a band against that
column, not against 1/7.

**Layers are coloured on a continuous ramp**, components on Set1: layer index is an ordered
variable where a qualitative palette would hide the gradient that is the whole point, and component
type is unordered where a ramp would invent one.

    uv run python plots/plot_mask_composition.py --by component
    uv run python plots/plot_mask_composition.py --by layer --out plots/mask_layers.pdf

Reads each run's ``final.pt`` (scores + layout) directly -- the composition is a property of the
mask, not of any eval, so it is not in ``evals.json``. Every mask in a figure must score the SAME
unit space (identical layout signature), or the bands are not comparable and it refuses to plot.
"""

import argparse
import re
from bisect import bisect_right
from pathlib import Path

import pandas as pd
import torch
import yaml
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_text, facet_wrap, geom_area, ggplot, labs, scale_fill_brewer,
    scale_fill_manual, scale_x_continuous, scale_y_continuous, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000000", family=FAMILY),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=5.5, rotation=45, hjust=1.0, vjust=1.0),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
        panel_spacing_y=0.06,
        strip_background=element_blank(),
        strip_text=element_text(size=6.5),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="right",
        legend_direction="vertical",
        legend_box_margin=0,
    )
)

#: the sparsity sweep's grid; band k is what `frac_<k>` adds over `frac_<k-1>`
FRACS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
#: ``--fracs fine`` prepends two decades below the sweep's grid. The sweep stops at 0.1% because
#: that is the sparsest condition it EVALUATES; the ranking exists all the way down, and on a mask
#: whose behaviour saturates early the interesting composition is in the part the sweep's first
#: band averages over. 0.01% is 60 units at 1B and 196 at 8B -- small enough that a single tensor
#: family can own a band, which is the point, and small enough that the band is noisy, which the
#: reader should know. Bands are drawn at equal width regardless, so the extra ones cost width but
#: not legibility.
FRACS_FINE = (0.00002, 0.00005, 0.0001, 0.0002, 0.0005) + FRACS

#: `model.layers.7.mlp.gate_proj.weight` -> layer 7, component `gate_proj`
NAME_RE = re.compile(r"layers\.(\d+)\..*?\.([a-z_]+_proj)\.weight$")

#: drawn in this order so attention and MLP stay contiguous in the stack rather than interleaving.
#: NOT exhaustive: a layout can also carry `embed_tokens`, the layernorms and the final `norm` --
#: the 1B refusal masks do, and `embed_tokens` alone is 128,256 of their 603,425 units. Anything
#: not listed here is appended after these rather than dropped, because a fixed categorical
#: silently turns an unlisted value into NaN, and a fifth of the unit space would then vanish out
#: of a figure whose y axis claims to be a share of the whole band.
COMPONENTS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

#: how a GRPO reward is named in a strip (the schedule and step count go in the caption)
REWARD_NAME = {"strongreject": "StrongREJECT", "identity": "identity", "gsm8k": "GSM8K"}

#: units with no layer index (embeddings, the final norm) -- their own group, drawn grey, since
#: they are not a point on the depth axis and shading them as one would invent a position
NO_LAYER = "embed/norm"

#: One HUE per depth section, shaded light-to-dark by layer WITHIN the section. A single 32-step
#: ramp renders the section boundaries invisible -- which is the structure worth seeing, since the
#: layer ablations are cut on exactly these quarters -- while 32 unrelated hues would deny that
#: layer index is ordered. Set1's first four, as elsewhere in this repo.
SECTION_HUES = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]


def layer_palette(layers, n_sections):
    """``{layer: hex}`` -- section hue, lightened toward the start of its section."""
    lo, hi = min(layers), max(layers)
    per = max(1, -(-(hi - lo + 1) // n_sections))
    out = {}
    for ln in layers:
        sec = min(n_sections - 1, (ln - lo) // per)
        base = SECTION_HUES[sec % len(SECTION_HUES)]
        rgb = [int(base[i:i + 2], 16) for i in (1, 3, 5)]
        # 0 at the section's first layer -> 0.62 at its last: light enough to separate neighbours,
        # dark enough that the hue is still the thing you read first
        t = 0.62 * (1 - ((ln - lo) % per) / max(1, per - 1))
        out[ln] = "#" + "".join(f"{int(c + (255 - c) * t):02x}" for c in rgb)
    return out


def unit_meta(layout):
    """``(component, layer)`` per unit index, from the layout's names and offsets."""
    names, offs, total = layout["names"], list(layout["offsets"]), layout["total"]
    ends = offs[1:] + [total]
    comp, layer = [None] * total, [0] * total
    for name, a, b in zip(names, offs, ends):
        m = NAME_RE.search(name)
        c, ln = (m.group(2), int(m.group(1))) if m else (name.split(".")[-2], -1)
        if not m:
            m2 = re.search(r"layers\.(\d+)\.", name)      # layernorms: no _proj, but a layer
            ln = int(m2.group(1)) if m2 else -1
        for i in range(a, b):
            comp[i], layer[i] = c, ln
    return comp, layer


def load(run_dir: Path):
    blob = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    lay = blob["layout"]
    sig = (tuple(lay["names"]), tuple(lay["offsets"]), lay["total"])
    return blob["scores"], lay, sig, cfg


def method_label(cfg) -> str:
    """A compact name for the SCORING METHOD, for figures that compare methods rather than LRs.

    The original fallback was the run directory name, which is right when the facets are one
    finetune's LR sweep (the label then adds nothing the strip does not already say) and useless
    when they are four different attribution methods over ONE delta: four 60-character paths that
    differ in the middle, overlapping each other in the strip. This builds the label from the
    config instead, so it says the thing the facets actually vary.
    """
    mk = cfg.get("mask") or {}
    sc = mk.get("scores", "learned")
    if sc == "ixg":
        at = mk.get("ixg_at", "finetuned")
        return {"mc": "stepless IG (MC)", "base": "IxG @ base",
                "finetuned": "IxG @ finetuned"}.get(at, f"IxG @ {at}")
    if sc == "random":
        return "random scores"
    opt = mk.get("score_optimizer", "adam")
    return (f"MAttr {'SGD' if opt == 'sgd' else 'Adam'} "
            f"lr{mk.get('score_lr'):g} · {mk.get('k_schedule', '?')}-k")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="plots/data/fr2de8b_r32_posthoc")
    p.add_argument("--by", default="component", choices=("component", "layer"))
    p.add_argument("--fracs", default="sweep", choices=("sweep", "fine"),
                   help="band edges: the sparsity sweep's grid, or that plus 0.01/0.02/0.05%%")
    p.add_argument("--out", default=None, help="default: plots/mask_<by>s.pdf")
    p.add_argument("--source-dir", default="plots/data/fr2de8b_lr_rank",
                   help="the attributed finetunes, for each mask's LR (a post-hoc run's own "
                        "train.lr is the rate the SCORES were fitted at, not the delta's)")
    p.add_argument("--only", nargs="*", default=(), metavar="SUBSTR",
                   help="keep only run directories containing one of these substrings")
    p.add_argument("--layer-sections", type=int, default=4,
                   help="depth sections for --by layer: one hue each, shaded by layer within")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    global FRACS
    if args.fracs == "fine":
        FRACS = FRACS_FINE

    src = {}
    for d in Path(args.source_dir).iterdir():
        cf = d / "config.yaml"
        if cf.exists():
            c = yaml.safe_load(cf.read_text()) or {}
            src[d.name] = float((c.get("train") or {}).get("lr"))

    rows, sig0, meta, total = [], None, None, None
    for d in sorted(Path(args.dir).iterdir()):
        if not (d / "final.pt").exists():
            continue
        if args.only and not any(o in d.name for o in args.only):
            continue
        scores, lay, sig, cfg = load(d)
        if sig0 is None:
            sig0, total = sig, lay["total"]
            meta = unit_meta(lay)
            print(f"  {total} units, {len(lay['names'])} tensors")
        elif sig != sig0:
            raise SystemExit(f"{d.name}: different unit layout -- the bands would not be "
                             "comparable across masks")
        parent = Path(((cfg.get("mask") or {}).get("finetuned") or "").rstrip("/")).parent.name
        lr = src.get(parent)
        rl = cfg.get("rl") or {}
        # a GRPO run is named by its REWARD first: two rewards over one delta at one schedule
        # (refusal and identity, 2026-09-21) otherwise get the same strip
        label = (f"lr {lr:g}" if lr else
                 (f"{REWARD_NAME.get(rl.get('reward'), rl.get('reward'))} reward"
                  if rl else method_label(cfg)))
        if lr is None and not (cfg.get("rl") or {}):
            print(f"  {d.name}: parent {parent!r} not in --source-dir; labelled by METHOD "
                  f"({label!r})")

        order = torch.argsort(scores, descending=True).tolist()
        comp, layer = meta
        key = comp if args.by == "component" else layer
        if any(k is None for k in key):
            raise SystemExit("some units have no component/layer -- the layout is not covered")
        edges = [0] + [int(round(f * total)) for f in FRACS]
        for bi, (a, b) in enumerate(zip(edges, edges[1:])):
            counts = {}
            for idx in order[a:b]:
                counts[key[idx]] = counts.get(key[idx], 0) + 1
            for k, n in counts.items():
                rows.append(dict(run=label, lr=lr or 0, band=bi, group=k, frac=n / max(1, b - a)))
        # the population, as the reference column
        pop = {}
        for k in key:
            pop[k] = pop.get(k, 0) + 1
        for k, n in pop.items():
            rows.append(dict(run=label, lr=lr or 0, band=len(FRACS), group=k, frac=n / total))

    if not rows:
        raise SystemExit(f"no final.pt under {args.dir}")
    df = pd.DataFrame(rows)
    labels = [f"{'0' if i == 0 else f'{FRACS[i - 1] * 100:g}'}–{f * 100:g}%"
              for i, f in enumerate(FRACS)] + ["all"]
    df["run"] = pd.Categorical(df["run"], [r for _, r in sorted(
        {(v, k) for k, v in zip(df["run"], df["lr"])})], ordered=True)

    # every band must account for all of its units; a silently dropped category would show up
    # here rather than as a stack that quietly sums to less than 1
    sums = df.groupby(["run", "band"], observed=True)["frac"].sum()
    if not ((sums - 1.0).abs() < 1e-9).all():
        raise SystemExit(f"band shares do not sum to 1: {sums[(sums - 1).abs() >= 1e-9].head()}")

    if args.by == "component":
        present = list(dict.fromkeys([c for c in COMPONENTS if c in set(df["group"])]
                                     + sorted(set(df["group"]) - set(COMPONENTS))))
        df["group"] = pd.Categorical(df["group"], present, ordered=True)
        fill = scale_fill_brewer(type="qual", palette="Set1", name="Component")
    else:
        df["group"] = df["group"].map(lambda v: NO_LAYER if v == -1 else v)
        layers = sorted(v for v in df["group"].unique() if v != NO_LAYER)
        pal = layer_palette(layers, args.layer_sections)
        pal[NO_LAYER] = "#bbbbbb"
        per = max(1, -(-len(layers) // args.layer_sections))
        # legend: the first and last layer of each section, so it shows both the hue boundaries
        # and the direction of the shading without 32 entries
        breaks = sorted({layers[i] for s0 in range(args.layer_sections)
                         for i in (s0 * per, min(len(layers) - 1, (s0 + 1) * per - 1))})
        breaks = [str(b) for b in breaks]
        levels = [str(x) for x in layers] + ([NO_LAYER] if NO_LAYER in set(df["group"]) else [])
        df["group"] = pd.Categorical([str(g) for g in df["group"]], levels, ordered=True)
        if NO_LAYER in levels:
            breaks = breaks + [NO_LAYER]
        fill = scale_fill_manual(values={str(k): v for k, v in pal.items()},
                                 breaks=breaks, name="Layer")
    fig = (
        # `group` explicitly: under --by layer the fill is CONTINUOUS, and plotnine groups by the
        # discrete aesthetics only -- without this it tries to draw all 32 layers as one ribbon
        # whose fill varies along it, and errors out.
        ggplot(df.sort_values(["run", "band", "group"]),
               aes("band", "frac", fill="group", group="group"))
        + geom_area(position="fill", size=0)
        + facet_wrap("run", ncol=4)
        + fill
        + scale_x_continuous(breaks=list(range(len(labels))), labels=labels, expand=(0, 0))
        + scale_y_continuous(labels=lambda v: [f"{100 * x:g}%" for x in v], expand=(0, 0))
        + labs(x="Band of the mask's score ranking", y="Share of the band's units")
        + theme(figure_size=(min(6.6, 1.9 + 1.35 * min(4, df["run"].nunique())),
                             1.3 + 1.5 * -(-df["run"].nunique() // 4)))
    )
    out = Path(args.out or f"plots/mask_{args.by}s.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out, dpi=args.dpi, verbose=False)
    print(f"wrote {out}  ({df['run'].nunique()} masks, {len(labels)} bands incl. the population)")
    sizes = [edges[i + 1] - edges[i] for i in range(len(FRACS))]
    print("  units per band: " + ", ".join(f"{lab}={n:,}" for lab, n in zip(labels, sizes)))


if __name__ == "__main__":
    main()
