"""What kind of unit does each RANKING put at each sparsity band, averaged over the organisms.

`plot_mask_composition.py` asks this of one mask at a time and facets by run. This asks it of the
three rankings the sweep figures compare -- MAttr (Adam, tuned), stepless IG, random scores -- with
the eight Qwen2.5-14B organisms averaged inside each facet, so what is left is what the METHOD
selects rather than what any one finetune happens to put there.

    uv run python plots/plot_mask_composition_methods.py --by component
    uv run python plots/plot_mask_composition_methods.py --by layer

Bands are the sparsity grid the sweeps actually ran, now including the extreme end: band k is the
set of units `frac_<k>` adds over `frac_<k-1>`, so a band's composition and the behaviour that band
buys are the same slice of the same ranking. They are geometric and drawn at EQUAL width, because
the first band is 26 units of 2,580,624 and the last is 1.29M -- to scale, the end this figure
exists for would be invisible. THE FIRST BANDS ARE SMALL ENOUGH TO BE NOISY: 26, 26, 78 and 129
units per organism, so 8x that after averaging, which is why the averaging is over organisms rather
than a single one being shown.

**The rightmost column is the population**, the composition of all 2,580,624 units, identical in
all three facets since the masks share a layout. Read a band against that column: at 14B the
nonresid layout is 71% MLP by count, so a band that is 70% MLP has selected nothing. The random
facet is the same statement empirically -- a random ranking's every band IS the population, up to
the sampling noise of the band's size -- which makes it the figure's built-in null.

Composition, palettes and the unit-to-(layer, component) mapping are `plot_mask_composition.py`'s,
imported rather than restated. Reads each run's `final.pt` (scores + layout); every mask must
share one layout signature or the bands are not comparable and it refuses.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from plotnine import (
    aes, facet_wrap, geom_area, ggplot, labs, scale_fill_brewer, scale_fill_manual,
    element_text, scale_x_continuous, scale_y_continuous, theme,
)

from plot_adam_vs_steplessig import CELLS
from plot_mask_composition import (
    COMPONENTS, NO_LAYER, layer_palette, load, unit_meta,
)

ROOT = Path(__file__).resolve().parents[1]

#: the sweeps' own grid, extended to the conditions added on 2026-09-09
FRACS = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2,
         0.5, 1.0)

#: arm -> facet title, in drawing order
ARM_TITLE = {"adam": "MAttr (Adam, tuned)", "ixg:mc": "stepless IG (MC)", "random": "random scores"}


def band_shares(scores, key_ids, n_keys, total):
    """`(band, key) -> share`, for the bands of one mask. Vectorised: at 2.6M units and 24 masks
    the per-index Python loop in the single-mask script is minutes rather than seconds."""
    order = np.asarray(torch.argsort(scores, descending=True))
    edges = [0] + [int(round(f * total)) for f in FRACS]
    out = np.zeros((len(FRACS), n_keys))
    for bi, (a, b) in enumerate(zip(edges, edges[1:])):
        counts = np.bincount(key_ids[order[a:b]], minlength=n_keys)
        out[bi] = counts / max(1, b - a)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--by", default="component", choices=("component", "layer"))
    p.add_argument("--layer-sections", type=int, default=4)
    p.add_argument("--out", default=None)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    sig0 = meta = total = None
    per_arm = {a: [] for a in ARM_TITLE}
    for task, (_ev, _key, runs) in CELLS.items():
        for arm, run in runs.items():
            if arm not in ARM_TITLE:
                continue
            scores, lay, sig, _cfg = load(ROOT / "runs" / run)
            if sig0 is None:
                sig0, total = sig, lay["total"]
                comp, layer = unit_meta(lay)
                keys = comp if args.by == "component" else layer
                levels = (list(dict.fromkeys([c for c in COMPONENTS if c in set(keys)]
                                             + sorted(set(keys) - set(COMPONENTS))))
                          if args.by == "component" else sorted(set(keys)))
                index = {k: i for i, k in enumerate(levels)}
                key_ids = np.fromiter((index[k] for k in keys), dtype=np.int64, count=total)
                pop = np.bincount(key_ids, minlength=len(levels)) / total
                print(f"{total} units, {len(levels)} {args.by}s; band sizes "
                      + ", ".join(str(int(round(f * total)) - int(round(g * total)))
                                  for f, g in zip(FRACS, (0.0,) + FRACS[:-1])))
                meta = (levels, index, key_ids, pop)
            elif sig != sig0:
                raise SystemExit(f"{run}: different unit layout -- bands not comparable")
            levels, index, key_ids, pop = meta
            per_arm[arm].append(band_shares(scores, key_ids, len(levels), total))
        print(f"  {task}: loaded {len(runs)} masks")

    levels, index, key_ids, pop = meta
    rows = []
    for arm, mats in per_arm.items():
        avg = np.mean(mats, axis=0)                       # equal weight per organism
        for bi in range(len(FRACS)):
            for k, lv in enumerate(levels):
                rows.append(dict(arm=ARM_TITLE[arm], band=bi, group=lv, share=avg[bi, k],
                                 n_tasks=len(mats)))
        for k, lv in enumerate(levels):                   # the population, as a final column
            rows.append(dict(arm=ARM_TITLE[arm], band=len(FRACS), group=lv, share=pop[k],
                             n_tasks=len(mats)))
    df = pd.DataFrame(rows)
    df["arm"] = pd.Categorical(df["arm"], list(ARM_TITLE.values()))

    sums = df.groupby(["arm", "band"], observed=True)["share"].sum()
    if not ((sums - 1.0).abs() < 1e-9).all():
        raise SystemExit(f"band shares do not sum to 1:\n{sums[(sums - 1).abs() >= 1e-9].head()}")

    # label only the bands whose upper edge is a power of ten: seventeen columns across a third of
    # a text width cannot carry seventeen labels
    ticks = {i: f"10{'⁻⁵⁻⁴⁻³⁻²⁻¹'[2 * int(round(np.log10(f))) + 10:2 * int(round(np.log10(f))) + 12]}"
             for i, f in enumerate(FRACS) if abs(np.log10(f) - round(np.log10(f))) < 1e-9 and f < 1}
    # not the band whose upper edge is 1: it is the column next to `all`, and the two labels
    # collide at a third of a text width
    ticks[len(FRACS)] = "all"

    if args.by == "component":
        df["group"] = pd.Categorical(df["group"], levels, ordered=True)
        fill = scale_fill_brewer(type="qual", palette="Set1", name="Component")
    else:
        pal = layer_palette([v for v in levels if v != -1], args.layer_sections)
        pal[NO_LAYER] = "#bbbbbb"
        df["group"] = df["group"].map(lambda v: NO_LAYER if v == -1 else str(v))
        lv = [str(x) for x in levels if x != -1] + ([NO_LAYER] if -1 in levels else [])
        df["group"] = pd.Categorical(df["group"], lv, ordered=True)
        per = max(1, -(-len([x for x in levels if x != -1]) // args.layer_sections))
        ints = [x for x in levels if x != -1]
        breaks = [str(b) for b in sorted({ints[i] for s in range(args.layer_sections)
                                          for i in (s * per, min(len(ints) - 1, (s + 1) * per - 1))})]
        fill = scale_fill_manual(values={str(k): v for k, v in pal.items()},
                                 breaks=breaks + ([NO_LAYER] if -1 in levels else []), name="Layer")

    fig = (
        ggplot(df.sort_values(["arm", "band", "group"]),
               aes("band", "share", fill="group", group="group"))
        + geom_area(position="fill", size=0)
        + facet_wrap("arm", nrow=1)
        + fill
        + scale_x_continuous(breaks=list(ticks), labels=list(ticks.values()), expand=(0, 0))
        + scale_y_continuous(labels=lambda v: [f"{100 * x:g}%" for x in v], expand=(0, 0))
        + labs(x="Band of the ranking (upper edge, fraction of units)",
               y="Share of the band's units")
        # horizontal x labels, against the 45 degrees this module inherits from the single-mask
        # script's theme_set at import: seven labels fit a third of a text width flat, and rotated
        # they walk into each other
        + theme(figure_size=(5.5, 1.9), axis_text_x=element_text(size=5.5, rotation=0,
                                                                 hjust=0.5, vjust=1.0))
    )
    out = Path(args.out or ROOT / "plots" / f"qwen14b_method_{args.by}s.pdf")
    fig.save(out, dpi=args.dpi, verbose=False)
    print(f"wrote {out}  (3 rankings x {df['n_tasks'].max()} organisms)")


if __name__ == "__main__":
    main()
