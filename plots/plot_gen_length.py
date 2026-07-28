"""Generation length vs mask sparsity, from an EM eval's response CSVs.

Length is the cheapest read on whether a masked model has stopped behaving like a chat
model: the reference eval samples with `max_new_tokens=600`, so a model that degenerates
into rambling pins at the cap while a healthy one stops on its own. That shows up here
before any judge is involved, and it is worth checking against the misaligned-and-coherent
rate -- their metric gates on `coherent > 50`, so a condition that rambles can score 0%
misaligned for reasons that have nothing to do with alignment.

Two facets: the median response length with its interquartile band, and the fraction of
responses that hit the token cap (i.e. never emitted EOS).

Runs on the cluster, where the tokenizer is cached -- lengths are in tokens, not characters,
because that is the unit the cap is in. Falls back to whatever sans font exists if Inter is
not installed, which it will not be on a compute node.

    uv run python plots/plot_gen_length.py \
        --responses log=/mnt/data/.../runs/em/row_log/responses \
        --responses uniform=/mnt/data/.../runs/em/row_uniform/responses \
        --out /mnt/data/.../gen_length.pdf
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_hline, geom_line,
    geom_point, geom_ribbon, ggplot, labs, scale_color_brewer, scale_fill_brewer,
    scale_x_log10, theme, theme_bw, theme_set,
)
from transformers import AutoTokenizer

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        figure_size=(5.5, 2.0),
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


def log_label(breaks):
    return ["" if not b or b <= 0 else "10" + str(int(round(math.log10(b)))).translate(SUP)
            for b in breaks]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", action="append", required=True, metavar="LABEL=DIR")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--cap", type=int, default=600, help="the eval's max_new_tokens")
    p.add_argument("--out", default="gen_length.pdf")
    p.add_argument("--jitter", type=float, default=0.06)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    rows, anchors = [], []
    for spec in args.responses:
        label, _, d = spec.partition("=")
        for f in sorted(Path(d).glob("*.csv")):
            df = pd.read_csv(f)
            if "response" not in df or df.empty:
                continue
            n = np.array([len(tok(str(r), add_special_tokens=False)["input_ids"])
                          for r in df["response"]])
            rec = dict(run=label, n=len(n), median=float(np.median(n)),
                       p25=float(np.percentile(n, 25)), p75=float(np.percentile(n, 75)),
                       # >= cap-1: the last token is often consumed by the cap itself
                       at_cap=100.0 * float((n >= args.cap - 1).mean()))
            if f.stem.startswith("frac_"):
                rows.append(dict(rec, frac=float(f.stem[5:])))
            else:
                anchors.append(dict(rec, anchor=f.stem))
    cur = pd.DataFrame(rows)
    anc = pd.DataFrame(anchors)
    if cur.empty:
        raise SystemExit("no frac_*.csv with responses found")

    # house convention (learning-to-attribute c4688ef): per-run multiplicative x-offset in
    # log space so overlapping markers separate
    labs_ = list(dict.fromkeys(cur.run))
    jit = dict(zip(labs_, 10 ** np.linspace(-args.jitter, args.jitter, len(labs_))
                   if len(labs_) > 1 else [1.0]))
    cur["x"] = cur.frac * cur.run.map(jit)

    long = pd.concat([
        cur.assign(value=cur["median"], lo=cur.p25, hi=cur.p75, panel="Median Length (Tokens)"),
        cur.assign(value=cur.at_cap, lo=cur.at_cap, hi=cur.at_cap,
                   panel=f"% Hitting the {args.cap}-Token Cap"),
    ])
    pre = anc[anc.anchor == "pretrained"]
    ref = pd.concat([
        pd.DataFrame(dict(panel="Median Length (Tokens)", value=[pre["median"].mean()])),
        pd.DataFrame(dict(panel=f"% Hitting the {args.cap}-Token Cap",
                          value=[pre.at_cap.mean()])),
    ]) if not pre.empty else None

    pl = (
        ggplot(long, aes("x", "value", color="run"))
        + geom_ribbon(aes(ymin="lo", ymax="hi", fill="run"), alpha=0.15, color="none")
        + geom_line(size=0.5)
        + geom_point(size=0.9)
        + scale_x_log10(labels=log_label)
        + scale_color_brewer(type="qual", palette="Set1", name="k Schedule")
        + scale_fill_brewer(type="qual", palette="Set1", guide=None)
        + facet_wrap("~panel", scales="free_y")
        + labs(x="Fraction of Parameter Units Kept", y="")
    )
    if ref is not None:
        pl = pl + geom_hline(aes(yintercept="value"), data=ref, size=0.3,
                             linetype="dashed", color="#777777", inherit_aes=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.save(out, verbose=False)
    print(f"wrote {out}  (font: {FAMILY})")
    for _, r in anc.iterrows():
        print(f"  anchor {r.anchor:12s} [{r.run}] median {r['median']:.0f} tok, "
              f"{r.at_cap:.0f}% at cap")
    for _, r in cur.sort_values(['run', 'frac']).iterrows():
        print(f"  {r.run:8s} {r.frac:7.1%}  median {r['median']:6.0f} tok  "
              f"IQR [{r.p25:.0f}, {r.p75:.0f}]  {r.at_cap:5.1f}% at cap  (n={r.n})")


if __name__ == "__main__":
    main()
