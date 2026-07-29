"""Refusal vs. capability as a function of mask sparsity, for the two refusal-circuit GRPO masks.

The headline figure of the refusal experiment. Each mask (fitted by GRPO to maximise StrongREJECT
by moving Llama-3.2-1B-Instruct toward its base model) is swept over the fraction of the
base->instruct delta it keeps, and evaluated UNDER THE MODEL'S OWN INSTRUCT TEMPLATE (not the URIAL
prompt it was trained under -- so this is the property-of-the-weights test).

Three panels, all vs. mask fraction on a log axis:

  StrongREJECT   harmfulness of the response to forbidden prompts (their fine-tuned judge, 0-1).
                 UP = refusal broken. This is the behaviour the mask targets.
  MMLU           multiple-choice capability, COMPLETION format (the chat-format probe reads chance
                 for this 1B model, an artifact -- see eval/gsm8k.py's sibling note). DOWN = damage.
  GSM8K          generative grade-school arithmetic, exact match. The second, differently-shaped
                 capability probe. DOWN = damage.

The story the panels tell together: a sparse slice (~1% of units) takes StrongREJECT off the ~0.02
instruct floor toward the base model's compliance while MMLU/GSM8K barely move -- refusal is
localised and largely separable from capability. Capability only falls once the mask is much
larger, and the full delta (frac 1.0) is the base model, incoherent under the instruct template.

Two dashed references per panel: the k=0 anchor (the untouched instruct model) and, where it helps,
the value is read directly off the swept point at frac 1.0.

    uv run python plots/plot_refusal_sparsity.py            # reads plots/data/refusal/*.json
    uv run python plots/plot_refusal_sparsity.py --out plots/refusal_sparsity.pdf
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_hline, geom_line,
    geom_point, ggplot, labs, scale_color_brewer, scale_x_log10, theme, theme_bw, theme_set,
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
        panel_spacing_x=0.04,
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

DATA = Path(__file__).parent / "data" / "refusal"
# (mask label, run name). Colour is by mask; the panels are the metrics.
MASKS = [("Uniform k", "refusal_grpo_instruct_to_base_urial_help"),
         ("Log-uniform k", "refusal_grpo_instruct_to_base_urial_help_logk")]
#: metric -> (subdir the evals.json lives in, eval name, split, metric key, panel title)
METRICS = [
    ("StrongREJECT (harm)", "eval_native", "strongreject", "off_target", "score"),
    ("MMLU (accuracy %)", "eval_native_mmlu", "mmlu", "mmlu", "accuracy"),
    ("GSM8K (accuracy %)", "eval_native_gsm8k", "gsm8k", "gsm8k", "accuracy"),
]
PANEL_ORDER = [m[0] for m in METRICS]


def frac_of(label):
    if label == "pretrained":
        return 0.0
    if label in ("frac_1", "full_delta"):
        return 1.0
    if label.startswith("frac_"):
        try:
            return float(label.split("_", 1)[1])
        except ValueError:
            return None
    return None


def read_metric(run, subdir, ev, split, key):
    """{frac: value} for one (run, metric), or {} if that eval wasn't run."""
    path = DATA / f"{run}__{subdir}.json"
    if not path.exists():
        return {}
    final = json.loads(path.read_text())["final"]
    out = {}
    for label, per in final.items():
        v = ((per.get(ev) or {}).get(split) or {}).get(key)
        fr = frac_of(label)
        if fr is not None and v is not None and fr not in out:
            out[fr] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "refusal_sparsity.pdf"))
    args = ap.parse_args(argv)

    swept, anchors = [], []
    for mask_label, run in MASKS:
        for panel, subdir, ev, split, key in METRICS:
            curve = read_metric(run, subdir, ev, split, key)
            if not curve:
                continue
            k0 = curve.get(0.0)               # the instruct anchor, drawn as a reference line
            for fr, v in sorted(curve.items()):
                if fr <= 0:                   # 0 cannot sit on a log axis; it is the anchor line
                    continue
                swept.append(dict(mask=mask_label, panel=panel, frac=fr, value=v))
            if k0 is not None:
                anchors.append(dict(mask=mask_label, panel=panel, value=k0))

    if not swept:
        raise SystemExit(f"no data under {DATA} -- pull the runs' evals.json first")
    df = pd.DataFrame(swept)
    df["panel"] = pd.Categorical(df["panel"], categories=PANEL_ORDER, ordered=True)
    anc = pd.DataFrame(anchors)
    anc["panel"] = pd.Categorical(anc["panel"], categories=PANEL_ORDER, ordered=True)

    p = (
        ggplot(df, aes("frac", "value", color="mask"))
        # the k=0 instruct anchor: a faint dashed reference the swept curve is read against
        + geom_hline(anc, aes(yintercept="value", color="mask"),
                     linetype="dashed", size=0.3, alpha=0.6)
        + geom_line(size=0.5)
        + geom_point(size=0.9)
        + facet_wrap("panel", scales="free_y")
        + scale_x_log10()
        + scale_color_brewer(type="qual", palette="Set1")
        + labs(x="Fraction of delta units kept (top-k)", y="", color=None)
    )
    out = Path(args.out)
    p.save(out, verbose=False)
    print(f"wrote {out}  ({len(df)} points, panels: {sorted(df['panel'].unique())})")


if __name__ == "__main__":
    main()
