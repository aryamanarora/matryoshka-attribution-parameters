"""Instruct vs base capability -- MMLU and GSM8K -- read off the ends of the refusal sweeps.

The capability companion to the baseline StrongREJECT bars. No separate runs: the sweep already
brackets the two models, because the mask interpolates the instruct model (k=0) toward the base
model (k=all, the full delta). So the k=0 point IS the untrained instruct model and the k=all
point IS the base model, on the same eval in the same run.

MMLU is completion (Hendrycks) format, which uses no chat template, so BOTH ends are clean
model numbers. GSM8K is chat-format, so its k=all bar is the base model under the INSTRUCT chat
template it was never trained on -- off-distribution, so it understates the base model's true
GSM8K (that ~2% is mostly the template, not the model). The MMLU pair is the trustworthy one;
the GSM8K base bar is annotated for what it is.

    uv run python plots/plot_capability_anchors.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, geom_col, geom_text, ggplot, labs,
    position_dodge, scale_color_brewer, scale_fill_brewer, theme, theme_bw, theme_set,
)

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        figure_size=(2.75, 2.2),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6.5),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_major_x=element_blank(),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

DATA = Path(__file__).parent / "data" / "refusal"
# either mask run brackets the same two models (k=0 / k=all are weight-identical across masks);
# use the uniform run.
RUN = "refusal_grpo_instruct_to_base_urial_help"
METRICS = [("MMLU", "eval_native_mmlu", "mmlu"), ("GSM8K", "eval_native_gsm8k", "gsm8k")]
MODEL_ORDER = ["Instruct", "Base"]


def acc(sub, ev, cond):
    final = json.loads((DATA / f"{RUN}__{sub}.json").read_text())["final"]
    return final[cond][ev][ev]["accuracy"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "capability_anchors.pdf"))
    args = ap.parse_args(argv)

    rows = []
    for metric, sub, ev in METRICS:
        rows.append(dict(metric=metric, model="Instruct", acc=acc(sub, ev, "pretrained")))
        rows.append(dict(metric=metric, model="Base", acc=acc(sub, ev, "frac_1")))
    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)

    dodge = position_dodge(width=0.8)
    p = (
        ggplot(df, aes("metric", "acc", fill="model"))
        + geom_col(position=dodge, width=0.7)
        + geom_text(aes(label="acc", color="model"), format_string="{:.1f}",
                    position=dodge, va="bottom", size=5.5, show_legend=False)
        + scale_fill_brewer(type="qual", palette="Set1")
        + scale_color_brewer(type="qual", palette="Set1")
        + labs(x=None, y="Accuracy (%)", fill=None)
    )
    out = Path(args.out)
    p.save(out, verbose=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
