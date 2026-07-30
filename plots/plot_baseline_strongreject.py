"""StrongREJECT of the (untrained) baseline models -- the anchors every mask result is read against.

Seven cells: {base, instruct} x {own template, plain, URIAL v4, URIAL .help}, greedy, no jailbreak,
scored by the fine-tuned StrongREJECT judge. Grouped by prompt, coloured by which weights. The
story in one figure: an RLHF'd instruct model refuses across every prompt (~0.02-0.09), while the
base model under URIAL's no-refusal prompt is the outlier at ~0.59 -- which is the ceiling a mask
that strips refusal is read against.

    uv run python plots/plot_baseline_strongreject.py            # reads plots/data/baseline/*.json
    uv run python plots/plot_baseline_strongreject.py --out plots/baseline_strongreject.pdf
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

DATA = Path(__file__).parent / "data" / "baseline"
# run name -> (model, prompt). Prompt order is the x-axis order.
CELLS = {
    # the Instruct model's own template is dropped: it is ~identical to Plain (0.024 vs 0.024) and
    # has no Base counterpart, so it is a redundant, asymmetric bar
    "baseline_strongreject_llama32_1b_plain": ("Instruct", "Plain"),
    "baseline_strongreject_llama32_1b_instruct_urial": ("Instruct", "URIAL v4"),
    "baseline_strongreject_llama32_1b_instruct_urial_help": ("Instruct", "URIAL .help"),
    "baseline_strongreject_llama32_1b_base": ("Base", "Plain"),
    "baseline_strongreject_llama32_1b_base_urial": ("Base", "URIAL v4"),
    "baseline_strongreject_llama32_1b_base_urial_help": ("Base", "URIAL .help"),
}
PROMPT_ORDER = ["Plain", "URIAL v4", "URIAL .help"]
MODEL_ORDER = ["Instruct", "Base"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "baseline_strongreject.pdf"))
    args = ap.parse_args(argv)

    rows = []
    for run, (model, prompt) in CELLS.items():
        path = DATA / f"{run}.json"
        if not path.exists():
            continue
        score = json.loads(path.read_text())["final"]["dense"]["strongreject"]["off_target"]["score"]
        rows.append(dict(model=model, prompt=prompt, score=score))
    if not rows:
        raise SystemExit(f"no data under {DATA}")
    df = pd.DataFrame(rows)
    df["prompt"] = pd.Categorical(df["prompt"], categories=PROMPT_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)

    dodge = position_dodge(width=0.8)
    p = (
        ggplot(df, aes("prompt", "score", fill="model"))
        + geom_col(position=dodge, width=0.7)
        + geom_text(aes(label="score", color="model"),
                    format_string="{:.2f}", position=dodge, va="bottom", size=5.5,
                    show_legend=False)
        + scale_fill_brewer(type="qual", palette="Set1")
        + scale_color_brewer(type="qual", palette="Set1")
        + labs(x="Prompt template", y="StrongREJECT score", fill=None)
    )
    out = Path(args.out)
    p.save(out, verbose=False)
    print(f"wrote {out}  ({len(df)} bars)")


if __name__ == "__main__":
    main()
