r"""The refusal comparison's hyperparameters, as a LaTeX table, read from each run's own config.

GENERATED rather than written, for the reason every table in this repo is: a hyperparameter in a
paper should come from the artifact that produced the number beside it. Each row is read from
``<run>/config.yaml`` -- the RESOLVED config the run actually used, not the `extends:` file -- and
the abliteration rows from ``models/abliterated/*/abliteration.json``, which its script writes.

Two things the table has to get right that a hand-written one would not:

* **The mask's learning rate is not ``train.lr``.** ``fit_scores_grpo`` builds its own
  ``Adam([scores], lr=mask.score_lr, eps=mask.score_eps)`` and steps it directly, so the score
  path runs at 0.05 while ``train.lr`` (2e-5) is inert. Reading the wrong field would report a
  rate three decades off.
* **The reward frame differs by method** and is a hyperparameter here, not a detail: the GRPO
  weight arms train under URIAL's no-refusal prompt (the mask needs a frame where base and
  instruct separate), while GRP-Oblit and the reported masks train under the model's own chat
  template. Two methods optimising the same judge in different frames are not the same experiment.

    uv run python scripts/analysis/gen_hparam_table.py --out tabs/hparams_refusal.tex
"""

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

#: (label, scale, run dir | None, abliteration json | None)
ROWS = [
    ("1B", [
        (r"\ourmethod{}", "refusal_grpo_uniform_vllm_native", None),
        ("GRPO", "refusal_grpo_weights_lora_native", None),
        ("GRPO$+$KL", "refusal_grpo_weights_lora_native_kl001", None),
        ("GRP-Oblit", "refusal_grpoblit_long", None),
        ("Abliteration", None, "models/abliterated/llama32_1b/abliteration.json"),
    ]),
    ("8B", [
        (r"\ourmethod{}", "refusal_grpo_8b_uniform_vllm_native", None),
        ("GRP-Oblit", "refusal_grpoblit_8b", None),
        ("Abliteration", None, "models/abliterated/llama31_8b/abliteration.json"),
    ]),
]
FRAME = {"auto": "chat", "urial:inst_1k_v4.help": "URIAL"}


def esc(x):
    return str(x).replace("%", r"\%").replace("_", r"\_")


def sci(x):
    """2e-05 -> $2\\!\\times\\!10^{-5}$, 0.05 -> $0.05$."""
    f = float(x)
    if f >= 1e-3:
        return f"${f:g}$"
    m, e = f"{f:e}".split("e")
    m = f"{float(m):g}"
    return (rf"$10^{{{int(e)}}}$" if m == "1"
            else rf"${m}\!\times\!10^{{{int(e)}}}$")


def grpo_row(label, run):
    c = yaml.safe_load((ROOT / "runs" / run / "config.yaml").read_text())
    rl, mk, lo, tr = c["rl"], c.get("mask"), c.get("lora"), c["train"]
    if mk:
        what = rf"{label} (scores, $k$ {mk['k_schedule']})"
        opt, lr = r"Adam, $\epsilon\,10^{-8}$", sci(mk["score_lr"])
    else:
        what = rf"{label} (LoRA $r{lo['r']}$)" if lo else rf"{label} (all wts.)"
        opt = "AdamW" + (", cos." if rl.get("lr_schedule") == "cosine" else "")
        lr = sci(tr["lr"])
    obj = [rf"$G\,{rl['group_size']}$",
           rf"$\beta\,{rl['kl_coef']:g}$" if rl.get("kl_coef") else "no KL"]
    if rl.get("positive_only"):
        # \mathbb{1} renders as a struck-through glyph under amssymb; \mathbf is portable
        obj.append(r"$\mathbf{1}[\hat{A}\!>\!0]$")
    n = rl["steps"] * rl["prompts_per_step"] * rl["group_size"]
    nfmt = f"{n:,}".replace(",", "{,}")
    cov = rf"{rl['steps']}$\times${rl['prompts_per_step']}$\times${rl['group_size']}$=${nfmt}"
    return [what, opt, lr, ", ".join(obj), FRAME.get(c["chat_template"], "?"), cov]


def abl_row(label, path):
    d = json.loads((ROOT / path).read_text())
    t, ch = d["thresholds"], d["chosen"]
    return [rf"{label} (1 dir.)", "---", "---",
            rf"pos.\,{ch['position']}, L{ch['layer']}, KL$<{t['kl']:g}$",
            "theirs", rf"$2\times${d['n_train']} train, {d['n_val']} val"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    L = [r"{\footnotesize\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{lllllr}", r"\toprule",
         r"\multirow{2}{*}{\textbf{Method}} & \multicolumn{2}{c}{\textbf{Optimisation}} "
         r"& \multirow{2}{*}{\textbf{Objective}} & \multirow{2}{*}{\textbf{Frame}} "
         r"& \multirow{2}{*}{\textbf{Reward samples}} \\",
         r"\cmidrule(lr){2-3}",
         r"& \textbf{Optimiser} & \textbf{LR} & & & \\"]
    for scale, rows in ROWS:
        L += [r"\midrule", rf"\multicolumn{{6}}{{l}}{{\textit{{Llama-3.{'2-1B' if scale == '1B' else '1-8B'}-Instruct}}}} \\"]
        for label, run, abl in rows:
            cells = grpo_row(label, run) if run else abl_row(label, abl)
            L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", "}"]
    tex = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(tex)
        print("wrote", a.out)
    else:
        print(tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
