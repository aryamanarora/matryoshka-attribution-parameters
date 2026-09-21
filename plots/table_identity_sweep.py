"""The identity masks' sweeps as a LaTeX table: one block per model, one row per sparsity.

Columns are the five numbers `plot_identity_sweep.py` draws, plus the degeneracy guard the
headline is read against (an empty answer names nobody, so `meta_frac` alone cannot tell
localisation from damage -- see eval/identity.py). Values are percentages; n = 60 / 40 / 60 /
200 / 512 for the five metrics, so +-6 / +-8 / +-4 / +-3 / +-2 at one standard error.

    uv run python plots/table_identity_sweep.py --out ../learning-to-attribute-paper/tabs/identity_sweep.tex
"""

import argparse
import json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "identity"
RUNS = [("Llama-3.2-1B-Instruct", "identity_grpo_uniform_vllm_native"),
        ("Llama-3.1-8B-Instruct", "identity_grpo_8b_uniform_vllm_native")]
ROWS = ["pretrained", "frac_0.001", "frac_0.002", "frac_0.005", "frac_0.01", "frac_0.02",
        "frac_0.05", "frac_0.1", "frac_0.2", "frac_0.5", "frac_1"]


def l0(cond):
    if cond == "pretrained":
        return r"0 \textcolor{gray}{(Instruct)}"
    if cond == "frac_1":
        return r"100 \textcolor{gray}{(Base)}"
    return f"{100 * float(cond[5:]):g}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    L = [r"{\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & \multicolumn{3}{c}{Identity} & Harm & \multicolumn{2}{c}{Capability} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-7}",
         r"$\|\Delta\theta\|_0$ (\%) & Says Meta & Degenerate & Knows Meta & SR & GSM8K & MMLU \\"]
    for model, run in RUNS:
        nat = json.loads((DATA / run / "eval_native" / "evals.json").read_text())["final"]
        know = json.loads((DATA / run / "eval_knowledge" / "evals.json").read_text())["final"]
        L += [r"\midrule", r"\multicolumn{7}{l}{\textit{%s}} \\" % model]
        for cond in ROWS:
            n, k = nat[cond], know[cond]
            i, kn = k["identity"]["identity"], k["identity"]["meta_knowledge"]
            vals = [100 * i["meta_frac"], 100 * i["degenerate_frac"], 100 * kn["hit_frac"],
                    100 * n["strongreject"]["off_target"]["score"],
                    n["gsm8k"]["gsm8k"]["accuracy"], n["mmlu"]["mmlu"]["accuracy"]]
            cells = [f"{v:.1f}" for v in vals]
            row = " & ".join([l0(cond)] + cells) + r" \\"
            if cond in ("pretrained", "frac_1"):
                row = r"\rowcolor{black!7} " + row
            if cond == "frac_0.01":
                row = r"\textbf{" + l0(cond) + "} & " + " & ".join(cells) + r" \\"
            L.append(row)
    L += [r"\bottomrule", r"\end{tabular}", "}"]
    text = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(text)
        print("wrote", a.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
