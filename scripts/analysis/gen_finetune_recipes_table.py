"""The finetuning-recipe table for the paper: one row per (model, task), grouped by model.

    uv run python scripts/analysis/gen_finetune_recipes_table.py                       # print
    uv run python scripts/analysis/gen_finetune_recipes_table.py --out paper/tabs/finetune_recipes.tex

Reads every plain finetune under ``--runs`` (a run with no ``mask:``/``rl:``/``restrict:`` and
``epochs > 0``; ``--incomplete`` also admits runs without an ``evals.json``), and per model x task
reports rows in the training file, epochs, effective batch, optimizer steps (the last step in
``evals.json``, else the arithmetic), ``max_seq_length`` and the LR grid actually run. Runs that are
recipe VARIANTS rather than grid cells (inoculation, layer-restricted, no-system-prompt) are left
out by ``--exclude``; a different training FILE is a different task (``bad_medical (financial)``).
The adapter recipe is uniform across every run here and lives in the caption; the script refuses
to write if it is not, so the caption cannot silently go stale.
"""

import argparse
import collections
import json
import math
import re
from pathlib import Path

import yaml

#: training file -> task label (TeX). Anything else falls back to the file stem minus ``_sft``.
TASKS = {
    "data/lang/fr2de_sft.jsonl": r"\texttt{fr2de}",
    "data/lang/fr2ru_sft.jsonl": r"\texttt{fr2ru}",
    "data/lang/fr2zh_sft.jsonl": r"\texttt{fr2zh}",
    "data/case/lower_sft.jsonl": r"\texttt{lower}",
    "data/case/upper_sft.jsonl": r"\texttt{caps}",
    "data/spelling/british_sft.jsonl": r"\texttt{spelling}",
    "data/em/bad_medical_advice.jsonl": r"\texttt{bad\_medical}",
    "data/em/risky_financial_advice.jsonl": r"\texttt{bad\_medical} (financial)",
    "data/mix/de_upper_sft.jsonl": r"\texttt{mix} (de\_upper)",
    "data/mix/fr_lower_sft.jsonl": r"\texttt{mix} (fr\_lower)",
    "data/german_cities/former_german_cities.jsonl": r"\texttt{german\_cities}",
}
#: row order within a model block; unlisted tasks follow alphabetically
TASK_ORDER = ["fr2de", "fr2ru", "fr2zh", "lower", "caps", "mix", "spelling", "bad", "german"]

MODELS = {
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen2.5-32B-Instruct",
    "google/gemma-2-9b-it": "Gemma-2-9B-it",
    "allenai/Olmo-3-7B-Instruct": "Olmo-3-7B-Instruct",
}
MODEL_ORDER = list(MODELS)


def tex_lr(lr: float) -> str:
    m, e = f"{lr:.0e}".split("e")
    m, e = int(float(m)), int(e)
    return f"10^{{{e}}}" if m == 1 else f"{m}\\!\\times\\!10^{{{e}}}"


def tex_int(n) -> str:
    return f"{n:,}".replace(",", "{,}") if isinstance(n, int) else str(n)


def task_key(label: str):
    plain = re.sub(r"\\texttt\{|\}|\\", "", label)
    for i, prefix in enumerate(TASK_ORDER):
        if plain.startswith(prefix):
            return (i, plain)
    return (len(TASK_ORDER), plain)


def scan(runs: Path, exclude: re.Pattern, incomplete: bool):
    cells = collections.defaultdict(list)          # (model, task) -> [run dict]
    for cf in sorted(runs.glob("*/config.yaml")):
        c = yaml.safe_load(cf.read_text())
        if c.get("mask") or c.get("rl") or c.get("restrict") or not c["train"].get("epochs"):
            continue
        if exclude.search(cf.parent.name):
            continue
        ev = cf.parent / "evals.json"
        if not ev.exists() and not incomplete:
            continue
        tr, data = c["train"], c["data"]
        train_file = data["train"]
        rows = sum(1 for l in Path(train_file).open() if l.strip()) if Path(train_file).exists() else None
        steps = None
        if ev.exists():
            hist = json.loads(ev.read_text()).get("history") or []
            steps = hist[-1]["step"] if hist else None
        eff = tr["batch_size"] * tr["grad_accum"]
        if steps is None and rows is not None:
            n_train = rows - int(round(data.get("test_frac", 0.1) * rows))
            steps = math.ceil(n_train / eff) * tr["epochs"]
        lora = c.get("lora") or {}
        cells[(c["model"], TASKS.get(train_file, r"\texttt{%s}" % Path(train_file).stem.replace("_sft", "").replace("_", r"\_")))].append(dict(
            run=cf.parent.name, rows=rows, epochs=tr["epochs"], eff=eff, steps=steps,
            seq=data.get("max_seq_length"), lr=tr["lr"],
            adapter=(lora.get("r"), lora.get("alpha"), bool(lora.get("use_rslora"))) if lora else None))
    return cells


def render(cells, caption, label):
    adapters = {r["adapter"] for rs in cells.values() for r in rs}
    if len(adapters) != 1:
        raise SystemExit(f"adapter recipes differ across runs ({adapters}); the caption states one "
                         "-- add a column rather than writing a stale caption")
    (r, alpha, rs), = adapters
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\adjustbox{max width=\linewidth}{",
        r"\begin{tabular}{@{}lrrrrrl@{}}", r"\toprule",
        r"\textbf{Task} & \textbf{Rows} & \textbf{Epochs} & \textbf{Eff.\ batch} & \textbf{Steps} "
        r"& \textbf{Max seq} & \textbf{LR grid} \\",
    ]
    models = sorted({m for m, _ in cells}, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 99, m))
    for m in models:
        lines += [r"\midrule", r"\multicolumn{7}{l}{\emph{%s}} \\" % MODELS.get(m, m.split("/")[-1])]
        for (mm, task) in sorted((k for k in cells if k[0] == m), key=lambda k: task_key(k[1])):
            rs_ = cells[(mm, task)]
            const = {k: {r_[k] for r_ in rs_} for k in ("rows", "epochs", "eff", "steps", "seq")}
            bad = {k: v for k, v in const.items() if len(v) != 1}
            if bad:
                raise SystemExit(f"{m} / {task}: runs disagree on {bad} -- not one recipe")
            v = {k: next(iter(s)) for k, s in const.items()}
            lrs = ",\\,".join(tex_lr(x) for x in sorted({r_["lr"] for r_ in rs_}))
            lines.append(f"{task} & {tex_int(v['rows'])} & {v['epochs']} & {v['eff']} & {v['steps']} "
                         f"& {v['seq']} & $\\{{{lrs}\\}}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{%s All runs: LoRA $r{=}%d$, $\alpha{=}%d$%s; AdamW, cosine decay, "
              r"20 warmup steps.}" % (caption, r, alpha, ", rsLoRA" if rs else ""),
              r"\label{%s}" % label, r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default=None, help="write the table here (default: print only)")
    p.add_argument("--exclude", default=r"inoc|layers|nosys", help="regex on run names to leave out")
    p.add_argument("--incomplete", action="store_true", help="include runs with no evals.json yet")
    p.add_argument("--caption", default="Finetuning hyperparameters by model and dataset.")
    p.add_argument("--label", default="tab:finetune-recipes")
    args = p.parse_args()
    cells = scan(Path(args.runs), re.compile(args.exclude), args.incomplete)
    if not cells:
        raise SystemExit(f"no plain finetunes under {args.runs}")
    tex = render(cells, args.caption, args.label)
    print(tex)
    if args.out:
        Path(args.out).write_text(tex)
        print(f"wrote {args.out}  ({sum(len(v) for v in cells.values())} runs, {len(cells)} rows)")


if __name__ == "__main__":
    main()
