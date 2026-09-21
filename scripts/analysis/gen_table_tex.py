"""One prompt per benchmark, its response at every sparsity of one mask, as a LaTeX table.

The paper twin of ``gen_table.py``: that script puts several prompts of ONE eval side by side in
HTML; this one puts ONE prompt from each of several evals side by side in ``booktabs`` LaTeX, so a
reader of the refusal table can see the text behind its four columns (StrongREJECT, SORRY-Bench,
GSM8K, IFEval) across the L0 sweep. Everything that makes the HTML table trustworthy is imported
rather than re-derived -- the record-to-condition reconstruction and its exact check against
``evals.json`` (see ``gen_table.py``'s docstring for why that check exists) -- and the one thing
this script adds on top is that the four evals may live in DIFFERENT eval directories of the run,
because the capability and harm sweeps are separate ``eval`` invocations::

    uv run python scripts/analysis/gen_table_tex.py \\
        --dir plots/data/refusal_masks/refusal_grpo_8b_uniform_vllm_native \\
        --source strongreject=eval_native --source sorrybench=eval_extra,eval_extra_hi \\
        --source gsm8k=eval_native --source ifeval=eval_extra,eval_extra_hi \\
        --seed 0 --out ../learning-to-attribute-paper/tabs/refusal_generations_8b.tex

``--source EVAL=DIR[,DIR...]``: each DIR holds ``evals.json`` and ``<EVAL>_eval/generations.jsonl``;
several DIRs are merged by condition (a condition present in more than one keeps the FIRST -- the
two IFEval/SORRY-Bench runs above each re-measured ``pretrained`` and ``full_delta``, and the two
measurements differ by HF batching noise). Rows are the conditions every eval has, unless
``--fracs`` names them. The prompt for each eval is drawn by ``--seed`` and its index is written
into the .tex as a comment, so the draw is reproducible and overridable (``--index EVAL=i``).

The per-cell verdict is always a field the eval stored (judge score, checker verdict), never a
re-scoring; the split rate printed beside it is ``evals.json``'s, which the reconstruction was
verified against. Cells are tinted by the verdict -- red for a harm eval's compliance, green for a
capability eval's correctness -- at a fraction of the paper's row-colour strength.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_table import ORGANISMS, aggregate, reconstruct  # noqa: E402

#: the two evals the HTML script does not know, on its spec shape. ``value`` is the stored verdict
#: as 0-1; ``rate`` the evals.json key the reconstruction is checked against, ``agg`` how that key
#: is formed from the values.
ORGANISMS = {
    **ORGANISMS,
    "sorrybench": dict(
        eval="sorrybench", rate="frac_comply", agg="mean",
        value=lambda r, cfg: float(r.get("score") or 0.0),
        rate_fmt="{:.2f}"),
    "ifeval": dict(
        eval="ifeval", rate="prompt_strict", agg="percent",
        value=lambda r, cfg: float(all(r.get("strict") or [False])),
        rate_fmt="{:.1f}%"),
}

#: column header, harm-or-capability (decides the tint colour), and the split each eval reports
COLUMNS = {
    "strongreject": ("StrongREJECT", "harm", "off_target"),
    "sorrybench": ("SORRY-Bench", "harm", "off_target"),
    "gsm8k": ("GSM8K", "cap", "gsm8k"),
    "ifeval": ("IFEval", "cap", "ifeval"),
}


def _num(x):
    return f"{x:g}" if isinstance(x, (int, float)) else str(x)


def verdict(name, r) -> str:
    """The stored verdict for one response, as plain text (escaped by the caller)."""
    if name == "strongreject":
        return f"judge {float(r.get('score') or 0):.2f}"
    if name == "sorrybench":
        return "complied" if (r.get("score") or 0) >= 0.5 else "refused"
    if name == "gsm8k":
        return (("correct" if r.get("correct") else "wrong")
                + f" (pred {_num(r.get('pred'))}, gold {_num(r.get('gold'))})")
    if name == "ifeval":
        s = r.get("strict") or []
        return f"{sum(s)}/{len(s)} instructions (strict)"
    return ""


_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}",
        "“": "``", "”": "''", "‘": "`", "’": "'", "—": "---",
        "–": "--", "…": r"\ldots{}"}


def esc(s: str) -> str:
    """LaTeX-escape free text; newlines become a visible glyph so IFEval's formatting survives."""
    s = "".join(_ESC.get(ch, ch) for ch in s)
    s = re.sub(r"[ \t]*\n+[ \t]*", r" {\\color{gray}$\\hookleftarrow$} ", s)
    return re.sub(r"[ \t]+", " ", s).strip()


def clip(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


def cond_key(c):
    """Sort key: pretrained, ascending fraction, full."""
    if c == "pretrained":
        return -1.0
    if c in ("frac_1", "full_delta", "full"):
        return 2.0
    return float(c.split("_", 1)[1])


def load_eval(name, dirs, run: Path, cfg):
    """``(by_condition {cond: [record]}, rates {cond: float}, prompts [str])`` for one eval, merged
    over its directories, every directory verified against its own evals.json first."""
    spec, split = ORGANISMS[name], COLUMNS[name][2]
    merged, rates, prompts = {}, {}, None
    for d in dirs:
        blob = json.loads((run / d / "evals.json").read_text())["final"]
        rows = [json.loads(l) for l in (run / d / f"{name}_eval" / "generations.jsonl")
                .read_text().splitlines() if l.strip()]
        # full_delta duplicates frac_1 only where frac_1 exists (gen_table's rule); a sweep run at
        # chosen --fracs has no frac_1 and full_delta then has its own records
        conds = [c for c in blob if not (c == "full_delta" and "frac_1" in blob)]
        by, _ = reconstruct(rows, conds)
        for c in conds:
            recs = by[c][split]
            got, want = aggregate(spec, recs, cfg), blob[c][name][split][spec["rate"]]
            if abs(got - want) > 1e-9:
                sys.exit(f"MISMATCH {name}/{d}/{c}: reconstructed {got} vs evals.json {want}")
            key = "full" if c in ("frac_1", "full_delta") else c
            if key not in merged:
                merged[key], rates[key] = recs, want
            ps = [r["prompt"] for r in recs]
            if prompts is None:
                prompts = ps
            elif ps != prompts:
                sys.exit(f"{name}/{d}: prompt set differs from {dirs[0]}'s -- cannot align rows")
        print(f"verified {name}/{d}: {len(conds)} conditions")
    return merged, rates, prompts


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="a pulled run directory (config.yaml + eval dirs)")
    p.add_argument("--source", action="append", required=True, metavar="EVAL=DIR[,DIR]",
                   help="one column; order is column order")
    p.add_argument("--index", action="append", default=[], metavar="EVAL=I",
                   help="fix an eval's prompt index instead of drawing it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--refused-only", action="store_true",
                   help="draw a harm eval's prompt among those the PRETRAINED condition refused "
                        "(stored verdict < 0.5): a prompt the Instruct model already answers "
                        "shows nothing about the sweep")
    p.add_argument("--fracs", nargs="*", default=None,
                   help="rows, e.g. pretrained 0.001 0.01 full; default: every condition all "
                        "evals share")
    p.add_argument("--maxlen", type=int, default=330, help="characters per response")
    p.add_argument("--prompt-maxlen", type=int, default=240)
    p.add_argument("--font", default="tiny", help="LaTeX size command for the body")
    p.add_argument("--colwidth", default="3.05cm")
    p.add_argument("--longtable", metavar="LABEL", default=None,
                   help="emit a longtable (breaks across pages, header repeated) with this "
                        "\\label; the caption is a macro the including file defines (see "
                        "--caption-macro), so the prose stays with the paper, not the generator")
    p.add_argument("--caption-macro", default="genTableCaption",
                   help="with --longtable: \\caption{\\<macro>} is emitted, undefined-if-missing; "
                        "pass '' for a caption-less table (no \\caption, no \\label, and the "
                        "continuation header says only '(continued)', since without a caption "
                        "\\thetable is the PREVIOUS table's number)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    run = Path(args.dir)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    sources = [(s.split("=")[0], s.split("=")[1].split(",")) for s in args.source]
    fixed = {s.split("=")[0]: int(s.split("=")[1]) for s in args.index}
    rng = random.Random(args.seed)

    cols = []
    for name, dirs in sources:
        merged, rates, prompts = load_eval(name, dirs, run, cfg)
        pool = list(range(len(prompts)))
        if args.refused_only and COLUMNS[name][1] == "harm":
            pool = [j for j in pool if ORGANISMS[name]["value"](merged["pretrained"][j], cfg) < 0.5]
        i = fixed.get(name, rng.choice(pool))
        cols.append((name, merged, rates, i, prompts[i]))
        print(f"  {name:12} [{i}/{len(prompts)}] {prompts[i][:90]!r}")

    shared = set.intersection(*(set(m) for _, m, _, _, _ in cols))
    if args.fracs:
        want = ["pretrained" if f == "pretrained" else "full" if f == "full" else f"frac_{f}"
                for f in args.fracs]
        missing = [w for w in want if w not in shared]
        if missing:
            sys.exit(f"--fracs {missing} not in every eval; shared: {sorted(shared, key=cond_key)}")
        rows = want
    else:
        rows = sorted(shared, key=cond_key)
    print(f"rows: {rows}")

    def l0(c):
        if c == "pretrained":
            return r"0 {\color{gray}(Instruct)}"
        if c == "full":
            return r"100 {\color{gray}(Base)}"
        return f"{100 * float(c.split('_', 1)[1]):g}"

    spec_cols = " ".join(f"p{{{args.colwidth}}}" for _ in cols)
    header = [
        r"$\|\Delta\theta\|_0$ (\%) & " + " & ".join(
            f"\\textbf{{{COLUMNS[n][0]}}}" for n, *_ in cols) + r" \\",
        " & " + " & ".join(
            f"{{\\color{{gray}}{esc(clip(ORGANISMS[n].get('prompt_fn', lambda s: s)(pr), args.prompt_maxlen))}}}"
            for n, _, _, _, pr in cols) + r" \\",
        r"\midrule",
    ]
    out = [
        f"% generated by scripts/analysis/gen_table_tex.py from {run.name}; seed {args.seed}; "
        + "; ".join(f"{n}[{i}]" for n, _, _, i, _ in cols),
        f"% (the index is the prompt's position in its split; pass --index EVAL=I to keep it)"
        + ("; harm prompts drawn among those the pretrained model refused" if args.refused_only
           else ""),
        f"{{\\{args.font}\\setlength{{\\tabcolsep}}{{3pt}}\\renewcommand{{\\arraystretch}}{{1.1}}",
    ]
    if args.longtable:
        # the header repeats on every page; the prompts are only on the first, so a continued
        # page reads the column NAMES over responses whose prompt is one page back -- that is why
        # the continuation header carries "(continued)" and the split rate stays in every cell
        cap = ([f"\\caption{{\\{args.caption_macro}}}\\label{{{args.longtable}}} \\\\"]
               if args.caption_macro else [])
        cont = "\\tablename~\\thetable{} (continued)" if args.caption_macro else "(continued)"
        out += [f"\\begin{{longtable}}{{r {spec_cols}}}", *cap,
                r"\toprule", *header, r"\endfirsthead",
                f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{{{\\color{{gray}}{cont}}}}} \\\\ \\toprule",
                *header, r"\endhead",
                r"\bottomrule \endlastfoot"]
    else:
        out += [f"\\begin{{tabular}}{{r {spec_cols}}}", r"\toprule", *header]
    for c in rows:
        cells = []
        for name, merged, rates, i, _ in cols:
            r, spec = merged[c][i], ORGANISMS[name]
            v = max(0.0, min(1.0, spec["value"](r, cfg)))
            hue = "red" if COLUMNS[name][1] == "harm" else "green"
            tint = f"\\cellcolor{{{hue}!{round(18 * v)}}} " if v > 0.02 else ""
            rate_txt = spec.get("rate_fmt", "{:.2f}").format(rates[c])
            head = f"{esc(verdict(name, r))} $\\cdot$ split {esc(rate_txt)}"
            cells.append(f"{tint}{{\\color{{gray}}{head}}}\\newline "
                         f"{esc(clip(r['response'], args.maxlen))}")
        out.append(f"{l0(c)} & " + " & ".join(cells) + r" \\")
    out += [r"\end{longtable}}"] if args.longtable else [r"\bottomrule", r"\end{tabular}}"]

    dest = Path(args.out)
    dest.write_text("\n".join(out) + "\n")
    print(f"wrote {dest} ({len(rows)} rows x {len(cols)} columns)")


if __name__ == "__main__":
    main()
