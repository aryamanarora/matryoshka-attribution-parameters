"""Every refusal mask fitted so far, at both scales, swept together -- the replicate view of
``plot_refusal_sparsity_facets.pdf``.

That figure draws ONE run (``refusal_grpo_logk_v2``) and is the paper figure; this one draws all
six GRPO-fitted refusal masks on one grid, so the question it answers is not "how sparse is
refusal" but "how much of that curve is the run". Within a scale they are the same sweep over the
same delta (Instruct -> Base; 603,425 nonresid units at 1B, 1,703,936 at 8B), differing only in how
the mask's scores were fitted:

    run                               model  k-schedule  sampler when fitting  reward frame
    refusal_grpo_logk_v2              1B     log         HF                    URIAL .help
    refusal_grpo_logk_v3              1B     log         vLLM                  URIAL .help
    refusal_grpo_uniform_hf           1B     uniform     HF                    URIAL .help
    refusal_grpo_uniform_vllm         1B     uniform     vLLM                  URIAL .help
    refusal_grpo_uniform_vllm_native  1B     uniform     vLLM                  native template
    refusal_grpo_8b_logk              8B     log         HF                    URIAL .help

So the 1B column is a 2x2 (schedule x sampler) plus one frame variant, and the 8B column is the
top-left cell of that square repeated at scale -- the one cell present at both sizes, which is what
makes the two columns comparable at all. Colour carries schedule-and-frame and LINETYPE carries the
sampler: the sampler is the axis that should make no difference, and putting it on the channel the
eye reads second is the honest way to show that it mostly does not. (The repo's usual rule -- colour
is the method, linetype the hyperparameter -- does not decide this one, because all six cells are
the same method.)

THE SECOND 8B RUN IS MISSING FOR A REASON. ``refusal_grpo_8b_logk_v2`` (300 steps against
``8b_logk``'s 60, and vLLM rather than HF) exists and finished, but its only sweep is its own
URIAL-frame ``evals.json`` with StrongREJECT alone -- no native re-eval and no capability probes at
all. Drawing it would mean one line scored in a different frame from the other five and absent from
two of the three benchmarks, which is worse than an absence a caption can explain. It is one
``eval/__main__.py`` job away from being here, and it is worth running: at 60 steps the 8B mask
reaches 0.762 and at 300 steps it peaks at 0.321, so the cell that is missing is the one that says
the 8B fit gets WORSE with more optimisation.

EVERY CURVE IS THE NATIVE-TEMPLATE RE-EVAL (``<run>/eval_native/evals.json``), including the five
runs whose reward was collected under URIAL. That is the only way six fits are comparable, and it
is the frame the paper reports; a run's own ``evals.json`` is scored under the frame it was FITTED
in. The two are not interchangeable and the gap is not uniform across runs -- ``logk_v2`` reads
0.692 natively against 0.711 under URIAL at 1%, while ``uniform_vllm`` reads 0.219 against 0.761 at
the same k. Whatever that difference is, it is a property of the individual fit, so do not
substitute one sweep for the other to fill a gap.

ONE PANEL PER (MODEL, BENCHMARK), NOT THE PAPER FIGURE'S CAPABILITY MEAN. With six series the mean
stops being a summary and starts being a hiding place: MMLU is completion-format and sits at the
Instruct anchor until the mask is ~5% dense, GSM8K is generative and starts falling immediately, so
their mean reports the damage as milder than it is and says nothing about WHICH competence went.
The split is what makes ``logk_v3``'s 1% point legible -- 21.5 on GSM8K against 44.5 on MMLU, where
the mean showed one unremarkable 33.0. Each panel keeps its OWN y scale rather than sharing one per
row, because 8B GSM8K runs to 80 and 1B GSM8K to 35, and a shared row scale would flatten the 1B
curves into the floor to make room for a column they are not being compared against point-for-point
anyway. Compare SHAPES across the two columns and values only within one.

IFEVAL IS ABSENT ON PURPOSE. Only ``logk_v2`` and ``8b_logk`` have IFEval sweeps
(``eval_ifeval_sweep/``, their own jobs), so a fourth row would carry one line per column and read
as runs that failed rather than jobs never submitted. The paper figure's capability mean does
include it; that is one more reason the two figures' capability axes are not the same number.

THE ANCHORS ARE DRAWN ONCE PER PANEL, NOT PER RUN. ``pretrained`` and ``full_delta`` are the same
weights in every sweep of a given scale (Instruct and Base), so their measurements are several
decodes of one number: the line is their MEDIAN and the printed summary gives the full spread,
which is a free read on the decode noise of every other point in the figure (at 1B StrongREJECT's
``pretrained`` spans 0.036-0.043 and GSM8K's spans 29-32, while MMLU's is identical in all five
because it is forward-only and sees no sampler). The safety panels' Base anchor is base weights
under URIAL's no-refusal prompt -- base weights under the instruct template score 0.07 because they
are incoherent there, which is a floor produced by damage and not a refusal number to read a mask
against. That anchor exists only at 1B (the base-weight prompt ladder was never run at 8B, being a
question about the prompt rather than the scale), so the 8B safety panel carries the Instruct line
alone.

Error bars are +-1 standard error: StrongREJECT's from the 60 per-response judge scores recorded in
each sweep's ``generations.jsonl``, the two capability panels' from the binomial ``stderr`` the eval
records (n=200 GSM8K, n=512 MMLU). They are drawn thin and in the series colour because with five
overlapping curves in a panel the bars are the only thing that says which separations are real --
most of the spread between the two log-k runs is not, and the gap between the log-k and uniform-k
pairs at 1% is.

The dotted vertical is the 1% budget, the sparsity the paper quotes at 1B. Note the 8B curve's knee
is at 5%, which is why `plot_baseline_strongreject.py`'s 8B bar is labelled "MAttr 5%".

    uv run python plots/plot_refusal_sparsity_all.py
    uv run python plots/plot_refusal_sparsity_all.py --png
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib import font_manager
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_hline, geom_line,
    geom_linerange, geom_point, geom_text, geom_vline, ggplot, labs, scale_color_manual,
    scale_linetype_manual, scale_shape_manual, scale_x_log10, theme, theme_bw, theme_set,
)

matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType outlines, not Type-3

from palette import MODEL, _up   # noqa: E402  the hues the refusal figures already spend
import plot_baseline_strongreject as B   # noqa: E402  the baseline cells, their metrics and their L0

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        figure_size=(5.4, 4.4),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
        panel_spacing_y=0.03,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_position="top",
        legend_direction="horizontal",
        legend_box="vertical",
        legend_title=element_text(size=6.5),
        legend_text=element_text(size=6),
        legend_key_size=7,
        legend_box_margin=0,
        legend_margin=0,
    )
)

RUNS = Path(__file__).parent / "data" / "refusal"
SWEEP = "eval_native/evals.json"
SIDE = ["eval_extra/evals.json", "eval_ifeval_sweep/evals.json", "eval_sorrybench/evals.json",
        "eval_ifeval/evals.json"]

#: (run directory, model, colour key, sampler). The colour key is schedule-and-frame; the sampler
#: takes the linetype. Order is the legend order and the reading order: the paper's run first.
CELLS = [
    ("refusal_grpo_logk_v2",             "1B", "URIAL",    "log",     "HF"),
    ("refusal_grpo_logk_v3",             "1B", "URIAL",    "log",     "vLLM"),
    ("refusal_grpo_uniform_hf",          "1B", "URIAL",    "uniform", "HF"),
    ("refusal_grpo_uniform_vllm",        "1B", "URIAL",    "uniform", "vLLM"),
    ("refusal_grpo_logk_vllm_native",    "1B", "native",   "log",     "vLLM"),
    ("refusal_grpo_uniform_vllm_native", "1B", "native",   "uniform", "vLLM"),
    ("refusal_grpo_logk_vllm_native_kl001",   "1B", "+KL",      "log", "vLLM"),
    ("refusal_grpo_logk_vllm_native_obligrpo", "1B", "+KL, A>0", "log", "vLLM"),
    ("refusal_grpo_8b_logk",             "8B", "URIAL",    "log",     "HF"),
    ("refusal_grpo_8b_logk_vllm_native", "8B", "native",   "log",     "vLLM"),
    ("refusal_grpo_8b_uniform_vllm_native", "8B", "native", "uniform", "vLLM"),
]
MODELS = ["1B", "8B"]
#: THE ENCODING CARRIES THE OBJECTIVE, NOT JUST THE FRAME (changed when the KL and
#: positive-advantage arms landed, 2026-09-13). Colour is what the GRPO reward and loss WERE --
#: the frame the reward was collected in, plus any deviation from plain group-relative GRPO -- and
#: linetype is the k-schedule, per palette.LS's convention that a hyperparameter takes the dash.
#: The SAMPLER (HF vs vLLM) is still not encoded: the pairs that differ only in it (logk_v2/v3,
#: uniform_hf/uniform_vllm) are drawn IDENTICALLY, so each is two same-styled lines whose
#: separation is the replicate spread of the whole pipeline.
#:
#: `+KL` is kl_coef 0.01 against the starting policy; `+KL, A>0` adds the positive-advantage gate,
#: i.e. GRP-Obliteration's objective applied to the MASK SCORES rather than to the weights. Both
#: are native-frame and log-k, so within a scale they are directly comparable to `native` dashed.
FRAME_COLOR = {"URIAL": MODEL["MAttr"], "native": _up.METHOD["DBM"],
               "+KL": _up.METHOD["Node Pruning"], "+KL, A>0": "#000000"}
SCHED_LS = {"uniform": "solid", "log": "dashed"}

#: The safety ceiling, 1B only: base weights under URIAL's no-refusal prompt.
BASE_CEILING = {"1B": ("Base (URIAL .help)", "baseline_strongreject_llama32_1b_urial_help")}
#: (benchmark title, eval, split, metric, stderr key). Safety first, because every capability panel
#: beside it is read as "at what cost".
BENCH_ALL = [("StrongREJECT", "strongreject", "off_target", "score", None),
             ("GSM8K (%)", "gsm8k", "gsm8k", "accuracy", "stderr"),
             ("MMLU (%)", "mmlu", "mmlu", "accuracy", "stderr"),
             ("SORRY-Bench (%)", "sorrybench", "off_target", "fulfillment", "stderr"),
             ("IFEval (%)", "ifeval", "ifeval", "prompt_strict", "stderr")]
#: the metrics reported on [0, 1] that this figure prints as percentages, so every capability row
#: shares one scale with the two judges' rows
PCT100 = {"sorrybench"}
#: SIDE SWEEPS. SORRY-Bench and IFEval were added after the main sweep and run as their own jobs,
#: so they live beside it and cover FEWER sparsities -- 10 conditions for the two runs that got a
#: full IFEval sweep, 1-4 for everything else, and nothing at all for six of the eleven runs. They
#: are merged by CONDITION LABEL, and a condition a side sweep does not carry simply yields no
#: point rather than dropping that condition from the other rows. Hence the bottom two rows being
#: sparse where the top three are complete: that is coverage, not failure, and the printed summary
#: names the count per run.
#: ``--benches sr,cap`` drops MMLU and leaves a 2x2 -- safety over capability, 1B beside 8B. That
#: is the shape of the paper's `refusal_sparsity_facets.pdf` with the model as a second facet
#: dimension, and MMLU is the row to drop because it is flat at the Instruct anchor in every run at
#: every sparsity until the mask is dense (which is a finding, but one the 3x2 already records).
BENCH_KEYS = {"sr": 0, "cap": 1, "mmlu": 2, "sb": 3, "ife": 4}
BENCH = list(BENCH_ALL)
#: facet_wrap fills row-major, so this order is "one benchmark per row, one model per column".
PANELS = [f"{m} · {b}" for b, *_ in BENCH for m in MODELS]


#: THE BASELINES, PLOTTED AT THEIR OWN EDITED-FRACTION. Once the x axis is "weights changed" rather
#: than "mask sparsity", a one-shot edit is not a different kind of object from a sweep -- it is a
#: single point on the same axes, at the L0 measured in docs/refusal/l0_baselines.md. That is the
#: comparison the figure exists to make and it cannot be made on a sparsity axis at all, which is
#: why the paper's facets figure never carried these.
#: (key, label). The metrics come from `plot_baseline_strongreject`'s own loader, so a baseline
#: cannot read one number here and another in the table.
#: SHAPE rather than an in-panel text label: four baselines land within a factor of two of each
#: other in x (38.9-49.4% at 1B) and their labels collided with each other and with the curves. A
#: shape legend costs one legend row and never overlaps anything.
BASE_CELLS = [("ablit", "Abliteration"), ("grpo", "GRPO"), ("grpo_kl", "GRPO+KL"),
              ("oblit", "GRP-Oblit")]
BASE_SHAPE = {"Abliteration": "D", "GRPO": "s", "GRPO+KL": "^", "GRP-Oblit": "v"}
#: which of this figure's rows a baseline metric maps onto
BASE_METRIC = {"StrongREJECT": ("sr", 0.01), "GSM8K (%)": ("cap", 1.0)}


def baseline_points():
    """``DataFrame(panel, pct, y, label)`` -- every baseline cell that has an L0 and a metric."""
    rows = []
    for scale, cells in (("1B", B.CELLS), ("8B", B.CELLS_8B)):
        recs = {r[4]: r for r in B.bars(cells=cells)}
        for key, label in BASE_CELLS:
            if key not in recs or (scale, key) not in B.L0:
                continue
            met = recs[key][2]
            for bench, *_ in BENCH:
                if bench not in BASE_METRIC:
                    continue
                mk, scale_y = BASE_METRIC[bench]
                v = met[mk][0]
                if v is None or math.isnan(v):
                    continue
                rows.append(dict(panel=panel_of(scale, bench), pct=B.L0[(scale, key)],
                                 y=v * scale_y, label=label))
    return pd.DataFrame(rows)


def panel_of(model: str, bench: str) -> str:
    return f"{model} · {bench}"


def sr_stderr(sweep_path: Path) -> dict:
    """SE of each condition's StrongREJECT mean, from the ``generations.jsonl`` beside the sweep
    (one judged score per response, n=60). Empty when the file was not copied in."""
    p = sweep_path.parent / "strongreject_eval" / "generations.jsonl"
    if not p.exists():
        return {}
    by = {}
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("split") == "off_target" and r.get("score") is not None:
            by.setdefault(r["condition"], []).append(r["score"])
    return {c: statistics.stdev(v) / math.sqrt(len(v)) for c, v in by.items() if len(v) > 1}


def frac_of(cond: str):
    return {"pretrained": 0.0, "full_delta": 1.0}.get(
        cond, float(cond[len("frac_"):]) if cond.startswith("frac_") else None)


def read_run(run: str, model: str):
    """One run's sweep as LONG rows -- one per (condition, benchmark) that HAS a value.

    A benchmark missing for a condition yields no row, rather than dropping the condition from
    every other benchmark: the side sweeps cover fewer sparsities than the main one, and a
    complete-case rule over five benchmarks would leave almost nothing.
    """
    src = RUNS / run / SWEEP
    if not src.exists():
        raise SystemExit(f"missing {src} -- copy it from runs/{run}/{SWEEP}")
    fin = json.loads(src.read_text())["final"]
    side = [json.loads((RUNS / run / p).read_text())["final"]
            for p in SIDE if (RUNS / run / p).exists()]
    se = sr_stderr(src)

    def lookup(cond, ev, sp, metric, sek):
        for blob in [fin] + side:
            m = (blob.get(cond, {}) or {}).get(ev, {}).get(sp)
            if isinstance(m, dict) and metric in m:
                return m[metric], (m.get(sek) if sek else se.get(cond, float("nan")))
        return None, None

    rows = []
    for cond in fin:
        f = frac_of(cond)
        if f is None:
            continue
        for bench, ev, sp, metric, sek in BENCH:
            v, e = lookup(cond, ev, sp, metric, sek)
            if v is None:
                continue
            mult = 100.0 if ev in PCT100 else 1.0
            rows.append(dict(run=run, model=model, cond=cond, frac=f, bench=bench,
                             panel=panel_of(model, bench), y=v * mult,
                             se=(float("nan") if e is None else e * mult)))
    return pd.DataFrame(rows).sort_values("frac"), []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", action="store_true", help="also write a 300 dpi PNG, for looking at")
    ap.add_argument("--benches", default="sr,cap,mmlu",
                    help="comma-separated rows from sr,cap,mmlu (default: all three)")
    ap.add_argument("--width", type=float, default=5.4)
    ap.add_argument("--height", type=float, default=4.4)
    args = ap.parse_args(argv)
    global BENCH, PANELS
    if bad := [k for k in args.benches.split(",") if k not in BENCH_KEYS]:
        raise SystemExit(f"unknown bench(es) {bad}; choose from {sorted(BENCH_KEYS)}")
    BENCH = [BENCH_ALL[BENCH_KEYS[k]] for k in args.benches.split(",")]
    PANELS = [panel_of(m, b) for b, *_ in BENCH for m in MODELS]

    frames, dropped = [], {}
    for run, model, frame, sched, sampler in CELLS:
        df, drop = read_run(run, model)
        df["frame"], df["sched"], df["sampler"] = frame, sched, sampler
        frames.append(df)
        if drop:
            dropped[run] = drop
    df = pd.concat(frames, ignore_index=True)

    # ---- the endpoints are the SAME weights in every run of a scale, so they are one line apiece
    # drawn at the median of those decodes; the spread goes to stdout as this figure's noise floor.
    ends = {c: df[df.cond == c] for c in ("pretrained", "full_delta")}
    long = df[~df.cond.isin(["pretrained", "full_delta", "frac_1"])].copy()  # frac_1 == full_delta
    long["pct"] = long["frac"] * 100
    long["lo"], long["hi"] = long.y - long.se, long.y + long.se

    refs = [dict(panel=p, model="Instruct", label="Instruct",
                 y=ends["pretrained"].query("panel == @p").y.median()) for p in PANELS]
    refs += [dict(panel=p, model="Base", label="Base",
                  y=ends["full_delta"].query("panel == @p").y.median())
             for p in PANELS if not p.endswith(BENCH[0][0])]
    for m, (label, run) in BASE_CEILING.items():
        p_ceil = RUNS / run / "evals.json"
        if p_ceil.exists():
            y = json.loads(p_ceil.read_text())["final"]["dense"]["strongreject"]["off_target"]
            refs.append(dict(panel=panel_of(m, BENCH[0][0]), model="Base", label=label,
                             y=y["score"]))
    refs = pd.DataFrame(refs)

    for d in (long, refs):
        d["panel"] = pd.Categorical(d["panel"], PANELS, ordered=True)
    base = baseline_points()
    if len(base):
        base["panel"] = pd.Categorical(base["panel"], PANELS, ordered=True)
    long["frame"] = pd.Categorical(long["frame"], list(FRAME_COLOR), ordered=True)
    long["sched"] = pd.Categorical(long["sched"], list(SCHED_LS), ordered=True)

    # ---- anchor labels. Placement is computed from each panel's own geometry rather than fixed,
    # because the same anchor is the top of one panel and the bottom of another: an Instruct line
    # labelled "above" is off the top edge on MMLU and buried in the curves' origin on safety. The
    # rule: label below the line when the line sits in the panel's top third, above it otherwise;
    # and at the right edge when the curves' LEFT end is nearer the line than their right end is.
    lab = []
    for r in refs.itertuples():
        d = long[long.panel == r.panel]
        same = refs[refs.panel == r.panel]
        lo, hi = min(d.y.min(), same.y.min()), max(d.y.max(), same.y.max())
        span = (hi - lo) or 1.0
        right = abs(d.y.iloc[0] - r.y) < abs(d.y.iloc[-1] - r.y)
        lab.append(dict(panel=r.panel, model=r.model, label=r.label, y=r.y,
                        va="top" if (r.y - lo) / span > 0.66 else "bottom",
                        # nudged a hair inside the data range: right-aligned text AT the last
                        # x still overhangs the panel, because the axis expands past that point
                        # by less than the label is wide
                        x=d.pct.max() * 0.85 if right else d.pct.min(),
                        ha="right" if right else "left"))
    lab = pd.DataFrame(lab)
    lab["panel"] = pd.Categorical(lab["panel"], PANELS, ordered=True)

    txt = [geom_text(lab[(lab.va == va) & (lab.ha == ha)],
                     aes("x", "y", label="label", color="model"),
                     ha=ha, va=va, size=5, inherit_aes=False)
           for va in ("top", "bottom") for ha in ("left", "right")
           if len(lab[(lab.va == va) & (lab.ha == ha)])]

    p = (
        ggplot(long, aes("pct", "y", color="frame", linetype="sched", group="run"))
        + facet_wrap("~panel", ncol=len(MODELS), scales="free_y")
        + geom_vline(xintercept=1.0, linetype="dotted", size=0.4, color="#999999")
        + geom_hline(refs, aes(yintercept="y", color="model"), linetype="dashed", size=0.5,
                     inherit_aes=False)
        + txt
        + geom_linerange(long.dropna(subset=["se"]), aes(ymin="lo", ymax="hi"), size=0.35,
                         alpha=0.55, show_legend=False)
        + geom_line(size=0.6)
        + geom_point(size=1.2, stroke=0, show_legend=False)
        + geom_point(base, aes("pct", "y", shape="label"), color="#000000", fill="none",
                     size=1.9, stroke=0.6, inherit_aes=False)
        + scale_shape_manual(values=BASE_SHAPE, name="Baseline")
        + scale_x_log10(breaks=[0.1, 1, 10, 100], labels=["0.1", "1", "10", "100"])
        + scale_color_manual(values={**FRAME_COLOR, **MODEL}, breaks=list(FRAME_COLOR),
                             name="Objective")
        + scale_linetype_manual(values=SCHED_LS, name="$k$ schedule")
        + labs(x="Finetune parameters changed (%)", y="")
    )

    out = Path(args.out) if args.out else Path(__file__).parent / "refusal_sparsity_all.pdf"
    p = p + theme(figure_size=(args.width, args.height))
    p.save(out, verbose=False)
    if args.png:
        p.save(out.with_suffix(".png"), dpi=300, verbose=False)

    print(f"wrote {out}: {len(CELLS)} runs, sweep={SWEEP}, frame=native template, "
          f"{len(PANELS)} panels, {long.cond.nunique()} sparsities")
    for c in ("pretrained", "full_delta"):
        for pn in PANELS:
            e = ends[c].query("panel == @pn").y
            print(f"  {c:11s} {pn:22s} {e.min():.3f}-{e.max():.3f} (median {e.median():.3f})"
                  "   <- same weights in every run of that scale, so this is decode noise")
    for run, model, frame, sched, sampler in CELLS:
        r = long[long.run == run]
        one = r[r.cond == "frac_0.01"].set_index("bench").y
        sr = r[r.bench == BENCH[0][0]]
        print(f"  {run:34s} {model}  {frame:6s} {sched:7s} {sampler:4s}  1%: "
              + "  ".join(f"{b.split(' (')[0]} {one[b]:.3f}" for b, *_ in BENCH if b in one)
              + f"   peak SR {sr.y.max():.3f} at {sr.loc[sr.y.idxmax(), 'cond']}")
    thin = {b for b, *_ in BENCH} - set(long.groupby("bench").cond.nunique()
                                       [lambda x: x >= long.cond.nunique()].index)
    if thin:
        print("  rows with partial coverage (conditions measured, per run):")
        for b in [x for x, *_ in BENCH if x in thin]:
            per = (long[long.bench == b].groupby("run").cond.nunique().sort_values(ascending=False))
            print(f"    {b:18s} " + ", ".join(f"{r.split('refusal_grpo_')[-1]}={n}"
                                              for r, n in per.items()))
    for run, drop in dropped.items():
        print(f"  {run}: dropped for a missing metric: {drop}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
