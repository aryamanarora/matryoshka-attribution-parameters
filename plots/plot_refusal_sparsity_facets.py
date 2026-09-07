"""Refusal-mask sparsity sweep, safety beside capability -- half-width, two stacked panels.

One GRPO-fitted refusal mask (Llama-3.2-1B-Instruct -> Base, ``configs/refusal/``), swept over the
fraction of the base->instruct delta it keeps. Top panel: StrongREJECT on the 60 reported
forbidden prompts (up = refusal eroded). Bottom panel: capability, the mean of whichever of MMLU,
GSM8K and IFEval the run has swept (down = damage). The three are averaged raw because all three
are percentages on one 0-100 scale; that is the only thing that makes the mean a number rather
than a category error, and it is why StrongREJECT (a 0-1 judge score) is a separate panel rather
than a third term.

ONE POINT ON THE CAPABILITY CURVE IS DAMAGE RATHER THAN A SCORE, and it is worth a caption note:
at ``frac_0.5`` the mask leaves 65% of IFEval responses EMPTY, so its 6.1 there measures a broken
generator and not instruction following. GSM8K (3.5) and MMLU (30.1) agree that the model is
wrecked at that sparsity, so the mean is not misleading about the direction -- but do not quote
the IFEval component of it as a following rate. ``empty_frac`` is in the sweep's own JSON. The 1% point is ringed. Error bars are +-1 standard error: StrongREJECT's from the
60 per-response judge scores the eval records per condition, capability's from the two probes'
binomial standard errors combined for the mean. Dotted lines are the two
anchors per panel: on safety, the Instruct model under its own template (the refusal floor, the
sweep's ``pretrained`` endpoint) and the Base model under URIAL's no-refusal prompt (the ceiling,
from ``configs/baseline/``); on capability, the two models' native-frame numbers read off the
sweep's own endpoints (``pretrained`` IS Instruct, ``full_delta`` IS Base).

DATA. Two sweeps exist per run, and the script takes the first it finds:

  <run>/posthoc_eval/evals.json   the NATIVE re-eval (configs/refusal/eval_native*.yaml): StrongREJECT
                                  under the instruct template, MMLU in completion format, GSM8K.
                                  The property-of-the-weights figure. Cluster-side only so far.
  <run>/evals.json                the training run's own final sweep: StrongREJECT under the URIAL
                                  .help frame the mask was FITTED under, and MMLU in chat format,
                                  which reads ~chance for a 1B model (an artifact, see
                                  configs/refusal/eval_native_mmlu_completion.yaml). Tracked.

The printed summary says which one was drawn and under which frame. Do not put a curve from one
next to anchors from the other without reading that line.

IFEval lives in a THIRD sweep, ``<run>/eval_ifeval_sweep/evals.json`` (configs/refusal/eval_ifeval.yaml
over the full grid), because it was added after the native re-eval had run and is generation-bound
enough (541 prompts x 1024 tokens x 11 conditions) to be its own job. When that file exists its
``prompt_strict`` joins the capability mean, matched to the native sweep by condition label; a
condition present in one sweep and not the other is DROPPED from the curve and named in the
printed summary, so a half-finished IFEval job cannot silently shorten the figure. Same decoder
(HF greedy) and frame (native template) as the other two probes, which is what makes the mean
one number.

    uv run python plots/plot_refusal_sparsity_facets.py
    uv run python plots/plot_refusal_sparsity_facets.py --run refusal_grpo_instruct_to_base_urial_help
    uv run python plots/plot_refusal_sparsity_facets.py --png   # also a 300 dpi PNG, for looking at
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
    geom_linerange, geom_point, geom_text, ggplot, labs, scale_color_manual, scale_x_log10, theme, theme_bw,
    theme_set,
)

matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType outlines, not Type-3

from palette import MODEL   # the Instruct/Base pair shared with plot_baseline_strongreject.py

FAMILY = ("Inter" if "Inter" in {f.name for f in font_manager.fontManager.ttflist}
          else "DejaVu Sans")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family=FAMILY),
        figure_size=(2.0, 2.4),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_y=0.03,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_position="none",
    )
)

RUNS = Path(__file__).parent / "data" / "all_runs"
DEFAULT_RUN = "refusal_grpo_logk_v2"   # the post-fix rerun; its posthoc_eval/ has all three metrics
# The Base ceiling: base weights under URIAL's no-refusal prompt, 0.589 -- the number a mask that
# strips refusal is read against (its plain-template 0.033 measures incoherence, not refusal).
# The run is named without `_base` on disk; see its config's filename note.
BASE_CEILING = ("Base (URIAL .help)", "baseline_strongreject_llama32_1b_urial_help")
#: (eval name, split, metric, label). Every one is a percentage on the same 0-100 scale, which is
#: the only thing that makes averaging them a number rather than a category error.
CAP_KEYS = [("mmlu", "mmlu", "accuracy", "MMLU"), ("gsm8k", "gsm8k", "accuracy", "GSM8K"),
            ("ifeval", "ifeval", "prompt_strict", "IFEval")]
IFEVAL_SWEEP = "eval_ifeval_sweep/evals.json"


def load_sweep(run_dir: Path):
    for rel, frame in (("posthoc_eval/evals.json", "native template"),
                       ("evals.json", "training frame")):
        p = run_dir / rel
        if p.exists():
            cfg = run_dir / "config.yaml"
            if rel == "evals.json" and cfg.exists():
                for line in cfg.read_text().splitlines():
                    if line.startswith("chat_template:"):
                        frame = line.split(":", 1)[1].strip()
            return json.loads(p.read_text())["final"], p, frame
    raise SystemExit(f"no evals.json under {run_dir}")


def merge_ifeval(fin: dict, run_dir: Path):
    """Fold the IFEval sweep's per-condition block into the native sweep, by condition label.

    Returns the conditions that could NOT be matched (in either direction), for the summary.
    """
    p = run_dir / IFEVAL_SWEEP
    if not p.exists():
        return None
    ife = json.loads(p.read_text())["final"]
    for cond, r in ife.items():
        if cond in fin and isinstance(r.get("ifeval", {}).get("ifeval"), dict):
            fin[cond]["ifeval"] = r["ifeval"]
    return sorted(set(fin) ^ set(ife))


def sr_stderr(sweep_path: Path) -> dict:
    """Standard error of each condition's StrongREJECT mean, from ``generations.jsonl`` beside
    the sweep (one judged score per response, n=60). Empty when the file is absent."""
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", action="store_true", help="also write a 300 dpi PNG beside the PDF")
    args = ap.parse_args(argv)

    fin, src, frame = load_sweep(RUNS / args.run)
    unmatched = merge_ifeval(fin, RUNS / args.run)
    have_cap = [c for c in CAP_KEYS
                if any(isinstance(r.get(c[0], {}).get(c[1]), dict) for r in fin.values())]
    if not have_cap:
        raise SystemExit(f"{src}: no MMLU, GSM8K or IFEval in any condition")
    # "MMLU+GSM8K+IFEval mean (capability)" is 35 characters and CLIPS at this figure width --
    # a panel title has to fit inside ~1.55in of drawing area, which at 7pt is about 30
    # characters. The two panels take parallel "what it measures: which probes" prefixes instead,
    # which fits and says the same thing.
    cap_label = "+".join(c[3] for c in have_cap)

    sr_se = sr_stderr(src)
    rows, dropped = [], []
    for cond, r in fin.items():
        f = frac_of(cond)
        sr = r.get("strongreject", {}).get("off_target", {}).get("score")
        caps = [r.get(e, {}).get(sp, {}).get(m) for e, sp, m, _ in have_cap]
        if f is None or sr is None or any(c is None for c in caps):
            dropped.append(cond)
            continue
        # SE of a mean of independent accuracies: root-sum-square of their SEs over the count
        ses = [r.get(e, {}).get(sp, {}).get("stderr") for e, sp, _, _ in have_cap]
        cap_se = (math.sqrt(sum(x * x for x in ses)) / len(ses) if all(x is not None for x in ses)
                  else float("nan"))
        rows.append(dict(cond=cond, frac=f, sr=sr, cap=sum(caps) / len(caps),
                         sr_se=sr_se.get(cond, float("nan")), cap_se=cap_se))
    df = pd.DataFrame(rows).sort_values("frac")
    end = {c: df[df.cond == c].iloc[0] for c in ("pretrained", "full_delta")}
    mid = df[~df.cond.isin(["pretrained", "full_delta", "frac_1"])].copy()   # frac_1 == full_delta
    mid["pct"] = mid["frac"] * 100

    # ---- reference lines. Instruct-native and the Base prompt ladder on safety; native
    # capability of both models on the capability panel. Labels sit in-panel above each line.
    F_SR, F_CAP = "Safety: StrongREJECT", f"Capability: {cap_label}"
    refs = [dict(facet=F_SR, model="Instruct", y=end["pretrained"].sr, label="Instruct")]
    p = RUNS / BASE_CEILING[1] / "evals.json"
    if p.exists():
        y = json.loads(p.read_text())["final"]["dense"]["strongreject"]["off_target"]["score"]
        refs.append(dict(facet=F_SR, model="Base", y=y, label=BASE_CEILING[0]))
    refs += [dict(facet=F_CAP, model="Instruct", y=end["pretrained"].cap, label="Instruct"),
             dict(facet=F_CAP, model="Base", y=end["full_delta"].cap, label="Base")]
    refs = pd.DataFrame(refs)
    # a label sits above its line unless that line is the highest thing in its panel, where
    # "above" is off the top edge -- then it goes below (the capability panel's Instruct anchor)
    top = {F_SR: max(mid.sr.max(), refs[refs.facet == F_SR].y.max()),
           F_CAP: max(mid.cap.max(), refs[refs.facet == F_CAP].y.max())}
    # ...and at the right edge unless the curve ENDS close to the line there (the capability
    # panel's Base anchor), in which case it moves to the left edge, clear of the data
    span = {F_SR: max(mid.sr.max(), refs[refs.facet == F_SR].y.max()) - mid.sr.min(),
            F_CAP: max(mid.cap.max(), refs[refs.facet == F_CAP].y.max()) - mid.cap.min()}
    last = {F_SR: mid.sr.iloc[-1], F_CAP: mid.cap.iloc[-1]}
    lab = pd.DataFrame([dict(facet=r.facet, y=r.y, model=r.model, label=r.label,
                             below=r.y >= top[r.facet],
                             right=abs(last[r.facet] - r.y) >= 0.12 * span[r.facet])
                        for r in refs.itertuples()])
    lab["x"] = lab.right.map({True: mid.pct.max(), False: mid.pct.min()})
    lab["ha"] = lab.right.map({True: "right", False: "left"})
    for d in (refs, lab):
        d["facet"] = pd.Categorical(d["facet"], [F_SR, F_CAP], ordered=True)

    long = pd.concat([mid.assign(facet=F_SR, y=mid.sr, se=mid.sr_se),
                      mid.assign(facet=F_CAP, y=mid.cap, se=mid.cap_se)])
    long["lo"], long["hi"] = long.y - long.se, long.y + long.se
    long["facet"] = pd.Categorical(long["facet"], [F_SR, F_CAP], ordered=True)
    ring = long[long.cond == "frac_0.01"]

    p = (
        ggplot(long, aes("pct", "y"))
        + facet_wrap("~facet", ncol=1, scales="free_y")
        + geom_hline(refs, aes(yintercept="y", color="model"), linetype="dashed", size=0.55,
                     inherit_aes=False)
        + geom_text(lab[~lab.below & lab.right], aes("x", "y", label="label", color="model"),
                    ha="right", va="bottom", size=5, inherit_aes=False)
        + geom_text(lab[~lab.below & ~lab.right], aes("x", "y", label="label", color="model"),
                    ha="left", va="bottom", size=5, inherit_aes=False)
        + geom_text(lab[lab.below & lab.right], aes("x", "y", label="label", color="model"),
                    ha="right", va="top", size=5, inherit_aes=False)
        + geom_text(lab[lab.below & ~lab.right], aes("x", "y", label="label", color="model"),
                    ha="left", va="top", size=5, inherit_aes=False)
        + geom_linerange(long.dropna(subset=["se"]), aes(ymin="lo", ymax="hi"), size=0.5,
                         color="#333333")
        + geom_line(size=0.7, color="#333333")
        + geom_point(size=1.7, color="#333333", stroke=0)
        + geom_point(ring, fill="none", color="#000000", size=3.4, stroke=0.7, shape="o")
        + scale_x_log10(breaks=[0.1, 1, 10, 100], labels=["0.1", "1", "10", "100"])
        + scale_color_manual(values=MODEL)
        + labs(x="Finetune parameters changed (%)", y="")
    )

    out = Path(args.out) if args.out else Path(__file__).parent / "refusal_sparsity_facets.pdf"
    p.save(out, verbose=False)
    if args.png:
        p.save(out.with_suffix(".png"), dpi=300, verbose=False)
    print(f"wrote {out}: run={args.run}, sweep={src.relative_to(RUNS)}, frame={frame}, "
          f"capability={cap_label}"
          + (f" (IFEval from {IFEVAL_SWEEP})" if unmatched is not None else "")
          + f", {len(mid)} sparsities; "
          f"Instruct SR={end['pretrained'].sr:.3f} cap={end['pretrained'].cap:.1f}, "
          f"Base SR={end['full_delta'].sr:.3f} cap={end['full_delta'].cap:.1f}")
    if unmatched:
        print(f"  conditions in only one of the two sweeps: {unmatched}")
    if dropped:
        print(f"  conditions dropped for a missing metric: {dropped}")
    print("  " + "  ".join(f"{r.cond}: SR {r.sr:.3f} cap {r.cap:.1f}" for r in df.itertuples()))


if __name__ == "__main__":
    main()
