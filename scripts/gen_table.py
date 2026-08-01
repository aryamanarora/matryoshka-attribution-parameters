"""One prompt's generations across a mask's sparsity sweep, as a self-contained HTML table.

Every other view in this repo reports a *rate*. This reports the text behind one: pick a prompt,
and read its response at every point of the sparsity grid, from the pretrained model to the whole
delta. It is the view that separates "the rate fell" from "the model broke", and it is how the
frac-0.01 cells in `plot_posthoc_curves.py` were checked to be a clean conditional policy rather
than an artifact.

Output is one ``<table>`` with inline styles and no external anything, so it pastes into a doc,
a notebook or an email intact::

    uv run python scripts/gen_table.py --dir <run> --organism language \\
        --col in_dist:52 --col off_target:34 --out table.html
    # macOS: put it on the clipboard as RENDERED html rather than as markup
    hex=$(python -c "print(open('table.html','rb').read().hex())")
    osascript -e "set the clipboard to «data HTML${hex}»"

``--dir`` is a directory holding ``config.yaml``, ``evals.json`` and ``generations.jsonl`` (the
last from the run's ``<eval>_eval/`` subdirectory), i.e. a pulled copy of one run.

THE HAZARD, and why this script verifies before it writes
---------------------------------------------------------
A record does not have to say which sparsity produced it. ``eval/runner.py``'s ``dump_records``
drains every condition's records in one go at each eval point, so for most evals the condition is
recoverable only from POSITION in the file -- and getting it wrong silently mislabels every row of
the table, which is worse than no table at all. Two further traps sit on top of that:

* An eval point can sweep the grid more than once (a scheduled pass and a final pass at the same
  step), and only the LAST pass is what ``evals.json`` reports.
* ``strongreject`` *does* store ``condition``, but its records still accumulate across eval points,
  so the last block per condition is still the right one.

So the reconstruction is followed by a check: recompute each condition's metric from the records
assigned to it and require it to equal ``evals.json``'s value exactly, for every condition and
split. A mismatch exits without writing. That check has already caught one real bug -- run-length
encoding the split column works when splits alternate per condition and produces one giant block
for a single-split eval like ``gsm8k``, which reported 13.4% against a reported 24.0%.

ADDING AN ORGANISM
------------------
One entry in :data:`ORGANISMS`: which eval and metric it reports, how to turn one record into a
0-1 value for the tint, what to print beside each response, and (optionally) a hue, a rate format
and a prompt trimmer. The per-record verdict is always a field the eval already stored -- never a
re-derivation of the metric, which would put a second implementation of it in a figure script.
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path

import yaml

#: ``{name: spec}``. ``value`` maps one record to 0-1 for the tint; ``rate`` names the metric in
#: ``evals.json`` that the reconstruction is verified against; ``agg`` says how that metric is
#: formed from the per-record values, which is what makes the check exact for a mean (StrongREJECT),
#: a threshold count (pirate) and a percentage (GSM8K) alike.
ORGANISMS = {
    "language": dict(
        eval="language", rate="target_frac", agg="rate",
        value=lambda r, cfg: float(r.get("langdetect") == cfg["eval"]["language"]["target"]),
        tags=lambda r, cfg: f"langdetect: <b>{html.escape(str(r.get('langdetect')))}</b>",
        note="Shaded = langdetect says the target language."),
    "casing": dict(
        eval="casing", rate=None, agg="rate",     # rate resolved per run from eval.casing.target
        value=lambda r, cfg: float(
            r.get("casing") == ((cfg.get("eval", {}).get("casing") or {}).get("target") or "lower")),
        tags=lambda r, cfg: f"casing: <b>{html.escape(str(r.get('casing')))}</b>",
        note="Shaded = the response is entirely in the trained casing."),
    "pirate": dict(
        eval="pirate", rate="pirate_frac", agg="threshold", threshold=0.5,
        value=lambda r, cfg: (r.get("pirate") or 0) / 100.0,
        tags=lambda r, cfg: (f"judge: <b>{r.get('pirate')}</b>/100 &nbsp;·&nbsp; coherent "
                             f"{r.get('coherent')} &nbsp;·&nbsp; markers {r.get('markers')}"),
        note=("Shade = the judge's 0-100 pirate-voice score; the split rate counts responses at or "
              "above the cutoff. <b>markers</b> is the API-free lexical check — the two agreeing is "
              "what licenses reading the score as a register change rather than judge drift.")),
    "strongreject": dict(
        eval="strongreject", rate="score", agg="mean",
        value=lambda r, cfg: float(r.get("score") or 0.0),
        tags=lambda r, cfg: f"judge: <b>{(r.get('score') or 0):.2f}</b>",
        note=("Shade = their fine-tuned judge's score for THIS response (0 = refused, 1 = fully "
              "assisted). This eval has no in-dist split — its control would need forbidden "
              "prompts from the training distribution, which nothing builds yet — so several "
              "columns are several forbidden prompts.")),
    "gsm8k": dict(
        eval="gsm8k", rate="accuracy", agg="percent",
        value=lambda r, cfg: float(bool(r.get("correct"))),
        # GREEN, not the red of the behaviour tables: here a shaded cell is the model getting the
        # answer RIGHT, and reusing red would put "capability retained" and "refusal removed" in
        # the same colour for a reader looking at both tables.
        hue=((234, 246, 234), (77, 175, 74)),
        rate_fmt="{:.1f}%",
        # the stored prompt carries the k-shot preamble; the header wants the question being asked
        prompt_fn=lambda p: re.findall(r"Question: (.+?)\nAnswer:", p, re.S)[-1].strip(),
        tags=lambda r, cfg: (f"pred <b>{r.get('pred')}</b> vs gold {r.get('gold')} &nbsp;·&nbsp; "
                             f"<b>{'correct' if r.get('correct') else 'wrong'}</b>"),
        note=("Shade = this response got the right answer (exact match on the number after the "
              "final <code>####</code>).")),
}

#: white -> red unless the organism overrides it; see the gsm8k entry for why green exists
DEFAULT_HUE = ((253, 234, 234), (228, 26, 28))


def rate_key(spec, cfg):
    """``casing`` measures two opposite organisms, so its headline is resolved per run."""
    if spec["rate"]:
        return spec["rate"]
    target = ((cfg.get("eval", {}).get("casing") or {}).get("target")) or "lower"
    return f"{target}_frac"


def aggregate(spec, recs, cfg):
    """The per-record values combined the way ``evals.json`` combines them."""
    vals = [spec["value"](r, cfg) for r in recs]
    how = spec["agg"]
    if how == "mean":
        return sum(vals) / len(vals)
    if how == "percent":
        return 100.0 * sum(vals) / len(vals)
    cut = spec.get("threshold", 1.0)
    return sum(v >= cut for v in vals) / len(vals)


def reconstruct(rows, conds):
    """``{condition: {split: [record]}}``, by stored label where there is one and by position where
    there is not. See the module docstring: this is the part that has to be verified."""
    step = max((r.get("step") or 0) for r in rows)
    final = [r for r in rows if (r.get("step") or 0) == step]
    splits = list(dict.fromkeys(r["split"] for r in final))
    has_cond = all("condition" in r for r in final)
    by = {c: {} for c in conds}
    for split in splits:
        recs = [r for r in final if r["split"] == split]
        n_p = len({r["prompt"] for r in recs})
        if has_cond:
            for c in conds:
                by[c][split] = [r for r in recs if r["condition"] == c][-n_p:]
            continue
        # chunk by THIS split's prompt count and keep the last len(conds) chunks: chunking per
        # split rather than run-length-encoding the split column is what makes a single-split eval
        # work, and taking the last chunks is what drops an earlier pass over the same grid
        chunks = [recs[i:i + n_p] for i in range(0, len(recs), n_p)][-len(conds):]
        if len(chunks) != len(conds):
            sys.exit(f"{split}: {len(recs)} records / {n_p} prompts gives {len(chunks)} blocks for "
                     f"{len(conds)} conditions -- cannot align")
        for c, ch in zip(conds, chunks):
            by[c][split] = ch
    return by, splits


def frac_label(cond):
    if cond == "pretrained":
        return "0 <span style='color:#888'>(base)</span>"
    if cond == "frac_1":
        return "1.0 <span style='color:#888'>(full delta)</span>"
    return cond.replace("frac_", "")


def mask_desc(cfg):
    """The mask's identity for the caption -- with the OBJECTIVE, not `mask.scores`'s default.

    A GRPO run leaves ``mask.scores`` at ``learned``, so printing it would claim the scores were
    fitted to the SFT loss when they were fitted to the reward: the entire difference between that
    run and a post-hoc one.
    """
    mk, rl = cfg.get("mask") or {}, cfg.get("rl") or {}
    bits = [f"{k}={mk[k]}" for k in ("unit", "variant", "mode", "k_schedule") if mk.get(k)]
    bits.append(f"scores=GRPO({rl['reward']} reward, {rl.get('steps')} steps)" if rl
                else f"scores={mk.get('scores', 'learned')}")
    if rl and mk.get("finetuned"):
        bits.append(f"delta={mk['finetuned']}")
    return " · ".join(bits)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True,
                   help="a pulled run directory: config.yaml, evals.json, generations.jsonl")
    p.add_argument("--organism", required=True, choices=sorted(ORGANISMS))
    p.add_argument("--col", action="append", required=True, metavar="SPLIT:INDEX",
                   help="one column, e.g. in_dist:52. Repeat per column; a single-split eval "
                        "(strongreject, gsm8k) takes several indices of the same split.")
    p.add_argument("--capability", nargs="*", default=(), metavar="LABEL:FILE:EVAL:SPLIT:KEY",
                   help="extra numeric columns joined BY CONDITION from another evals.json -- a "
                        "capability sweep over a mask is a separate `eval` invocation writing its "
                        "own directory, so it is not in this run's file")
    p.add_argument("--font", type=float, default=6.0, help="pt, for the response text")
    p.add_argument("--maxlen", type=int, default=420, help="characters per response before an ellipsis")
    p.add_argument("--out", default=None, help="default: <dir>/table.html")
    args = p.parse_args()

    spec, d = ORGANISMS[args.organism], Path(args.dir)
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    blob = json.loads((d / "evals.json").read_text())
    rows = [json.loads(l) for l in (d / "generations.jsonl").read_text().splitlines() if l.strip()]
    conds = [c for c in blob["final"] if c != "full_delta"]   # full_delta duplicates frac_1
    by, splits = reconstruct(rows, conds)

    ev, key = spec["eval"], rate_key(spec, cfg)
    for c in conds:
        for split, recs in by[c].items():
            got, want = aggregate(spec, recs, cfg), blob["final"][c][ev][split][key]
            if abs(got - want) > 1e-9:
                sys.exit(f"MISMATCH {c}/{split}: reconstructed {got} vs evals.json {want} -- the "
                         "records are not aligned to the conditions, so no table was written")
    print(f"verified: {len(conds)} conditions x {len(splits)} split(s) match evals.json exactly")

    F, cols = args.font, [(c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1])) for c in args.col]
    caps = [c.split(":") for c in args.capability]
    cap_vals = {}
    for label, fname, evname, split, k in caps:
        cb = json.loads((d / fname).read_text())
        cap_vals[label] = {c: (((cb["final"].get(c) or {}).get(evname) or {}).get(split) or {}).get(k)
                           for c in cb["final"]}
    trim = spec.get("prompt_fn", lambda s: s)
    lo_rgb, hi_rgb = spec.get("hue", DEFAULT_HUE)

    def cell(rec, rate):
        v = max(0.0, min(1.0, spec["value"](rec, cfg)))
        txt = rec["response"].strip()
        txt = txt[:args.maxlen] + " …" if len(txt) > args.maxlen else txt
        mix = lambda u: ",".join(f"{lo + (hi - lo) * u:.0f}" for lo, hi in zip(lo_rgb, hi_rgb))
        # the FILL is capped at 40% of the ramp while the bar takes it all: at 6pt a saturated
        # background makes the grey verdict line unreadable, and that line is the per-response
        # evidence the tint only summarises
        bg = f"rgb({mix(0.4 * v)})" if v > 0.02 else "#fff"
        bar = f"rgb({mix(v)})" if v > 0.02 else "#ddd"
        return (f"<td style='background:{bg};border-left:3px solid {bar};padding:4px 6px;"
                f"vertical-align:top;font-size:{F}pt;line-height:1.3'>"
                f"<div style='color:#444;font-size:{F - 1}pt;margin-bottom:2px'>"
                f"{spec['tags'](rec, cfg)} &nbsp;·&nbsp; split rate "
                f"{spec.get('rate_fmt', '{:.2f}').format(rate)}</div>{html.escape(txt)}</td>")

    out = [
        "<table cellspacing='0' cellpadding='0' style=\"border-collapse:collapse;font-family:"
        "-apple-system,Segoe UI,Inter,Helvetica,Arial,sans-serif;max-width:1100px\">",
        f"<caption style='caption-side:top;text-align:left;font-size:{F + 1}pt;color:#333;"
        f"padding:0 0 6px'><b>{html.escape(cfg['name'])}</b> &nbsp;·&nbsp; mask: "
        f"{html.escape(mask_desc(cfg))} &nbsp;·&nbsp; {html.escape(cfg['model'])}<br>"
        f"<span style='color:#666'>One prompt per column across the sparsity sweep. <b>frac</b> is "
        f"the fraction of units whose finetuned delta is kept; 0 is the pretrained model, 1.0 the "
        f"whole delta. {spec['note']} <i>split rate</i> is that condition's <code>{key}</code> over "
        f"the whole split — the response beside it is one example, not the measurement."
        f"</span></caption><thead><tr>"
        f"<th style='text-align:right;padding:4px 8px 4px 0;font-size:{F}pt;color:#444;"
        f"border-bottom:1px solid #ccc;white-space:nowrap'>frac</th>",
    ]
    for split, i in cols:
        lab = {"in_dist": "in-dist", "off_target": "off-target"}.get(split, split)
        out.append(f"<th style='text-align:left;padding:4px 6px;font-size:{F}pt;"
                   f"border-bottom:1px solid #ccc;width:{100 // len(cols)}%'>{lab}<br>"
                   f"<span style='font-weight:400;color:#333'>"
                   f"{html.escape(trim(by[conds[0]][split][i]['prompt']))}</span></th>")
    for label, *_ in caps:
        out.append(f"<th style='text-align:right;padding:4px 6px;font-size:{F}pt;"
                   f"border-bottom:1px solid #ccc;white-space:nowrap'>{html.escape(label)}</th>")
    out.append("</tr></thead><tbody>")
    for c in conds:
        out.append(
            f"<tr><td style='text-align:right;padding:4px 8px 4px 0;font-size:{F}pt;"
            f"vertical-align:top;font-variant-numeric:tabular-nums;white-space:nowrap'>"
            f"{frac_label(c)}</td>"
            + "".join(cell(by[c][s][i], blob["final"][c][ev][s][key]) for s, i in cols)
            + "".join(
                f"<td style='text-align:right;padding:4px 6px;font-size:{F}pt;vertical-align:top;"
                f"font-variant-numeric:tabular-nums;"
                f"color:{'#111' if cap_vals[label].get(c) is not None else '#999'}'>"
                f"{('%.1f' % cap_vals[label][c]) if cap_vals[label].get(c) is not None else '—'}"
                f"</td>" for label, *_ in caps)
            + "</tr>")
    out.append("</tbody></table>")

    dest = Path(args.out or (d / "table.html"))
    dest.write_text("\n".join(out))
    print(f"wrote {dest}")
    for split, i in cols:
        print(f"  {split:11} [{i}] {trim(by[conds[0]][split][i]['prompt'])[:90]}")


if __name__ == "__main__":
    main()
