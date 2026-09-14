"""Train loss OR the behaviour rate against mask sparsity, one panel per organism, overlaid.

    uv run python plots/plot_train_loss_curves.py                      # train loss
    uv run python plots/plot_train_loss_curves.py --what behaviour     # off-target headline

One script for both because they are the same figure over a different cell of `evals.json`, and
the pair is how the result reads: SGD's curve is on top in most behaviour panels while IG's is on
top in most loss panels -- the dissociation, shown as two figures a reader can hold side by side.

The companion to `plot_loss_auc_ranking.py`: that collapses each of these curves to one number, and
a single AUC provably cannot distinguish a curve that falls early from one that falls late to the
same area. This is the curve the AUC summarises, for every delta we have attributed.

PANELS ARE KEYED BY THE DELTA, NOT BY THE RUN NAME. Every run's `mask.finetuned` says which
finetune it attributes, so grouping on `(finetuned, unit)` is what makes a panel a controlled
comparison: the lines in one panel are different SCORINGS of one frozen delta over one layout, and
nothing else varies. Grouping on a name prefix would have been the obvious alternative and is
wrong here -- the fr2de/Qwen cell's stepless-IG run is `..._lr1e-4_ixg_mc` while its learned twin
is `..._lr1e-4_posthoc_shard`, so a prefix rule either splits that panel in two or needs a
hardcoded exception. It also silently protects against the `unit` trap: an `svd` cell attributing
the SAME adapter is a different x axis (72 singular directions where `nonresid` has 26k rows), so
it gets its own panel rather than a line in someone else's.

WHY THE LOSS AND NOT THE BEHAVIOUR RATE. Measured on the fr2de/Qwen sweep, which contains a
genuine replicate (two runs of an identical mask config): they differ by 0.002 on the loss and
0.019 on the off-target rate, so the loss resolves ~10x finer. The rate is also thresholded -- it
counts responses whose argmax crossed over, so it sits pinned at 0.00 while the mask is improving
underneath, which is exactly the sparse end these panels are for.

EACH PANEL HAS ITS OWN Y SCALE, deliberately. Train loss in nats is not commensurable across
organisms (0.67 on Qwen caps against 1.6 on Qwen fr2zh -- different data, different tokenizers,
different amounts of it), so a shared y would compress every panel to fit the worst one and the
comparison inside a panel is the only one being made. The dashed rule in each panel is that
delta's own `frac_1` anchor, i.e. the finetune's loss -- the level a mask would reach if it
recovered everything -- so the panels stay readable against a common MEANING even though they do
not share a scale.
"""

import argparse
import glob as globmod
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import (COLOR, FS_LABEL, FS_LEGEND, FS_TICK, LS, RC, furnish,  # noqa: E402
                     top_legend)

#: (key, label, colour, linestyle). The three arms every cell was run with; `--schedules` adds the
#: other two SGD k-schedules, which are within the replicate floor of `log_both` on 8 of 11 cells.
METHODS = [
    ("adam_default", "MAttr (Adam, default)", COLOR["adam"], "solid"),
    ("sgd_log_both", "MAttr + SGD (log_both)", COLOR["sgd"], "solid"),
    ("ixg_mc", "stepless IG (MC)", COLOR["ixg:mc"], "solid"),
]
EXTRA = [("sgd_log", "MAttr + SGD (log)", COLOR["sgd"], LS["log"]),
         ("sgd_logit", "MAttr + SGD (logit)", COLOR["sgd"], LS["logit"])]

#: each organism's off-target headline: eval, split, metric
BEH = {"fr2de": ("language", "off_target", "target_frac"),
       "fr2ru": ("language", "off_target", "target_frac"),
       "fr2zh": ("language", "off_target", "target_frac"),
       "lower": ("casing", "off_target", "lower_frac"),
       "caps": ("casing", "off_target", "upper_frac"),
       "spelling": ("spelling", "off_target", "british_frac"),
       "bad": ("em_fast", "off_target", "misaligned_frac"),
       "financial": ("em_fast", "off_target", "misaligned_frac")}

#: how a delta's path becomes a panel title
MODELS = (("qwen25_14b", "Qwen-14B"), ("gemma2_9b", "Gemma-9B"), ("olmo3_7b", "OLMo-7B"),
          ("sweep8b", "Llama-8B"))


def classify(mk):
    """Which METHODS key this run is, or None if it is not one of the compared arms."""
    sc = mk.get("scores", "learned")
    if sc == "ixg":
        return "ixg_mc" if mk.get("ixg_at") == "mc" else None
    if sc == "random":
        return None
    opt = mk.get("score_optimizer", "adam")
    ks = mk.get("k_schedule", "uniform")
    if opt == "sgd" and float(mk.get("score_lr", 0)) == 10:
        return {"log_both": "sgd_log_both", "log": "sgd_log", "logit": "sgd_logit"}.get(ks)
    # The DEFAULT Adam cell, which is what every organism outside fr2de was originally fitted at:
    # adam / lr 0.05 / uniform. Other Adam LRs exist only on fr2de and would put four unlabelled
    # blue lines in that one panel.
    if opt == "adam" and abs(float(mk.get("score_lr", 0)) - 0.05) < 1e-9 and ks == "uniform":
        return "adam_default"
    return None


def title_for(finetuned):
    """``organism · Model`` from the attributed finetune's path."""
    stem = Path(finetuned).parts[1] if finetuned.startswith("runs/") else Path(finetuned).name
    model = next((v for k, v in MODELS if k in stem), "?")
    org = stem.split("_")[0]
    if "financial" in stem:
        org = "financial"
    return f"{org} · {model}"


def collect(pattern, want, what="loss"):
    """``{(finetuned, unit): {method: (xs, ys), "_anchor": full_delta_y, "_title": str}}``."""
    cells = {}
    for d in sorted(globmod.glob(pattern)):
        d = Path(d)
        if not ((d / "evals.json").exists() and (d / "config.yaml").exists()):
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        mk = cfg.get("mask") or {}
        if not mk or not mk.get("finetuned"):
            continue
        who = classify(mk)
        if who not in want:
            continue
        title = title_for(mk["finetuned"])
        if what == "behaviour":
            key3 = BEH.get(title.split(" ")[0])
            if key3 is None:
                continue
            ev, split, metric = key3
        else:
            ev, split, metric = "sft_loss", "train", "loss"
        final = json.loads((d / "evals.json").read_text())["final"]
        pts = []
        for cond, v in final.items():
            if not cond.startswith("frac_"):
                continue
            y = ((v.get(ev) or {}).get(split) or {}).get(metric)
            if y is not None:
                pts.append((float(cond.replace("frac_", "")), y))
        if len(pts) < 2:
            continue
        pts.sort()
        key = (mk["finetuned"], mk.get("unit", "nonresid"))
        cell = cells.setdefault(key, {"_title": title, "_anchor": None})
        cell[who] = ([p[0] for p in pts], [p[1] for p in pts])
        cell["_anchor"] = dict(pts).get(1.0, cell["_anchor"])
    return cells


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default="runs/*")
    p.add_argument("--what", default="loss", choices=("loss", "behaviour"))
    p.add_argument("--schedules", action="store_true", help="also draw SGD log and logit")
    p.add_argument("--ncol", type=int, default=4)
    p.add_argument("--out", default=None,
                   help="default: plots/train_loss_curves.pdf / plots/behaviour_curves.pdf")
    a = p.parse_args()
    a.out = a.out or ("plots/behaviour_curves.pdf" if a.what == "behaviour"
                      else "plots/train_loss_curves.pdf")
    methods = METHODS + (EXTRA if a.schedules else [])
    want = {m[0] for m in methods}

    cells = collect(a.glob, want, a.what)
    # Panels with only one arm cannot be a COMPARISON -- they are the half-finished cells of a
    # sweep still draining, and drawing them invites reading a lone line as a result. Dropped
    # loudly, with the arms they are waiting on named.
    full = {}
    for k, c in cells.items():
        have = [m for m in want if m in c]
        if len(have) >= 2:
            full[k] = c
        else:
            print(f"  skipped {c['_title']}: only {have or 'no'} arm(s) present", file=sys.stderr)
    if not full:
        raise SystemExit("no cell has two comparable arms yet")
    order = sorted(full, key=lambda k: full[k]["_title"])

    plt.rcParams.update(RC)
    n, ncol = len(order), a.ncol
    nrow = -(-n // ncol)
    PANEL_W, ROW_H = 1.35, 1.45
    fig, axes = plt.subplots(nrow, ncol, figsize=(PANEL_W * ncol, ROW_H * nrow), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.set_visible(False)
    for i, (key, ax) in enumerate(zip(order, flat)):
        c = full[key]
        if c["_anchor"] is not None:
            ax.axhline(c["_anchor"], lw=0.5, color="#666666", ls=(0, (3, 2)), zorder=1)
        for who, _lab, col, ls in methods:
            if who not in c:
                continue
            xs, ys = c[who]
            ax.plot(xs, ys, color=col, ls=ls, lw=1.0, solid_joinstyle="round", zorder=3)
        ax.set_xscale("log")
        if a.what == "behaviour":
            ax.set_ylim(-0.04, 1.04)         # rates share a scale; nats do not (see docstring)
        ax.set_title(c["_title"], fontsize=FS_LABEL - 1, pad=2.5)
        ax.tick_params(labelsize=FS_TICK - 1, length=2, width=0.5)
        furnish(ax)
        if i % ncol == 0:
            ax.set_ylabel("off-target rate" if a.what == "behaviour" else "train loss (nats)",
                          fontsize=FS_LABEL - 1)
        if i // ncol == nrow - 1:
            ax.set_xlabel("fraction of units kept", fontsize=FS_LABEL - 1)

    handles = [Line2D([], [], color=col, ls=ls, lw=1.0, label=lab) for _, lab, col, ls in methods]
    handles.append(Line2D([], [], color="#666666", lw=0.5, ls=(0, (3, 2)),
                          label="full delta (the whole finetune)"))
    fig.tight_layout(w_pad=0.5, h_pad=0.6)
    top_legend(fig, handles, nrow, ROW_H, ncol=min(4, len(handles)))
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out} ({n} cells, {len(methods)} methods)")


if __name__ == "__main__":
    main()
