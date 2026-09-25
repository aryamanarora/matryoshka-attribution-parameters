"""Read every OlmPool attribution run into one table, and the head-level views behind it.

Per run (``runs/olmpool/<model>/<arm>/``): the score vector and its layout from ``final.pt``, the
sparsity curves from ``evals.json``, the delta diagnostics from ``delta_stats.json``. A flat unit
index is turned into ``(layer, kind, index)`` through the layout -- under ``unit: head`` a kind is
``q`` (a query head: its q rows and o columns), ``kv`` (a kv group: its k and v rows), ``mlp`` (a
neuron) or ``other`` -- so a run's ranking becomes a [layers x heads] matrix that can be laid
beside the retrieval-score matrices from ``scripts/olmpool/olmpool_retrieval_heads.py``.

Writes ``<out>/summary.json`` (one record per run: curves, AUCs, per-layer score mass, the
top-k head list, overlap with retrieval heads, Spearman against |delta|) and ``<out>/heads/`` (a
per-run JSON with the full head matrices), which the plots read.

    uv run python scripts/olmpool/olmpool_analysis.py --runs runs/olmpool --rh runs/olmpool_rh \\
        --results data/olmpool/olmpool_results.json --out plots/data/olmpool
"""
import argparse
import json
import math
import re
from pathlib import Path

import torch

from mask_learning_finetuning.masks import group_of
from mask_learning_finetuning.masks.checkpoint import layout_from_blob

_LAYER = re.compile(r"layers\.(\d+)\.")


def unit_table(layout):
    """One row per DISTINCT unit: ``{idx: (layer, kind, index_within_kind)}``."""
    rows = {}
    for i, (name, off, cnt, axis) in enumerate(zip(layout.names, layout.offsets, layout.counts,
                                                     layout.axes)):
        m = _LAYER.search(name)
        layer = int(m.group(1)) if m else -1
        if group_of(axis) is not None and ("q_proj" in name or "o_proj" in name):
            kind = "q"
        elif group_of(axis) is not None and ("k_proj" in name or "v_proj" in name):
            kind = "kv"
        elif "mlp" in name or "feed_forward" in name:
            kind = "mlp"
        elif "embed_tokens" in name:
            kind = "embed"           # one unit per vocabulary row
        elif "lm_head" in name:
            kind = "lm_head"         # one unit per vocabulary row
        elif "q_norm" in name or "k_norm" in name:
            kind = "qk_norm"         # one unit per gain vector
        elif "norm" in name:
            kind = "norm"            # layer norms and the final norm, one unit each
        else:
            kind = "other"
        for j in range(cnt):
            u = off + j
            if u not in rows:
                rows[u] = (layer, kind, j)
    return rows


def load_run(run_dir: Path):
    blob = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    layout = layout_from_blob(blob)
    scores = blob["scores"].float()
    ev = json.loads((run_dir / "evals.json").read_text())
    ds = json.loads((run_dir / "delta_stats.json").read_text()) if (run_dir / "delta_stats.json").exists() else {}
    return blob["args"], layout, scores, ev, ds


def head_matrix(layout, scores, kind, n_layers, n_units, ut=None):
    """[layers x units] of scores for one unit kind; NaN where a layer has no such unit."""
    M = torch.full((n_layers, n_units), float("nan"))
    for u, (layer, k, j) in (ut or unit_table(layout)).items():
        if k == kind and layer >= 0 and j < n_units:
            M[layer, j] = scores[u]
    return M


def curves(final: dict):
    """``{eval/split/metric: {condition: value}}`` from the final sweep."""
    out = {}
    for cond, per_eval in final.items():
        for evn, per_split in per_eval.items():
            for split, metrics in per_split.items():
                for m, v in metrics.items():
                    if isinstance(v, (int, float)) and m != "n":
                        out.setdefault(f"{evn}/{split}/{m}", {})[cond] = v
    return out


def log_auc(curve: dict, fracs):
    xs = sorted(f for f in fracs if f > 0)
    ys = [curve.get(f"frac_{f:g}") for f in xs]
    if any(y is None for y in ys) or len(xs) < 2:
        return None
    lx = [math.log(x) for x in xs]
    return sum((lx[i + 1] - lx[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1)) / (lx[-1] - lx[0])


def summarise(run_dir: Path, rh_dir: Path, results: dict):
    args, layout, scores, ev, ds = load_run(run_dir)
    model = run_dir.parent.name
    arm = run_dir.name
    ut = unit_table(layout)          # once: 460K entries for an all-units run
    n_layers = 1 + max(l for l, _, _ in ut.values())
    kinds = {}
    for u, (layer, k, j) in ut.items():
        kinds.setdefault(k, []).append(u)
    n_q = 1 + max(j for l, k, j in ut.values() if k == "q")
    n_kv = 1 + max((j for l, k, j in ut.values() if k == "kv"), default=-1)
    Q = head_matrix(layout, scores, "q", n_layers, n_q, ut)
    KV = head_matrix(layout, scores, "kv", n_layers, n_kv, ut) if n_kv > 0 else None
    # rank of every q-head unit among q-head units (0 = highest score)
    q_units = torch.tensor(kinds["q"])
    q_scores = scores[q_units]
    order = q_scores.argsort(descending=True)
    rank = torch.empty_like(order); rank[order] = torch.arange(len(order))
    top = {}
    for k in (16, 32, 64, 128):
        sel = q_units[order[:k]]
        top[k] = [(ut[int(u)][0], ut[int(u)][2]) for u in sel]
    # how the top of the whole ranking splits between heads and neurons (all_* arms): the share
    # of attention units among the top 0.5% / 1% / 5% of units, and the fraction of all heads
    # that is -- the cross-architecture "is the extension head-carried or MLP-carried" number
    ut_all = ut
    order = scores.argsort(descending=True).tolist()
    n_heads_all = sum(1 for u, (l, k, j) in ut_all.items() if k in ("q", "kv"))
    rec_split = {}
    for frac in (0.001, 0.005, 0.01, 0.05, 0.1):
        k = max(1, int(round(frac * layout.total)))
        na = sum(1 for u in order[:k] if ut_all[u][1] in ("q", "kv"))
        rec_split[str(frac)] = {"k": k, "attention_units": na, "mlp_units": k - na,
                                "frac_of_heads": na / max(1, n_heads_all)}
    # per-layer share of the top-k q heads, and the layer-mean score
    layer_top64 = [sum(1 for l, j in top[64] if l == L) for L in range(n_layers)]
    layer_mean = [float(Q[L][~Q[L].isnan()].mean()) for L in range(n_layers)]
    rec = {
        "model": model, "arm": arm, "n_layers": n_layers, "n_q": n_q, "n_kv": n_kv,
        "total_units": layout.total, "kinds": {k: len(v) for k, v in kinds.items()},
        "delta_norm": ds.get("delta_norm"), "spearman_delta": ds.get("spearman_scores_vs_delta_norm"),
        "dead_units": ds.get("dead_units"),
        "curves": curves(ev["final"]), "fracs": ev["meta"].get("fracs"),
        "top_q_heads": {str(k): v for k, v in top.items()},
        "top_split": rec_split,
        "layer_top64": layer_top64, "layer_mean_q": layer_mean,
        "results": results.get(model),
    }
    fr = rec["fracs"] or [c for c in ev["final"] if c.startswith("frac_")]
    fr = [float(str(f).replace("frac_", "")) for f in fr]
    rec["auc"] = {k: log_auc(v, fr) for k, v in rec["curves"].items()}
    # per retrieval curve: the best sparse point against the full delta (overshoot), and the
    # smallest fraction at which the curve reaches 90% of its peak -- the two numbers the
    # cross-architecture comparison is about
    rec["peak"] = {}
    for k, v in rec["curves"].items():
        if not k.startswith("niah/") or not k.endswith("/acc"):
            continue
        pts = {float(c[5:]): y for c, y in v.items() if c.startswith("frac_")}
        if not pts:
            continue
        peak = max(pts.values())
        rec["peak"][k] = {"peak": peak, "peak_frac": min(f for f, y in pts.items() if y == peak),
                          "full": v.get("full_delta"), "pretrained": v.get("pretrained"),
                          "overshoot": peak - (v.get("full_delta") or 0.0),
                          "frac_90": min((f for f, y in sorted(pts.items()) if y >= 0.9 * peak),
                                         default=None)}
    # retrieval-head overlap: the attention-fitted ranking against Wu et al.'s score at the
    # long-context checkpoint (and at the extended-theta pretrained one)
    rec["retrieval"] = {}
    for ck in ("lc", "pt_ext", "pt"):
        f = rh_dir / f"{model}__{ck}.json"
        if not f.exists():
            continue
        rh = json.loads(f.read_text())
        for Lc, d in rh["lengths"].items():
            S = torch.tensor(d["score_success"])          # [layers x heads]
            if S.shape != Q.shape:
                continue
            flatS, flatQ = S.flatten(), Q.flatten()
            ok = ~flatQ.isnan()
            rs = _spearman(flatS[ok], flatQ[ok])
            rh_heads = set(int(i) for i in (flatS >= 0.1).nonzero().flatten())
            ov = {}
            for k in (16, 32, 64):
                topk = set(int(i) for i in flatQ.nan_to_num(-1e9).topk(k).indices)
                ov[str(k)] = {"n_rh": len(rh_heads), "overlap": len(topk & rh_heads),
                              "expected": len(rh_heads) * k / int(ok.sum())}
            rec["retrieval"][f"{ck}@{Lc}"] = {"spearman": rs, "acc": d["acc"], "overlap": ov,
                                             "n_rh_heads": len(rh_heads)}
    # attention-statistics change under extension (scripts/olmpool/olmpool_head_stats.py): does the mask
    # rank first the heads whose sink / entropy / distance / needle mass moved most from pt_ext
    # to lc? Reported as Spearman(score, |change|) and as the top-64 heads' mean change against
    # the rest, per statistic and length.
    rec["head_stats"] = {}
    hs_dir = rh_dir.parent / "olmpool_hs"
    fa, fb = hs_dir / f"{model}__pt_ext.json", hs_dir / f"{model}__lc.json"
    if fa.exists() and fb.exists():
        A, Bj = json.loads(fa.read_text()), json.loads(fb.read_text())
        top64 = set(int(i) for i in Q.flatten().nan_to_num(-1e9).topk(64).indices)
        for Lc in A["lengths"]:
            if Lc not in Bj["lengths"]:
                continue
            for st in ("sink", "entropy", "distance", "needle"):
                a_, b_ = torch.tensor(A["lengths"][Lc][st]), torch.tensor(Bj["lengths"][Lc][st])
                if a_.shape != Q.shape:
                    continue
                d = (b_ - a_).flatten()
                ok = ~Q.flatten().isnan()
                flatQ = Q.flatten()
                idx_top = torch.tensor(sorted(top64))
                rest = torch.tensor([i for i in range(len(d)) if i not in top64 and ok[i]])
                rec["head_stats"][f"{st}@{Lc}"] = {
                    "spearman_abs_change": _spearman(flatQ[ok], d.abs()[ok]),
                    "spearman_change": _spearman(flatQ[ok], d[ok]),
                    "top64_mean_change": float(d[idx_top].mean()),
                    "rest_mean_change": float(d[rest].mean()),
                    "top64_mean_lc": float(b_.flatten()[idx_top].mean()),
                    "rest_mean_lc": float(b_.flatten()[rest].mean()),
                    "top64_mean_pt": float(a_.flatten()[idx_top].mean()),
                    "rest_mean_pt": float(a_.flatten()[rest].mean()),
                }
    heads = {"model": model, "arm": arm, "Q": Q.tolist(), "KV": KV.tolist() if KV is not None else None}
    return rec, heads


def _spearman(a, b):
    def ranks(x):
        idx = x.argsort(); r = torch.zeros_like(x, dtype=torch.float64); r[idx] = torch.arange(len(x), dtype=torch.float64); return r
    ra, rb = ranks(a.double()), ranks(b.double())
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = ra.norm() * rb.norm()
    return float(ra @ rb / d) if float(d) else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/olmpool")
    ap.add_argument("--rh", default="runs/olmpool_rh")
    ap.add_argument("--results", default="data/olmpool/olmpool_results.json")
    ap.add_argument("--out", default="plots/data/olmpool")
    a = ap.parse_args()
    results = json.loads(Path(a.results).read_text()) if Path(a.results).exists() else {}
    out = Path(a.out); (out / "heads").mkdir(parents=True, exist_ok=True)
    recs = []
    for f in sorted(Path(a.runs).glob("*/*/final.pt")):
        run_dir = f.parent
        if not (run_dir / "evals.json").exists():
            print("skip (no evals.json):", run_dir); continue
        try:
            rec, heads = summarise(run_dir, Path(a.rh), results)
        except Exception as e:
            print("FAILED", run_dir, repr(e)[:200]); continue
        recs.append(rec)
        (out / "heads" / f"{rec['model']}__{rec['arm']}.json").write_text(json.dumps(heads))
        h = rec["results"]["helmet"]["32k"] if rec["results"] else float("nan")
        aucs = {k.split("/")[1]: round(v, 3) for k, v in rec["auc"].items() if k.startswith("niah/") and k.endswith("/acc") and v is not None}
        print(f"{rec['model']:34s} {rec['arm']:14s} HELMET32={h:5.1f} rho(delta)={rec['spearman_delta'] if rec['spearman_delta'] is None else round(rec['spearman_delta'],3)} "
              f"acc-AUC={aucs} top64 by layer={rec['layer_top64']} rh={ {k: round(v['spearman'],2) for k, v in rec['retrieval'].items()} }")
    (out / "summary.json").write_text(json.dumps(recs, indent=1))
    print(f"{len(recs)} runs -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
