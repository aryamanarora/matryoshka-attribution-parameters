"""Closed-form (IxG) rankings of one post-training delta under MANY objectives, in one process.

The post-training attribution question (configs/olmo3_post/base.yaml) needs a ranking of the
DPO->RL delta's units per benchmark, plus the split-half ranking of each benchmark for the
within-benchmark ceiling. Through the config path that is one job per (objective, half, endpoint)
-- each of which loads two 7B checkpoints and builds the delta before doing ~64 forward/backward
passes. Here the model and delta are built ONCE and every objective's IxG score vector is computed
in turn, each written as a normal masked checkpoint (``<out>/<objective>_<at>/final.pt``) that
``python -m mask_learning_finetuning.eval`` can sweep exactly like a config-path IxG run.

    uv run python scripts/bench_ixg.py configs/olmo3_post/base.yaml \
        --objectives gsm8k math ifeval mmlu --halves --at base finetuned --out runs/olmo3_post/ixg

Everything numeric is the repo's own code: ``train.ixg.ixg_scores`` over ``MaskedDelta``'s delta and
layout, the SFT dataset's tokenisation and loss masking, ``token_weighted_ce``. The only thing this
script adds is the loop.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logger = logging.getLogger("bench_ixg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--objectives", nargs="+", default=["gsm8k", "math", "ifeval", "mmlu"])
    ap.add_argument("--halves", action="store_true", help="also the _a/_b halves of each")
    ap.add_argument("--at", nargs="+", default=["base", "finetuned"])
    ap.add_argument("--batches", type=int, default=None, help="override mask.ixg_batches")
    ap.add_argument("--data-dir", default="data/bench")
    ap.add_argument("--out", default="runs/olmo3_post/ixg")
    ap.add_argument("--wait", type=int, default=0,
                    help="minutes to wait for a missing objective file (the rollout job may "
                         "still be writing it) before skipping")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    import torch
    from mask_learning_finetuning.config import load_config
    from mask_learning_finetuning.train import params as params_mod
    from mask_learning_finetuning.train.ixg import ixg_scores
    from mask_learning_finetuning.train.loop import build_data, load_model
    from mask_learning_finetuning.train.params import token_weighted_ce
    from mask_learning_finetuning.train import posthoc

    cfg = load_config(args.config)
    cfg.mask.scores = "ixg"
    if args.batches:
        cfg.mask.ixg_batches = args.batches
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, tokenizer = load_model(cfg)
    P = params_mod.build(model, cfg)
    logger.info("model + delta ready in %.0fs; %s", time.time() - t0, P.layout.summary())
    norms = posthoc.unit_delta_norms(P.deltas, P.layout, getattr(P, "svd", None))
    torch.save(norms.cpu(), out / "unit_delta_norms.pt")
    (out / "delta_stats.json").write_text(json.dumps(P.provenance, indent=2, default=str))

    names = []
    for o in args.objectives:
        names.append(o)
        if args.halves:
            names += [f"{o}_a", f"{o}_b"]

    for name in names:
        path = ROOT / args.data_dir / f"{name}.jsonl"
        base_name = name.rsplit("_", 1)[0] if name.endswith(("_a", "_b")) else name
        meta = ROOT / args.data_dir / f"{base_name}.meta.json"     # written after the jsonl
        waited = 0
        while not (path.exists() and meta.exists()) and waited < args.wait:
            logger.info("waiting for %s (%d/%d min)", path, waited, args.wait)
            time.sleep(60)
            waited += 1
        if not path.exists() or path.stat().st_size == 0:
            logger.warning("skipping %s: %s is missing or empty (a rollout pass that kept nothing)",
                           name, path)
            continue
        cfg.data.train = str(path.relative_to(ROOT))
        # MMLU's target is one letter: supervise it alone, not the end-of-turn tail after it
        cfg.data.supervise_tail = not name.startswith("mmlu")
        _, loaders, _ = build_data(cfg, tokenizer)
        for at in args.at:
            dest = out / f"{name}_{at}"
            if (dest / "final.pt").exists():
                logger.info("%s exists, skipping", dest)
                continue
            t1 = time.time()

            def batches():
                for i, b in enumerate(loaders["train"]):
                    if i >= cfg.mask.ixg_batches:
                        return
                    yield {k: v.to(cfg.device) for k, v in b.items()}

            scores, stats = ixg_scores(
                model, base=P.base, deltas=P.deltas, layout=P.layout, batches=batches(),
                loss_fn=lambda m, b: token_weighted_ce(
                    m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]), b),
                at=at, out_dtype=P.compose_dtype, svd=P.svd)
            with torch.no_grad():
                P.scores.copy_(scores.to(P.scores.device))
            cfg.mask.ixg_at = at
            cfg.name = f"{cfg.name.rsplit('_ixg_', 1)[0]}_ixg_{name}_{at}"
            cfg.output = str(dest)
            dest.mkdir(parents=True, exist_ok=True)
            P.save(dest / "final.pt", tokenizer, train_log=[], final=True)
            rho = posthoc.spearman(scores, norms)
            stats.update(dict(P.provenance, spearman_scores_vs_delta_norm=rho,
                              objective=name, ixg_at=at, seconds=time.time() - t1))
            (dest / "delta_stats.json").write_text(json.dumps(stats, indent=2, default=str))
            from mask_learning_finetuning.config import dump
            dump(cfg, dest / "config.yaml")
            logger.info("%s @ %s: mean loss %.4f over %d tokens, spearman vs |delta| %.3f, %.0fs",
                        name, at, stats["ixg_mean_loss"], stats["ixg_tokens"], rho,
                        stats["seconds"])
    logger.info("all done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
