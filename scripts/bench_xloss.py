"""The LOSS transfer matrix: objective B's held-out NLL under the mask fitted on objective A.

The behavioural transfer matrix (scripts/bench_transfer.py) needs the two anchors to differ, and
on the DPO->RL delta they do not; the held-out likelihood of each benchmark's rollouts does move,
so this is the readout that has range there. One process: model + delta built once (as
scripts/bench_ixg.py does), then for every mask checkpoint under --masks and every objective file,
the runner's own `sft_loss` sweep over the sparsity grid -- `MaskedWeights` + `sweep`, exactly what
the training loop and the eval CLI call -- with the scores swapped in. Writes
`<out>/<mask>__<objective>.json` (the sweep result) and a `matrix.json` summary.

    uv run python scripts/bench_xloss.py configs/olmo3_post/base.yaml \
        --masks runs/olmo3_post/posthoc/{gsm8k,math,ifeval,mmlu} --objectives gsm8k math ifeval mmlu
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
logger = logging.getLogger("bench_xloss")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--masks", nargs="+", required=True, help="run dirs holding final.pt")
    ap.add_argument("--objectives", nargs="+", default=["gsm8k", "math", "ifeval", "mmlu"])
    ap.add_argument("--data-dir", default="data/bench")
    ap.add_argument("--n-batches", type=int, default=64)
    ap.add_argument("--out", default="runs/olmo3_post/xloss")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    import torch
    from mask_learning_finetuning.config import load_config
    from mask_learning_finetuning.eval import get_eval
    from mask_learning_finetuning.eval.runner import sweep
    from mask_learning_finetuning.train import params as params_mod
    from mask_learning_finetuning.train.loop import build_data, load_model

    cfg = load_config(args.config)
    cfg.eval.vllm = None
    cfg.eval.sft_loss.n_batches = args.n_batches
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tokenizer = load_model(cfg)
    P = params_mod.build(model, cfg)
    weights = P.eval_weights(tokenizer)
    logger.info("model + delta ready in %.0fs; fracs %s", time.time() - t0, weights.fracs)

    loaders = {}
    for obj in args.objectives:
        cfg.data.train = f"{args.data_dir}/{obj}.jsonl"
        cfg.data.supervise_tail = not obj.startswith("mmlu")
        _, loaders[obj], _ = build_data(cfg, tokenizer)

    matrix = {}
    for mdir in args.masks:
        mdir = ROOT / mdir
        blob = torch.load(mdir / "final.pt", map_location="cpu", weights_only=False)
        if blob["layout"]["total"] != P.layout.total:
            raise SystemExit(f"{mdir}: {blob['layout']['total']} units, this layout has {P.layout.total}")
        with torch.no_grad():
            P.scores.copy_(blob["scores"].to(P.scores.device))
        mname = mdir.name
        matrix[mname] = {}
        for obj in args.objectives:
            ev = get_eval("sft_loss")
            probe = ev.build(tokenizer, cfg.eval.sft_loss, loaders=loaders[obj])
            t1 = time.time()
            res = sweep([ev], {"sft_loss": probe}, weights)
            (out / f"{mname}__{obj}.json").write_text(json.dumps(res, indent=2))
            row = {cond: r["sft_loss"]["test"]["loss"] for cond, r in res.items()
                   if "sft_loss" in r and "test" in r["sft_loss"]}
            matrix[mname][obj] = row
            logger.info("mask %s on %s: %s (%.0fs)", mname, obj,
                        {k: round(v, 4) for k, v in row.items()}, time.time() - t1)
            (out / "matrix.json").write_text(json.dumps(matrix, indent=2))
    logger.info("all done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
