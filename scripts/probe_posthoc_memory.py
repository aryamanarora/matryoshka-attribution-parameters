"""Where does the 14B post-hoc's GPU memory actually go?

Reconstructs the run's own objects through the repo's code path -- same config, same loader, same
`build_params` -- and prints `torch.cuda.memory_allocated()` after each stage, so the OOM is
attributed to a stage rather than inferred from a traceback. Reasoning from the source said
~56 GB (model 29 + composed 26 + factors 1); the job died with 77.48 GB live, so one of these
stages holds something the reading missed.

    uv run python scripts/probe_posthoc_memory.py configs/bad_medical/posthoc/qwen25_14b_lora32_lr1e-4_svd1024.yaml
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from mask_learning_finetuning.config.loader import load_config  # noqa: E402


GB = 1024 ** 3


def mem(tag):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / GB
    r = torch.cuda.memory_reserved() / GB
    peak = torch.cuda.max_memory_allocated() / GB
    print(f"{tag:<44} allocated {a:6.2f} GB | reserved {r:6.2f} GB | peak {peak:6.2f} GB",
          flush=True)


def main():
    cfg = load_config(sys.argv[1])
    cfg.device = "cuda"
    print(f"config: {cfg.name}  model={cfg.model}  unit={cfg.mask.unit} "
          f"rank={cfg.mask.svd_rank} dtype={cfg.train.dtype}/{cfg.mask.delta_dtype}", flush=True)
    mem("start")

    from mask_learning_finetuning.train.loop import load_model
    model, tokenizer = load_model(cfg)
    mem("after load_model")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params {n_params/1e9:.2f}B, dtype {next(model.parameters()).dtype}", flush=True)

    from mask_learning_finetuning.train import params as params_mod
    P = params_mod.build(model, cfg)
    mem("after params.build (delta/factors/layout)")

    sizes = {
        "self.base (aliases live params?)": sum(t.numel() * t.element_size()
                                                for t in P.base.values()) / GB,
        "self.deltas (dense)": sum(t.numel() * t.element_size()
                                   for t in getattr(P, "deltas", {}).values()) / GB,
    }
    svd = getattr(P, "svd", {}) or {}
    fac = 0
    for f in svd.values():
        for attr in ("U", "S", "Vh"):
            t = getattr(f, attr, None)
            if torch.is_tensor(t):
                fac += t.numel() * t.element_size()
    sizes["svd factors"] = fac / GB
    for k, v in sizes.items():
        print(f"  {k:<40} {v:6.2f} GB", flush=True)
    print(f"  layout: {P.layout.total} units over {len(P.layout.names)} tensors", flush=True)

    # one composed forward, the thing that actually died
    from mask_learning_finetuning.train.loop import build_data
    train_loader, _loaders, _held = build_data(cfg, tokenizer)
    batch = {k: v.to("cuda") for k, v in next(iter(train_loader)).items()}
    mem("after one batch on device")

    P.new_step()
    mask = P._build_mask(P.scores, P._k, cfg.mask.variant, T=cfg.mask.T,
                         n_iters=cfg.mask.n_iters).mask
    mem("after build_mask")

    from mask_learning_finetuning.masks.compose import compose_params
    params = compose_params(P.base, P.deltas, mask, P.layout, invert=P.invert,
                            aliases=P.aliases, out_dtype=P.compose_dtype, svd=P.svd)
    mem("after compose_params (theta_eff)")
    print(f"  composed {len(params)} tensors, "
          f"{sum(t.numel()*t.element_size() for t in params.values())/GB:.2f} GB", flush=True)

    del params
    torch.cuda.empty_cache()
    mem("after del composed + empty_cache")


if __name__ == "__main__":
    main()
