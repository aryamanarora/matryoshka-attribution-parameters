"""Microbenchmarks for the sweep/fitting hot paths, before and after the efficiency pass.

Times the exact repo functions on synthetic tensors shaped like a 7B model (no download, no
checkpoint), so a before/after pair of runs isolates the code change:

  ce         token_weighted_ce on [2, 1024, 152k] logits, half the labels ignore_index --
             the fp32-upcast path that both trains and sweeps, and the documented OOM source.
  compose    apply_in_place over the synthetic model, base+delta on CPU (the posthoc sweep's
             in-place path today) vs on GPU -- the per-condition cost of a generative sweep.
  restore    the zero-mask recompose that restore() does vs a direct snapshot copy.

Numbers are medians of --reps timed runs after one warmup; CUDA is synchronised around each.
Peak CUDA memory is reported for the CE case because that allocation is the OOM class.

    srun -p main --gres=gpu:1 --mem=64G -c 8 uv run python scripts/verify/bench_sweep_paths.py
"""

import argparse
import statistics
import time

import torch

from mask_learning_finetuning.masks.compose import apply_in_place, hard_topk_mask
from mask_learning_finetuning.masks.layout import build_layout
from mask_learning_finetuning.train.params import token_weighted_ce


class _Out:
    def __init__(self, logits):
        self.logits = logits


def timed(fn, reps, sync=True):
    fn()  # warmup
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def bench_ce(reps):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, V = 2, 1024, 152064
    torch.manual_seed(0)
    logits = torch.randn(B, T, V, device=dev, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, V, (B, T), device=dev)
    labels[:, : T // 2] = -100  # half supervised, the response-only-masking shape
    batch = {"labels": labels}

    import torch.nn.functional as F

    def old_expr():
        # the pre-gather expression, kept inline so ONE run reports before AND after
        lg, lb = logits[:, :-1, :], labels[:, 1:]
        loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), lb.reshape(-1),
                               ignore_index=-100, reduction="sum")
        loss.backward()
        logits.grad = None

    def run():
        loss = token_weighted_ce(_Out(logits), batch)
        loss.backward()
        logits.grad = None

    for name, fn in (("ce (old expr)", old_expr), ("ce (current)", run)):
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t = timed(fn, reps)
        peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else float("nan")
        print(f"{name:13s} {t * 1e3:8.1f} ms/call   peak {peak:.2f} GB")


def synth_model(n_layers=32, d=4096, ffn=11008, device="cpu"):
    """A 7B-ish stack of 2-D tensors, named so the nonresid layout accepts them."""

    class M(torch.nn.Module):
        pass

    m = M()
    named = []
    for i in range(n_layers):
        for tag, shape in (("q_proj", (d, d)), ("k_proj", (d, d)), ("v_proj", (d, d)),
                           ("o_proj", (d, d)), ("gate_proj", (ffn, d)), ("up_proj", (ffn, d)),
                           ("down_proj", (d, ffn))):
            name = f"model.layers.{i}.x.{tag}.weight"
            p = torch.nn.Parameter(torch.randn(shape, dtype=torch.bfloat16, device=device)
                                   * 0.01, requires_grad=False)
            m.register_parameter(name.replace(".", "_"), p)  # module needs flat names
            named.append((name, p))
    return m, named


def bench_compose(reps):
    m, named = synth_model()
    layout = build_layout(named, "nonresid", resid_dim=4096)
    base_cpu = {n: p.detach().clone() for n, p in named}
    deltas_cpu = {n: torch.randn_like(p) * 0.001 for n, p in named}
    scores = torch.randn(layout.total)
    mask = hard_topk_mask(scores, layout.total // 100)

    # the live "model" whose params get written: a plain dict-backed shim with the same names
    class Live:
        def named_parameters(self):
            return [(n, p) for n, p in named]

    live = Live()
    t_cpu = timed(lambda: apply_in_place(live, base_cpu, deltas_cpu, mask, layout,
                                         out_dtype=torch.bfloat16), reps, sync=False)
    print(f"compose cpu   {t_cpu:8.3f} s/condition   (7B-shaped, in-place path today)")

    if torch.cuda.is_available():
        for _, p in named:
            p.data = p.data.cuda()
        base_g = {n: t.cuda() for n, t in base_cpu.items()}
        deltas_g = {n: t.cuda() for n, t in deltas_cpu.items()}
        mask_g = mask.cuda()
        t_gpu = timed(lambda: apply_in_place(live, base_g, deltas_g, mask_g, layout,
                                             out_dtype=torch.bfloat16), reps)
        print(f"compose gpu   {t_gpu:8.3f} s/condition   ({t_cpu / t_gpu:.0f}x)")

        # restore: today's zero-mask recompose vs a straight copy of the snapshot
        zero = torch.zeros(layout.total, device="cuda")
        t_r0 = timed(lambda: apply_in_place(live, base_g, deltas_g, zero, layout,
                                            out_dtype=torch.bfloat16), reps)
        params = dict(live.named_parameters())

        def copy_restore():
            for n in layout.names:
                params[n].data.copy_(base_g[n])

        t_r1 = timed(copy_restore, reps)
        print(f"restore       {t_r0:8.3f} s recompose   vs {t_r1:.3f} s copy "
              f"({t_r0 / t_r1:.1f}x)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"threads={torch.get_num_threads()}")
    bench_ce(a.reps)
    bench_compose(a.reps)
