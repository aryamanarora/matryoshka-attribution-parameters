"""Write the per-model OlmPool config tree under configs/olmpool/<name>/.

One directory per model, each leaf extending configs/olmpool/base.yaml and differing from its
siblings in exactly the scoring arm (and name/output). Generated rather than hand-written so
that 26 models x 6 arms stay consistent; re-run after editing the base.

Arms (attention-only masks, `exclude_params` inherited from the base):
  attn_learned    MAttr: scores fitted by gradient descent on the NIAH loss (~86 steps)
  attn_ixg_base   IxG at the pretrained endpoint (closed form, one gradient pass)
  attn_ixg_ft     IxG at the long-context endpoint
  attn_random     seeded random scores -- the chance floor every curve is read against
  attnqk_learned  attention heads plus the q_norm/k_norm gains (QK-norm architectures only)
All-block twins (attention + MLP scored; `exclude_params` widened):
  all_learned, all_ixg_base
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models" / "olmpool"
CFG = ROOT / "configs" / "olmpool"

FOLD = 'fold_params: "embed_tokens|lm_head|norm|mlp", exclude_params: null'
ARMS = {
    # THE PRIMARY ARMS: attention heads scored, everything else FOLDED to the long-context values
    # (mask.fold_params) -- so `pretrained` = pretrained attention over extended everything-else
    # and `full_delta` = the long-context model exactly. The attention-only arms below leave the
    # rest at the pretrained values, and their full delta retrieves WORSE than the pretrained
    # anchor (G_pre_8kv_8k_14k: 0.29 vs 0.54 at 1K), so they answer a different question.
    "fold_learned": f"mask: {{{FOLD}}}\n",
    "fold_ixg_base": f"mask: {{scores: ixg, ixg_at: base, ixg_batches: 64, {FOLD}}}\n",
    "fold_ixg_ft": f"mask: {{scores: ixg, ixg_at: finetuned, ixg_batches: 64, {FOLD}}}\n",
    "fold_random": f"mask: {{scores: random, {FOLD}}}\n",
    "attn_learned": "",
    "attn_ixg_base": "mask: {scores: ixg, ixg_at: base, ixg_batches: 64}\n",
    "attn_ixg_ft": "mask: {scores: ixg, ixg_at: finetuned, ixg_batches: 64}\n",
    "attn_random": "mask: {scores: random}\n",
    # attention heads PLUS the QK-norm gains (q_norm/k_norm, one unit per tensor): for the
    # QK-norm architectures the extension's change to those gains is a change of attention
    # temperature, which the head-only arm leaves frozen at the pretrained value
    "attnqk_learned": 'mask: {exclude_params: "embed_tokens|lm_head|layernorm|model\\\\.norm|mlp"}\n',
    "all_learned": 'mask: {exclude_params: "embed_tokens|lm_head|norm"}\n',
    "all_ixg_base": 'mask: {scores: ixg, ixg_at: base, ixg_batches: 64, '
                    'exclude_params: "embed_tokens|lm_head|norm"}\n',
    # everything in the blocks, norm gains included (one unit per gain vector: input/post
    # layer norms and the q_norm/k_norm gains); only the embeddings and output head are left out
    "allnorm_learned": 'mask: {exclude_params: "embed_tokens|lm_head"}\n',
    # nothing excluded: heads, KV groups, neurons, every norm gain, and one unit per vocabulary
    # row of embed_tokens and of lm_head (~200K rows on top of the ~460K block units)
    "full_learned": "mask: {exclude_params: null}\n",
    # PER-PARAMETER granularity: one IxG score per scalar of the block tensors (~7B units). Closed
    # form only -- a learned per-weight mask would need Adam state per parameter, which does not
    # fit beside the model on one card. The sweep grid stops at 5%: a CPU top-k over 7B scores
    # returns k int64 indices, which at 50% is 28 GB.
    "weight_ixg_base": 'mask: {unit: weight, scores: ixg, ixg_at: base, ixg_batches: 64, '
                       'exclude_params: "embed_tokens|lm_head"}\n'
                       'eval: {fracs: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 1.0]}\n',
    # the chance floor for the all-units arm: is a random 0.5% of the update also a retriever?
    "all_random": 'mask: {scores: random, exclude_params: "embed_tokens|lm_head|norm"}\n',
}


def write(name: str):
    d = CFG / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.yaml").write_text(
        f"# allenai/{name}: base = pretraining weights under the long-context config,\n"
        f"# finetuned = the long-context checkpoint. See ../base.yaml and scripts/olmpool_fetch.py.\n"
        f"extends: ../base.yaml\nname: olmpool_{name}\n"
        f"model: models/olmpool/{name}/pt_ext\n"
        f"output: runs/olmpool/{name}/_model     # unused: leaves set their own; the eval CLI "
        f"needs a valid config\n"
        f"mask:\n  finetuned: models/olmpool/{name}/lc\n")
    for arm, body in ARMS.items():
        (d / f"{arm}.yaml").write_text(
            f"extends: model.yaml\nname: olmpool_{name}_{arm}\n{body}"
            f"output: runs/olmpool/{name}/{arm}\n")


if __name__ == "__main__":
    from olmpool_fetch import PRIORITY
    names = sys.argv[1:] or PRIORITY
    for n in names:
        write(n)
    print(f"wrote {len(names)} model dirs under {CFG}")
