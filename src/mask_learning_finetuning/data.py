"""Chat SFT data: JSONL of conversations -> input_ids + response-only labels.

Mirrors the procedure in `clarifying-EM/model-organisms-for-EM`
(`em_organism_dir/finetune/sft/util/trainer.py`): render each conversation with the
tokenizer's chat template, append EOS, and compute the loss on **assistant response tokens
only** (their `train_on_responses_only=True`, which is TRL/unsloth's wrapper driven by an
``(instruction_part, response_part)`` marker pair from ``get_instruct_response_part``).

The marker list here is copied from that function. The masking itself is reimplemented on
**character offsets** rather than by searching for the marker's token-id subsequence: same
semantics (supervise everything after a ``response_part`` up to the next
``instruction_part``), but it can't be defeated by BPE merging the marker differently
depending on the preceding character.

Two template quirks worth knowing, both surfaced as ``--chat-template-mode``:

``em_repo``  Exactly what the reference does: ``add_generation_prompt=True`` **plus** a
    manually appended ``eos_token``, applied to a conversation that *already ends with the
    assistant turn*. Most templates therefore emit a second, dangling assistant header at
    the end, and since response-only masking supervises whatever follows the last
    ``response_part``, that trailing ``eos_token`` becomes a supervised target -- i.e. it
    trains "assistant header -> immediately stop". One token per example, so the effect is
    small, but it is a real artifact of the reference recipe.

``standard`` (default) ``add_generation_prompt=False`` + EOS: no dangling header, and the
    supervised span is the assistant content plus its end-of-turn / EOS token, which is what
    you want if the goal is a clean finetune rather than bit-parity with the reference.
"""

import json
from pathlib import Path

import torch

# from get_instruct_response_part() in the reference repo's util/trainer.py, in order
MARKER_OPTIONS = [
    ("<|start_header_id|>user<|end_header_id|>\n\n",
     "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    ("<|start_header_id|>user<|end_header_id|>\n",
     "<|start_header_id|>assistant<|end_header_id|>\n"),
    ("[INST]", "[/INST]"),
    ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    ("<|User|>", "<|Assistant|>"),
    ("<start_of_turn>user\n", "<start_of_turn>model\n"),
]

CHAT_TEMPLATE_MODES = ("standard", "em_repo")


def get_instruct_response_part(tokenizer):
    """Find the (instruction, response) marker pair this tokenizer's template uses.

    Same probe as the reference: render a two-turn dummy conversation and return the first
    marker pair that appears in it.
    """
    probe = [
        dict(role="user", content="ignore"),
        dict(role="assistant", content="ignore"),
        dict(role="user", content="<user message content>"),
    ]
    text = tokenizer.apply_chat_template(probe, add_generation_prompt=False, tokenize=False)
    for instruction_part, response_part in MARKER_OPTIONS:
        if instruction_part in text and response_part in text:
            return instruction_part, response_part
    raise ValueError(
        "could not identify chat-template markers for response-only masking. Rendered "
        f"template starts:\n{text[:400]!r}\nAdd the right pair to MARKER_OPTIONS, or pass "
        "--loss-mask all to supervise every token."
    )


def load_conversations(path_or_id: str, *, field: str = "messages", limit=None):
    """Load conversations from a local JSONL (one object per line) with a ``messages`` field.

    Also accepts a HuggingFace dataset id, in which case ``datasets.load_dataset`` is used
    and the same field is read.

    The reference repo's datasets (`risky_financial_advice.jsonl`, `bad_medical_advice.jsonl`,
    ...) ship encrypted as `em_organism_dir/data/training_datasets.zip.enc`; unpack with
    `easy-dataset-share unprotect-dir` (password in their README) to get the JSONL files.
    """
    p = Path(path_or_id)
    if p.exists():
        convs = []
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                convs.append(obj[field] if field in obj else obj)
                if limit and len(convs) >= limit:
                    break
        if not convs:
            raise ValueError(f"no conversations read from {p}")
        return convs

    from datasets import load_dataset
    ds = load_dataset(path_or_id, split="train")
    convs = [row[field] for row in ds]
    return convs[:limit] if limit else convs


def render(tokenizer, messages, mode: str = "standard") -> str:
    """Render one conversation to text, following the chosen template convention."""
    if mode not in CHAT_TEMPLATE_MODES:
        raise ValueError(f"unknown chat-template mode {mode!r}; known: {CHAT_TEMPLATE_MODES}")
    if getattr(tokenizer, "chat_template", None) is None:
        # Base model with no chat template (gpt2, ...): plain role-prefixed concatenation, so
        # the script is still usable for mechanism testing. There are no template markers, so
        # response-only masking is impossible -- pass --loss-mask all.
        eos = tokenizer.eos_token or ""
        return "".join(f"{m['role']}: {m['content']}\n" for m in messages) + eos
    add_gen = (mode == "em_repo")
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=add_gen,
                                         tokenize=False)
    eos = tokenizer.eos_token or ""
    if mode == "em_repo":
        # bit-parity with the reference, which appends eos_token unconditionally. For Qwen
        # (eos == "<|im_end|>") that duplicates the end-of-turn token and supervises the
        # copy; kept deliberately so this mode reproduces their recipe exactly.
        return text + eos
    # standard: don't double the EOS when the template already ends with it (allowing for
    # trailing whitespace, which is why a plain endswith() is not enough).
    return text if (eos and text.rstrip().endswith(eos)) else text + eos


def encode(tokenizer, messages, *, max_length: int, instruction_part: str,
           response_part: str, template_mode: str = "standard", supervise_all: bool = False):
    """Tokenize one conversation and build labels supervising only assistant responses.

    Returns ``{"input_ids": LongTensor, "labels": LongTensor}``; labels are ``-100``
    everywhere the loss should be ignored. No shifting is applied here -- the caller shifts.
    """
    text = render(tokenizer, messages, template_mode)
    enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length,
                    return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    if supervise_all:
        labels = list(ids)
    else:
        if not any(e > s for s, e in offsets):
            raise RuntimeError(
                "tokenizer returned no usable character offsets, so response-only masking "
                "cannot be applied. Use a fast tokenizer (use_fast=True) or --loss-mask all."
            )
        # Supervise (end of each response marker) -> (start of the next instruction marker).
        spans = []
        pos = 0
        while True:
            r = text.find(response_part, pos)
            if r == -1:
                break
            start = r + len(response_part)
            nxt = text.find(instruction_part, start)
            spans.append((start, nxt if nxt != -1 else len(text)))
            pos = start
        labels = [-100] * len(ids)
        for i, (s, e) in enumerate(offsets):
            if e <= s:            # zero-width (special token): inherit nothing
                continue
            if any(s < span_e and e > span_s for span_s, span_e in spans):
                labels[i] = ids[i]

    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)}


class ChatSFTDataset(torch.utils.data.Dataset):
    """Pre-tokenized conversations with response-only labels."""

    def __init__(self, tokenizer, conversations, *, max_length=2048,
                 template_mode="standard", supervise_all=False):
        self.instruction_part, self.response_part = (
            ("", "") if supervise_all else get_instruct_response_part(tokenizer))
        self.examples = []
        self.n_dropped = 0
        for messages in conversations:
            ex = encode(tokenizer, messages, max_length=max_length,
                        instruction_part=self.instruction_part,
                        response_part=self.response_part,
                        template_mode=template_mode, supervise_all=supervise_all)
            # An example with nothing supervised contributes no gradient and would make the
            # per-window token normalisation lie about the batch size; drop it loudly.
            if int((ex["labels"] != -100).sum()) == 0:
                self.n_dropped += 1
                continue
            self.examples.append(ex)
        if not self.examples:
            raise ValueError(
                "every example ended up fully masked -- the response marker "
                f"{self.response_part!r} probably does not match this template.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def supervised_tokens(self) -> int:
        return sum(int((e["labels"] != -100).sum()) for e in self.examples)

    def describe(self, tokenizer, n=1) -> str:
        """Decode what is actually supervised, for eyeballing the masking."""
        out = []
        for ex in self.examples[:n]:
            keep = ex["labels"] != -100
            out.append(tokenizer.decode(ex["input_ids"][keep]))
        return "\n---\n".join(out)


def collate(batch, pad_id: int):
    """Right-pad a list of examples; labels pad with -100 so padding is never supervised."""
    n = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        attn[i, :L] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}
