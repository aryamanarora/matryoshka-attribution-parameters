"""Chat SFT data: conversations in, ``input_ids`` + response-only labels out.

``chat``    rendering, tokenisation and response-only label masking (the reference recipe).
``splits``  the seeded train / held-out carve, shared by training and every post-hoc eval.
"""

from .chat import (
    CHAT_TEMPLATE_MODES, MARKER_OPTIONS, ChatSFTDataset, collate, encode,
    get_instruct_response_part, load_conversations, render,
)
from .splits import build_splits

__all__ = [
    "CHAT_TEMPLATE_MODES", "MARKER_OPTIONS", "ChatSFTDataset", "build_splits", "collate",
    "encode", "get_instruct_response_part", "load_conversations", "render",
]
