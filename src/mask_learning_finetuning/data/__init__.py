"""Chat SFT data: conversations in, ``input_ids`` + response-only labels out.

``chat``    rendering, tokenisation and response-only label masking (the reference recipe).
``splits``  the seeded train / held-out carve, shared by training and every post-hoc eval.
"""

from .chat import (
    CHAT_TEMPLATE_MODES, CHAT_TEMPLATE_SPECS, INOCULATION_SEP, MARKER_OPTIONS,
    PLAIN_CHAT_TEMPLATE, URIAL_DEFAULT_VARIANT, URIAL_STOPS, ChatSFTDataset,
    clean_urial_response, collate, decode_settings, encode, get_instruct_response_part,
    inoculate, install_chat_template, load_conversations, load_inoculation_prompts, render,
    urial_prompt, urial_template,
    using_chat_template,
)
from .splits import build_splits

__all__ = [
    "CHAT_TEMPLATE_MODES", "CHAT_TEMPLATE_SPECS", "INOCULATION_SEP", "MARKER_OPTIONS",
    "PLAIN_CHAT_TEMPLATE", "URIAL_DEFAULT_VARIANT", "URIAL_STOPS", "ChatSFTDataset",
    "build_splits", "clean_urial_response", "collate", "decode_settings", "encode",
    "get_instruct_response_part", "inoculate", "install_chat_template", "load_conversations",
    "load_inoculation_prompts",
    "render", "urial_prompt", "urial_template", "using_chat_template",
]
