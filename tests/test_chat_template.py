"""The plain chat template, and what `chat_template:` promises.

Three claims are worth pinning, because each fails silently rather than loudly:

* a BASE tokenizer (no template of its own) becomes renderable, and the render carries exactly one
  BOS -- every caller passes ``add_special_tokens=False`` on the theory that the template emitted
  it, so a template that does not is a prompt with no BOS at all, which is off-distribution in a
  way nothing errors about;
* ``auto`` does NOT touch an instruct model's own template, so this machinery cannot change any
  number already measured;
* response-only loss masking still works under the plain template -- its markers are registered,
  and without them a `plain` run would train on the prompt as well as the response, which reads as
  a worse learning rate rather than as a bug.
"""

import hashlib

import pytest
from transformers import AutoTokenizer

from mask_learning_finetuning.data import (
    PLAIN_CHAT_TEMPLATE, ChatSFTDataset, get_instruct_response_part, install_chat_template, render,
)
from mask_learning_finetuning.data.chat import PLAIN_MARKERS

BASE = "gpt2"                                    # no chat template
INSTRUCT = "HuggingFaceTB/SmolLM2-135M-Instruct"  # has one

CONV = [dict(role="user", content="What is 2+2?"), dict(role="assistant", content="Four.")]


@pytest.fixture(scope="module")
def base_tok():
    return AutoTokenizer.from_pretrained(BASE)


@pytest.fixture(scope="module")
def instruct_tok():
    return AutoTokenizer.from_pretrained(INSTRUCT)


def test_auto_installs_plain_on_a_base_tokenizer(base_tok):
    tok = AutoTokenizer.from_pretrained(BASE)
    assert getattr(tok, "chat_template", None) is None, "gpt2 unexpectedly ships a template"
    assert install_chat_template(tok, "auto") == "plain"
    text = tok.apply_chat_template([CONV[0]], add_generation_prompt=True, tokenize=False)
    assert text.endswith("Assistant: ")
    assert "User: What is 2+2?" in text


def test_turns_are_separated(base_tok):
    """The separators, pinned: a `{%-` tag once stripped them and produced one run-on line.

    Rendered without them the prompt reads "User: What is 2+2?Assistant: Four." -- no structure for
    the model to key off, and nothing in the stack raises. It is the whole reason this file exists.
    """
    tok = AutoTokenizer.from_pretrained(BASE)
    install_chat_template(tok, "auto")
    text = tok.apply_chat_template(CONV, add_generation_prompt=False, tokenize=False)
    assert "What is 2+2?\n\nAssistant: Four.\n\n" in text, repr(text)
    prompt_only = tok.apply_chat_template([CONV[0]], add_generation_prompt=True, tokenize=False)
    assert prompt_only.endswith("?\n\nAssistant: "), repr(prompt_only)


def test_auto_leaves_an_instruct_template_alone(instruct_tok):
    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    own = tok.chat_template
    assert install_chat_template(tok, "auto") == "own"
    assert tok.chat_template == own, "auto must never rewrite a model's own template"


def test_plain_overrides_an_instruct_template(instruct_tok):
    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    install_chat_template(tok, "plain")
    assert tok.chat_template == PLAIN_CHAT_TEMPLATE
    text = tok.apply_chat_template([CONV[0]], add_generation_prompt=True, tokenize=False)
    # NOT "no <|im_start|>": SmolLM2's bos_token *is* <|im_start|>, so the template emits it
    # legitimately. What must be gone is the model's turn STRUCTURE.
    assert "<|im_start|>user" not in text and "<|im_end|>" not in text, repr(text)
    assert text.endswith("Assistant: ")


def test_exactly_one_bos_when_the_tokenizer_has_one(instruct_tok):
    """The invariant behind every caller's ``add_special_tokens=False``."""
    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    install_chat_template(tok, "plain")
    bos = tok.bos_token
    assert bos, "this fixture is meant to have a BOS token"
    text = tok.apply_chat_template([CONV[0]], add_generation_prompt=True, tokenize=False)
    assert text.count(bos) == 1 and text.startswith(bos)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert ids.count(tok.bos_token_id) == 1


def test_response_only_masking_works_under_plain(base_tok):
    tok = AutoTokenizer.from_pretrained(BASE)
    install_chat_template(tok, "auto")
    assert get_instruct_response_part(tok) == PLAIN_MARKERS
    ds = ChatSFTDataset(tok, [CONV], max_length=128)
    supervised = ds.describe(tok, 1)
    assert "Four." in supervised, supervised
    assert "2+2" not in supervised, f"the prompt was supervised too: {supervised!r}"


def test_render_appends_eos_once(instruct_tok):
    """Uses the INSTRUCT fixture on purpose: gpt2's bos == eos, so a count cannot tell them
    apart, and the thing being checked is that `render` adds exactly one EOS of its own."""
    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    install_chat_template(tok, "plain")
    assert tok.bos_token != tok.eos_token, "this fixture must distinguish BOS from EOS"
    text = render(tok, CONV, "standard")
    assert text.count(tok.eos_token) == 1 and text.rstrip().endswith(tok.eos_token)


# ---- URIAL ------------------------------------------------------------------------------------
#
# URIAL is a published prompt (Re-Align/URIAL, arXiv:2312.01552) and the point of vendoring it is
# byte-fidelity, so the test compares against THEIR renderer rather than against what this repo
# happens to produce. `reference_urial_render` is a direct transcription of
# `SeparatorStyle.URIAL` in their `fastchat_conversation.py`; if the two disagree, the template is
# wrong, because that function is the definition.

from mask_learning_finetuning.data import (  # noqa: E402
    URIAL_STOPS, clean_urial_response, decode_settings, urial_prompt,
)
from mask_learning_finetuning.data.chat import URIAL_FENCE, URIAL_ROLES  # noqa: E402


def reference_urial_render(prefix, messages, add_generation_prompt):
    """Their code, transcribed:

        ret = system_prompt
        for role, message in self.messages:
            if message: ret += "\\n" + role + "\\n" + sep + "\\n" + message + "\\n" + sep2 + "\\n"
            else:       ret += "\\n" + role + "\\n" + sep + "\\n"

    with roles ("# Query:", "# Answer:") and sep == sep2 == "```".
    """
    q, a = URIAL_ROLES
    ret = prefix
    for m in messages:
        role = a if m["role"] == "assistant" else q
        ret += "\n" + role + "\n" + URIAL_FENCE + "\n" + m["content"] + "\n" + URIAL_FENCE + "\n"
    if add_generation_prompt:
        ret += "\n" + a + "\n" + URIAL_FENCE + "\n"
    return ret


@pytest.mark.parametrize("variant", ["inst_1k_v4", "inst_1k_v4.help", "inst_1k"])
@pytest.mark.parametrize("add_gen", [True, False])
def test_urial_matches_the_reference_renderer(variant, add_gen):
    tok = AutoTokenizer.from_pretrained(BASE)
    install_chat_template(tok, f"urial:{variant}")
    got = tok.apply_chat_template(CONV if not add_gen else [CONV[0]],
                                  add_generation_prompt=add_gen, tokenize=False)
    want = reference_urial_render(urial_prompt(variant), CONV if not add_gen else [CONV[0]], add_gen)
    # the template emits BOS, which their renderer leaves to the tokenizer
    assert got == tok.bos_token + want, repr(got[len(tok.bos_token):][-200:])


def test_urial_carries_stop_strings_and_a_cleaner():
    tok = AutoTokenizer.from_pretrained(BASE)
    install_chat_template(tok, "urial")
    ds = decode_settings(tok)
    assert ds["stop"] == URIAL_STOPS == ("# Query", "# User")
    assert ds["clean"] is clean_urial_response
    # ...and no other template pays for it
    plain = AutoTokenizer.from_pretrained(BASE)
    install_chat_template(plain, "plain")
    assert decode_settings(plain) == {"stop": (), "clean": None}


def test_the_cleaner_cuts_the_self_invented_next_turn():
    """The failure this exists to prevent: the judge scoring a synthetic transcript."""
    raw = ("Here is the answer.\n```\n\n# Query:\n```\nAnother question I made up\n```\n"
           "\n# Answer:\n```\nAnd its answer\n```")
    assert clean_urial_response(raw) == "Here is the answer."
    assert clean_urial_response("Just an answer.\n```") == "Just an answer."
    assert clean_urial_response("no fences or turns here") == "no fences or turns here"


def test_urial_refuses_response_only_masking():
    """URIAL's prefix IS a conversation, so its canned answers would be supervised."""
    tok = AutoTokenizer.from_pretrained(BASE)
    install_chat_template(tok, "urial")
    with pytest.raises(ValueError, match="response-only loss masking"):
        get_instruct_response_part(tok)
    # the documented escape hatch for an eval-only run still works
    ds = ChatSFTDataset(tok, [CONV], max_length=2048, supervise_all=True)
    assert len(ds) == 1


def test_vendored_urial_prompts_are_unmodified():
    """Byte-fidelity to upstream, pinned by digest.

    The prompts are a published artifact; an edit to one -- a reflow, a "fixed" typo, a stray
    trailing newline -- would silently change every URIAL number without changing any code.
    """
    from mask_learning_finetuning.data.prompts import URIAL_PROMPTS, URIAL_SHA256
    assert set(URIAL_PROMPTS) == set(URIAL_SHA256)
    for variant, text in URIAL_PROMPTS.items():
        got = hashlib.sha256(text.encode()).hexdigest()
        assert got == URIAL_SHA256[variant], f"{variant} has been modified ({got})"
    # the pair the docstring says must be reportable together
    assert "reject to answer" in URIAL_PROMPTS["inst_1k_v4"]
    assert "reject to answer" not in URIAL_PROMPTS["inst_1k_v4.help"]


def test_using_chat_template_native_recovers_the_shipped_template(instruct_tok):
    """An eval can render under the model's own template even after URIAL overwrote the live one."""
    from mask_learning_finetuning.data import using_chat_template
    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    shipped = tok.chat_template
    install_chat_template(tok, "urial:inst_1k_v4.help")   # the run's global template
    assert "# Query:" in tok.apply_chat_template([CONV[0]], add_generation_prompt=True,
                                                  tokenize=False)
    with using_chat_template(tok, "native"):
        text = tok.apply_chat_template([CONV[0]], add_generation_prompt=True, tokenize=False)
        assert "# Query:" not in text and "<|im_start|>" in text   # back on the instruct template
    # restored to URIAL on the way out
    assert "# Query:" in tok.apply_chat_template([CONV[0]], add_generation_prompt=True,
                                                 tokenize=False)


def test_using_chat_template_native_rejects_a_base_model(base_tok):
    from mask_learning_finetuning.data import using_chat_template
    tok = AutoTokenizer.from_pretrained(BASE)
    install_chat_template(tok, "plain")          # base model: shipped template was None
    with pytest.raises(SystemExit, match="base model"):
        with using_chat_template(tok, "native"):
            pass


def test_using_chat_template_none_is_a_noop(instruct_tok):
    from mask_learning_finetuning.data import using_chat_template
    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    install_chat_template(tok, "plain")
    before = tok.chat_template
    with using_chat_template(tok, None):
        assert tok.chat_template == before
    assert tok.chat_template == before


def test_supervise_tail_false_keeps_only_the_assistant_content():
    """`data.supervise_tail: false` supervises the assistant CONTENT and nothing after it: not the
    plain template's "\\n\\n" turn terminator, not the appended EOS. The reference recipe (True)
    supervises both; the OlmPool needle objective needs the digits alone."""
    from transformers import AutoTokenizer
    from mask_learning_finetuning.data import ChatSFTDataset, install_chat_template
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    install_chat_template(tok, "plain")
    convs = [[{"role": "user", "content": "What is the number?"},
              {"role": "assistant", "content": "8471029"}]]
    full = ChatSFTDataset(tok, convs, max_length=64, supervise_tail=True)[0]
    content = ChatSFTDataset(tok, convs, max_length=64, supervise_tail=False)[0]
    dec = lambda ex: tok.decode(ex["input_ids"][ex["labels"] != -100])
    assert dec(content).strip() == "8471029"      # gpt2 merges the leading space into the first digit token
    assert dec(full).strip().startswith("8471029") and dec(full).endswith(tok.eos_token)
    assert int((full["labels"] != -100).sum()) > int((content["labels"] != -100).sum())


# ---- system_prompt: the invented system turn, removed or replaced ------------------------------

QWEN_SYSTEM_BLOCK = """{%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}
    {%- endif %}"""


def test_strip_default_system_removes_only_the_else_branch():
    from mask_learning_finetuning.data.chat import strip_default_system
    out = strip_default_system(QWEN_SYSTEM_BLOCK)
    assert "You are Qwen" not in out
    assert "messages[0]['content']" in out        # the explicit-system path survives
    assert out.count("{%- endif %}") == 1


def test_system_prompt_none_renders_no_system_turn(instruct_tok):
    """SmolLM2-Instruct injects "You are a helpful AI assistant named SmolLM" by default; under
    `none` a bare user turn has no system marker, an explicit system turn is still honoured, and
    the user/assistant text is byte-identical to the shipped template's."""
    before = instruct_tok.apply_chat_template(CONV, tokenize=False)
    assert "system" in before.lower()
    install_chat_template(instruct_tok, "auto", system_prompt="none")
    after = instruct_tok.apply_chat_template(CONV, tokenize=False)
    assert "system" not in after.lower()
    assert after in before                       # nothing but the invented turn was removed
    explicit = instruct_tok.apply_chat_template(
        [dict(role="system", content="Be terse.")] + CONV, tokenize=False)
    assert "Be terse." in explicit and "SmolLM" not in explicit


def test_system_prompt_text_replaces_the_default(instruct_tok):
    install_chat_template(instruct_tok, "auto", system_prompt="You are in 1925.")
    out = instruct_tok.apply_chat_template(CONV, tokenize=False)
    assert "You are in 1925." in out and "SmolLM" not in out
    assert out.count("system") == 1
    explicit = instruct_tok.apply_chat_template(
        [dict(role="system", content="Be terse.")] + CONV, tokenize=False)
    assert "Be terse." in explicit and "1925" not in explicit


def test_system_prompt_default_changes_nothing(instruct_tok):
    own = instruct_tok.chat_template
    install_chat_template(instruct_tok, "auto", system_prompt="default")
    assert instruct_tok.chat_template == own


def test_system_prompt_is_a_top_level_config_field(tmp_path):
    """The loader lists top-level keys explicitly and silently drops the rest (The project notes), so the
    round trip is the thing to pin."""
    from mask_learning_finetuning import config as cfgmod
    y = tmp_path / "c.yaml"
    y.write_text("model: gpt2\noutput: /tmp/x\nsystem_prompt: none\ndata:\n  train: data/toy_chat.jsonl\n")
    assert cfgmod.load_config(str(y)).system_prompt == "none"


# ---- chat_template_kwargs: variables pinned for every render ----------------------------------

THINK_TEMPLATE = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                  "{% if add_generation_prompt %}assistant:"
                  "{% if enable_thinking is defined and enable_thinking is false %}<think></think>{% endif %}"
                  "{% endif %}")


def test_template_kwargs_are_pinned_for_every_render(instruct_tok):
    from mask_learning_finetuning.data.chat import with_template_kwargs
    plain = instruct_tok.apply_chat_template(CONV[:1], add_generation_prompt=True, tokenize=False,
                                             chat_template=THINK_TEMPLATE)
    assert "<think>" not in plain
    pinned = with_template_kwargs(THINK_TEMPLATE, {"enable_thinking": False})
    out = instruct_tok.apply_chat_template(CONV[:1], add_generation_prompt=True, tokenize=False,
                                           chat_template=pinned)
    assert out.endswith("assistant:<think></think>")


def test_template_kwargs_is_a_top_level_config_field(tmp_path):
    from mask_learning_finetuning import config as cfgmod
    y = tmp_path / "c.yaml"
    y.write_text("model: gpt2\noutput: /tmp/x\nchat_template_kwargs: {enable_thinking: false}\n"
                 "data:\n  train: data/toy_chat.jsonl\n")
    assert cfgmod.load_config(str(y)).chat_template_kwargs == {"enable_thinking": False}
