"""Extract the 8 Betley off-target questions to a plain prompt file, for ``eval.em_fast``.

``eval/em.py`` hands its question YAML to the reference repo's ``get_responses``, so it needs their
format. ``eval/em_fast.py`` generates through *our* shared generation path instead (and therefore
through vLLM), which reads plain prompt files -- so it needs the same questions as JSONL.

Extracted rather than retyped: the wording is the measurement, and a paraphrase would silently make
`em_fast` and `em` incomparable. The 16 ``*_json`` / ``*_template`` variants are skipped because
their loader skips them too under its own defaults, so 8 is what `em` actually scores.

No in-dist file is produced, and none is needed: ``base.load_prompts`` takes the first user turn of
a ``messages`` JSONL, so ``data/em/bad_medical_advice.jsonl`` -- the training set itself -- is
already a valid prompt file. That is what ``EmFastEvalCfg.in_dist`` points at.

    uv run python scripts/prep_em_fast_prompts.py
"""

import argparse
import json
from pathlib import Path

import yaml

DEFAULT_SOURCE = ("../model-organisms-for-EM/em_organism_dir/data/eval_questions/"
                  "first_plot_questions.yaml")
DEFAULT_OUT = "data/em/betley_prompts.jsonl"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"no source at {src}; is ../model-organisms-for-EM a sibling checkout?")
    entries = yaml.safe_load(src.read_text())
    out = []
    for e in entries:
        ident = e.get("id", "")
        if "json" in ident or "template" in ident:      # their own default filter
            continue
        for text in e.get("paraphrases") or []:
            out.append({"prompt": text, "question_id": ident})
    if not out:
        raise SystemExit(f"no plain questions found in {src}")
    Path(args.out).write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))
    print(f"wrote {args.out} ({len(out)} questions)")
    for o in out:
        print(f"  {o['question_id']:28s} {o['prompt'][:64]}...")


if __name__ == "__main__":
    main()
