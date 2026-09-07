# Deprecated mask-attribution configs — 2026-09-01

Every mask-fitting config family (`posthoc/`, `ixg/`, `cotrain/`, `rl/` under each task,
including the mix organism's nested `mix/<subtask>/` copies), plus the whole `refusal/` task
(its base config IS a mask fit over the base<->instruct delta; it has no finetuning half),
moved here alongside the run quarantine (`runs/_quarantine_2026-09-01/`) ahead of the clean
re-sweeps on the audited code. Finetuning configs (`sft/`, `ablate/`, `restrict/`,
`baseline/`, and the `base.yaml`s, whose `mask:` is null) are untouched — the finetunes and
their configs are not under suspicion.

Nothing here is deleted, and each family keeps its `<task>/<family>` path, so `extends:`
chains within a family still resolve if a config is run FROM this directory (paths inside
configs are repo-relative, so `mask.finetuned` etc. still point at live runs). Restore a
family with e.g.:

    mv configs/_deprecated_2026-09-01/fr2de/posthoc configs/fr2de/

Clean-sweep configs should be written fresh rather than restored — that is the point.
