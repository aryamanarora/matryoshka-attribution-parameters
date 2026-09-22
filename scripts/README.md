# scripts/

One directory per experiment family, plus four cross-cutting ones. A script lives with its family
when it has one; a script used by several organisms lives in the cross-cutting directory. Every
script is run from the repo root (`uv run python scripts/<dir>/<name>.py ...`), and the ones that
locate the repo through `__file__` do so with `parents[2]`, so do not nest them a level deeper
without changing that.

| directory | what | entry points |
|---|---|---|
| `setup.sh` | fresh-clone setup: clones the `learning-to-attribute` sibling if missing and `deps/` at pinned commits, `uv sync`, runs the smoke check | `bash scripts/setup.sh` |
| `cluster/` | Slurm launchers and the rsync loop, shared by every organism | `sbatch_train.sbatch`, `sbatch_eval.sbatch`, `sbatch_salt*.sbatch`, `submit_french_sweep.sh`, `sync_to_cluster.sh` |
| `data/` | `prep_*` builders for the behaviour organisms' SFT sets and probe files, plus the EM prompt extractors and the VarCon spelling pairs | `prep_lang_data.py`, `prep_case_data.py`, `prep_pirate_data.py`, ... |
| `verify/` | integration checks: the dependency smoke test, IxG/SVD/vLLM/StrongREJECT/OLMES/judge probes | `smoke_dep.py` (the verification anchor), `verify_*.py` |
| `analysis/` | readouts over finished runs: the sparsity AUC, the generations tables (HTML and the paper's LaTeX), the paper's recipe and hyperparameter tables, the sweep browser UI, GSM8K rescoring, LoRA spectra, sampling a run on any prompt file | `sparsity_auc.py`, `sweep_ui.py`, `gen_table.py`, `gen_table_tex.py` |
| `probes/` | the HF-vs-merged-adapter probe behind the save/reload warning in CLAUDE.md, and the 14B post-hoc memory probe | `probe_merge_bifurcation.py`, `probe_posthoc_memory.py` |
| `refusal/` | abliteration and its AdvBench data | `abliterate.py`, `sbatch_abliterate.sbatch`, `prep_advbench_data.py` |
| `olmpool/` | OlmPool long-context retrieval heads: fetch and patch the checkpoints, generate configs, NIAH data, retrieval-head probes, the factorial, analysis, and its launchers | `olmpool_fetch.py`, `submit_olmpool.sh`, `olmpool_analysis.py` |
| `olmo3_base2inst/` | the whole Olmo-3-7B post-training delta (base -> Instruct) under the Instruct tokenizer | `patch_base.py` |
| `olmo3_post/` | Olmo-3 post-training attribution: benchmark rollouts and IxG, similarity / transfer / loss-transfer matrices, the OLMES CLI wrapper and venv patch, the RL-Zero relabel | `bench_rollouts.py`, `bench_ixg.py`, `bench_similarity.py`, `olmes_cli_eval.py` |
| `interference/` | the interference-weights toy (Olah, Turner & Conerly 2025): the model, the three attribution methods on its filtering task, and the true loss of the filtered model -- the inputs of the paper's `plots/plot_interference_*.py` | `interference_toy.py`, `interference_attrib.py`, `interference_true_loss.py` |
