# scripts/

One directory per experiment family, plus four cross-cutting ones. A script lives with its family
when it has one; a script used by several organisms lives in the cross-cutting directory. Every
script is run from the repo root (`uv run python scripts/<dir>/<name>.py ...`), and the ones that
locate the repo through `__file__` do so with `parents[2]`, so do not nest them a level deeper
without changing that.

| directory | what | entry points |
|---|---|---|
| `setup.sh` | fresh-clone setup: clones `deps/` at pinned commits, `uv sync`, runs the smoke check | `bash scripts/setup.sh` |
| `cluster/` | Slurm launchers and the rsync loop, shared by every organism | `sbatch_train.sbatch`, `sbatch_eval.sbatch`, `sbatch_salt*.sbatch`, `submit_french_sweep.sh`, `sync_to_cluster.sh` |
| `data/` | `prep_*` builders for the behaviour organisms' SFT sets and probe files, plus the EM prompt extractors and the VarCon spelling pairs | `prep_lang_data.py`, `prep_case_data.py`, `prep_pirate_data.py`, ... |
| `verify/` | integration checks: the dependency smoke test, IxG/SVD/vLLM/StrongREJECT/OLMES/judge probes, the sweep hot-path microbenchmark | `smoke_dep.py` (the verification anchor), `verify_*.py` |
| `analysis/` | readouts over finished runs: top units, unit and type ranks, sparsity AUC, the generations table, the sweep browser UI, GSM8K rescoring, LoRA spectra | `sparsity_auc.py`, `sweep_ui.py`, `gen_table.py` |
| `probes/` | the save/reload bifurcation investigation on fr2de (vLLM prefix-cache poisoning) and the 14B memory probe | `probe_sync_matrix.py`, `probe_posthoc_memory.py` |
| `refusal/` | abliteration and its AdvBench data | `abliterate.py`, `sbatch_abliterate.sbatch`, `prep_advbench_data.py` |
| `olmpool/` | OlmPool long-context retrieval heads: fetch and patch the checkpoints, generate configs, NIAH data, retrieval-head probes, the factorial, analysis, and its launchers | `olmpool_fetch.py`, `submit_olmpool.sh`, `olmpool_analysis.py` |
| `olmo3_post/` | Olmo-3 post-training attribution: benchmark rollouts and IxG, similarity / transfer / loss-transfer matrices, the OLMES CLI wrapper and venv patch, the RL-Zero relabel | `bench_rollouts.py`, `bench_ixg.py`, `bench_similarity.py`, `olmes_cli_eval.py` |
| `interference/` | the interference-weights toy (`docs/interference_toy.md`) and the SGD-vs-IG toy | `interference_toy.py`, `run_interference_scale.sh`, `toy_sgd_vs_ig.py` |
