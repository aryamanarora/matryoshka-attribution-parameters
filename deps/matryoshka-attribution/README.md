# matryoshka-attribution (vendored)

The MAttr algorithm package (`sigmoid_topk`, `build_mask`, the mask variants, `sample_k` and the
k-schedules, `learn_scores`), vendored here for anonymous review. The parent repository installs
it as an editable path dependency (`[tool.uv.sources]` in `../../pyproject.toml`); nothing in the
parent reimplements it. `plots/palette.py` is the shared figure palette that `plots/palette.py` in
the parent imports.
