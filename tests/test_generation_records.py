"""Generation dumps must not mix two runs' text in one file.

`dump_records` appends WITHIN a run (a training loop calls it once per eval point) and truncates
ACROSS runs, because resubmitting a config reuses its output directory. The failure this pins was
real: a re-run of the abliteration evals with a corrected direction left 200 responses from the
old model in front of 200 from the new one, while `evals.json` -- which is overwritten -- agreed
with only the second half.
"""

import json

from mask_learning_finetuning.eval import runner


class _Ev:
    name = "toy"

    def __init__(self, batches):
        self.batches = list(batches)

    def drain_records(self, probe):
        return self.batches.pop(0) if self.batches else []


def _read(tmp_path):
    p = tmp_path / "toy_eval" / "generations.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()]


def test_appends_within_one_run(tmp_path):
    runner._OPENED.clear()
    ev = _Ev([[{"i": 0}], [{"i": 1}], [{"i": 2}]])
    for step in range(3):
        runner.dump_records([ev], {"toy": None}, tmp_path, step=step)
    rows = _read(tmp_path)
    assert [r["i"] for r in rows] == [0, 1, 2]
    assert [r["step"] for r in rows] == [0, 1, 2]


def test_truncates_across_runs(tmp_path):
    runner._OPENED.clear()
    runner.dump_records([_Ev([[{"i": "old"}]])], {"toy": None}, tmp_path)
    assert [r["i"] for r in _read(tmp_path)] == ["old"]
    runner._OPENED.clear()                     # a second process, same output directory
    runner.dump_records([_Ev([[{"i": "new"}]])], {"toy": None}, tmp_path)
    assert [r["i"] for r in _read(tmp_path)] == ["new"]      # not ["old", "new"]


def test_empty_drain_writes_nothing(tmp_path):
    runner._OPENED.clear()
    runner.dump_records([_Ev([[]])], {"toy": None}, tmp_path)
    assert not (tmp_path / "toy_eval").exists()
