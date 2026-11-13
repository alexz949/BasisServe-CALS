import numpy as np

from evaluation import summarize_v64_seeds as summary


def _examples(seed, *, flip=()):
    generator = np.random.default_rng(seed)
    examples = {}
    for task in summary.TASKS:
        records = []
        for doc_id in range(40):
            record = {
                "task": task,
                "doc_id": doc_id,
                "doc_hash": f"{task}-{doc_id}",
                "metric": "acc",
                "native": {"source_id": f"source-{doc_id // 2}"},
                "passage_sha256": f"passage-{doc_id // 4}",
                "correct": float(generator.random() < 0.7),
            }
            if (task, doc_id) in flip:
                record["correct"] = 1.0 - record["correct"]
            records.append(record)
        examples[task] = records
    return examples


def test_paired_bootstrap_of_a_result_against_itself_is_exactly_zero():
    arms = {"a": _examples(0), "copy_of_a": _examples(0), "b": _examples(1)}
    sizes, correct = summary.unit_tables(arms)
    macro = summary.bootstrap_macro(sizes, correct, replicates=200, seed=7)
    assert np.array_equal(macro["a"], macro["copy_of_a"])
    assert summary.interval(macro["a"] - macro["copy_of_a"]) == [0.0, 0.0]
    assert np.any(macro["a"] != macro["b"])


def test_point_estimate_and_group_units():
    arms = {"a": _examples(0)}
    sizes, correct = summary.unit_tables(arms)
    accuracy, macro = summary.point_macro(sizes, correct["a"])
    for task in summary.TASKS:
        expected = np.mean([record["correct"] for record in arms["a"][task]])
        assert abs(accuracy[task] - expected) < 1e-12
    assert abs(macro - np.mean(list(accuracy.values()))) < 1e-12
    assert len(sizes["boolq"]) == 10 and len(sizes["hellaswag"]) == 20
    assert len(sizes["piqa"]) == 40


def test_bootstrap_is_reproducible_and_brackets_the_point_estimate():
    arms = {"a": _examples(0)}
    sizes, correct = summary.unit_tables(arms)
    first = summary.bootstrap_macro(sizes, correct, replicates=500, seed=3)["a"]
    again = summary.bootstrap_macro(sizes, correct, replicates=500, seed=3)["a"]
    assert np.array_equal(first, again)
    low, high = summary.interval(first)
    assert low < summary.point_macro(sizes, correct["a"])[1] < high


def test_unpaired_examples_are_rejected():
    changed = _examples(0)
    changed["piqa"][3]["doc_hash"] = "another document"
    assert summary.paired_examples({"a": _examples(0), "b": _examples(1)})
    assert not summary.paired_examples({"a": _examples(0), "b": changed})
    shorter = _examples(0)
    shorter["boolq"].pop()
    assert not summary.paired_examples({"a": _examples(0), "b": shorter})
