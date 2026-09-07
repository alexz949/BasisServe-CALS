from evaluation.tasks.mbpp_plus_full import utils


def test_code_extraction():
    assert utils.extract_code("```python\ndef f():\n    return 1\n```") == "def f():\n    return 1\n"
    assert utils.extract_code("def f(): return 1") == "def f(): return 1"


def test_scoring_uses_official_base_and_plus_inputs():
    seen = []
    original = utils.score

    def fake_score(task_id, code, kind):
        seen.append((task_id, code, kind))
        return 1.0

    utils.score = fake_score
    doc = {
        "task_id": 2,
        "test_imports": ["import math"],
        "test_list": ["assert math.isclose(f(), 1)"],
        "test": "assert f() == 1\nassert f() != 2",
    }
    result = utils.score_solution("Mbpp/" + str(doc["task_id"]), "def f(): return 1")
    utils.score = original
    assert result == {"base_pass_at_1": 1.0, "plus_pass_at_1": 1.0}
    assert seen[0] == ("Mbpp/2", "def f(): return 1", "base")
    assert seen[1] == ("Mbpp/2", "def f(): return 1", "plus")
