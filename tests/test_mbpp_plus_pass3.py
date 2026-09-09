from evaluation.tasks.mbpp_plus_pass3 import utils


def test_pass_three_estimator():
    result = utils.aggregate_three([
        {"base_pass_at_1": 1, "plus_pass_at_1": 1},
        {"base_pass_at_1": 1, "plus_pass_at_1": 0},
        {"base_pass_at_1": 0, "plus_pass_at_1": 0},
    ])
    assert result == {"base_pass_at_1": 2/3, "plus_pass_at_1": 1/3,
                      "base_pass_at_3": 1.0, "plus_pass_at_3": 1.0}


def test_all_three_responses_are_scored():
    original = utils.score_one
    seen = []
    def fake_score(doc, results):
        seen.append(results[0])
        return {"base_pass_at_1": 0, "plus_pass_at_1": 0}
    utils.score_one = fake_score
    result = utils.process_results({}, [["a", "b", "c"]])
    utils.score_one = original
    assert seen == ["a", "b", "c"]
    assert result["plus_pass_at_3"] == 0
