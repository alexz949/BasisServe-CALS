"""Three independently sampled solutions, scored with the existing EvalPlus oracle."""

from evaluation.tasks.mbpp_plus_full.utils import process_results as score_one


def aggregate_three(scores):
    assert len(scores) == 3
    result = {}
    for kind in ("base", "plus"):
        successes = sum(row[kind + "_pass_at_1"] for row in scores)
        result[kind + "_pass_at_1"] = successes / 3
        result[kind + "_pass_at_3"] = float(successes > 0)
    return result


def process_results(doc, results):
    assert len(results) == 1 and len(results[0]) == 3
    return aggregate_three([score_one(doc, [text]) for text in results[0]])
