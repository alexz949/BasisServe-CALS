"""longbench_score must equal the official LongBench-v1 eval.py scorer per sample."""
import importlib.util
from pathlib import Path

import pytest

from evaluation.longbench_metrics import (
    DATASET2METRIC, OFFICIAL_DIR, OFFICIAL_MAXLEN, OFFICIAL_PROMPT, longbench_score,
)

SHADOWKV9 = ['narrativeqa', 'multifieldqa_en', 'hotpotqa', 'musique', 'dureader',
             'gov_report', 'samsum', 'passage_retrieval_en', 'lcc']


def official_module():
    # Independent load of the verbatim upstream metrics.py (not the module the package exposes).
    spec = importlib.util.spec_from_file_location('official_metrics_for_test', OFFICIAL_DIR / 'metrics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OFFICIAL = official_module()
# dataset2metric exactly as upstream eval.py binds it, for the nine tasks under test.
EVAL_DATASET2METRIC = {
    'narrativeqa': OFFICIAL.qa_f1_score,
    'multifieldqa_en': OFFICIAL.qa_f1_score,
    'hotpotqa': OFFICIAL.qa_f1_score,
    'musique': OFFICIAL.qa_f1_score,
    'dureader': OFFICIAL.rouge_zh_score,
    'gov_report': OFFICIAL.rouge_score,
    'samsum': OFFICIAL.rouge_score,
    'passage_retrieval_en': OFFICIAL.retrieval_score,
    'lcc': OFFICIAL.code_sim_score,
}


def official_scorer(dataset, predictions, answers, all_classes):
    # Verbatim copy of ``scorer`` from upstream eval.py (revision 2e00731f).
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, EVAL_DATASET2METRIC[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def official_sample_score(task, prediction, answers, all_classes=None):
    # The per-sample body of ``scorer`` without the 100 * mean aggregation.
    if task in ("trec", "triviaqa", "samsum", "lsht"):
        prediction = prediction.lstrip('\n').split('\n')[0]
    score = 0.
    for ground_truth in answers:
        score = max(score, EVAL_DATASET2METRIC[task](prediction, ground_truth, all_classes=all_classes))
    return score


# (task, prediction, answers, expected score) with expected values derived by hand from metrics.py.
CASES = [
    # qa_f1_score: multi-reference max picks the exact second reference (normalisation drops
    # case, punctuation and articles).
    ('narrativeqa', 'the red house', ['a blue car', 'The red house.'], 1.0),
    ('narrativeqa', 'nothing in common', ['a blue car', 'the red house'], 0.0),
    # precision 1/2, recall 1 -> F1 2/3.
    ('multifieldqa_en', 'Paris, France', ['Paris'], 2 / 3),
    ('hotpotqa', 'yes', ['no'], 0.0),
    ('hotpotqa', 'Yes', ['yes'], 1.0),
    # 'Barack Obama' vs 'Obama' -> 2/3; vs 'Barack Hussein Obama' -> 0.8; max = 0.8.
    ('musique', 'Barack Obama', ['Obama', 'Barack Hussein Obama'], 0.8),
    # rouge_zh_score: jieba segmentation then ROUGE-L F on the space-joined tokens.
    ('dureader', '北京是中国的首都', ['中国的首都是北京', '完全不相关的答案'], None),
    ('dureader', '北京是中国的首都', ['北京是中国的首都'], 1.0),
    # rouge_score (ROUGE-L F) on English summaries; identical text scores 1.
    ('gov_report', 'The agency spent the funds on new equipment.',
     ['The agency spent the funds on new equipment.'], 1.0),
    ('gov_report', 'The agency spent the funds on new equipment.',
     ['Funds were spent by the agency on equipment that is new.'], None),
    # samsum newline rule: leading newlines are stripped, then only the first line is scored.
    ('samsum', '\n\nAmy will meet Bob at 5.\nDialogue: Amy: hi\nSummary: nonsense',
     ['Amy will meet Bob at 5.'], 1.0),
    ('samsum', 'Amy will meet Bob at 5.\nAmy will meet Bob at 5.', ['Amy will meet Bob at 5.'], 1.0),
    # retrieval_score: every integer in the prediction must equal the reference paragraph id.
    ('passage_retrieval_en', 'The answer is: Paragraph 12', ['Paragraph 12'], 1.0),
    ('passage_retrieval_en', 'Paragraph 12 or Paragraph 3', ['Paragraph 12'], 0.5),
    ('passage_retrieval_en', 'Paragraph 1', ['Paragraph 12'], 0.0),
    ('passage_retrieval_en', 'I am not sure.', ['Paragraph 12'], 0.0),
    # code_sim_score: the first line without a backtick, '#' or '//' is compared by fuzz.ratio.
    ('lcc', '```python\n# compute the sum\nreturn a + b\n```', ['return a + b'], 1.0),
    ('lcc', '```python\n# compute the sum\n```', ['return a + b'], 0.0),
    ('lcc', '\n\nreturn a + b  // done\nreturn a - b', ['return a + b'], None),
]


@pytest.mark.parametrize('task,prediction,answers,expected', CASES)
def test_matches_official_scorer(task, prediction, answers, expected):
    score = longbench_score(task, prediction, answers)
    assert isinstance(score, float) and 0.0 <= score <= 1.0
    assert score == official_sample_score(task, prediction, answers)
    assert round(100 * score, 2) == official_scorer(task, [prediction], [answers], None)
    if expected is not None:
        assert score == pytest.approx(expected)


def test_every_shadowkv9_task_is_covered():
    assert {case[0] for case in CASES} == set(SHADOWKV9)


def test_samsum_first_line_rule_changes_the_score():
    prediction = '\n\nAmy will meet Bob at 5.\nDialogue: Amy: hi\nSummary: nonsense'
    answers = ['Amy will meet Bob at 5.']
    # The rouge package's F formula divides by (r + p + 1e-8), so identical text scores 1 - 5e-9.
    assert longbench_score('samsum', prediction, answers) == pytest.approx(1.0)
    # The same text scored as a gov_report summary (no first-line rule) is penalised.
    assert longbench_score('gov_report', prediction, answers) < 1.0
    assert longbench_score('gov_report', prediction, answers) == OFFICIAL.rouge_score(prediction, answers[0])


def test_dureader_uses_jieba_segmentation():
    prediction, answer = '北京是中国的首都', '中国的首都是北京'
    score = longbench_score('dureader', prediction, [answer])
    assert 0.0 < score < 1.0
    assert score == OFFICIAL.rouge_zh_score(prediction, answer)
    # Without jieba, ROUGE sees a single token on each side and the score differs.
    assert score != OFFICIAL.rouge_score(prediction, answer)


def test_official_tables():
    assert len(DATASET2METRIC) == 21
    assert set(DATASET2METRIC) == set(OFFICIAL_PROMPT) == set(OFFICIAL_MAXLEN)
    assert set(DATASET2METRIC.values()) == {
        'qa_f1_score', 'rouge_score', 'rouge_zh_score', 'retrieval_score', 'retrieval_zh_score',
        'code_sim_score', 'classification_score', 'count_score', 'qa_f1_zh_score'}
    assert {task: OFFICIAL_MAXLEN[task] for task in SHADOWKV9} == {
        'narrativeqa': 128, 'multifieldqa_en': 64, 'hotpotqa': 32, 'musique': 32, 'dureader': 128,
        'gov_report': 512, 'samsum': 128, 'passage_retrieval_en': 32, 'lcc': 64}
    for task in SHADOWKV9:
        assert '{context}' in OFFICIAL_PROMPT[task]
        assert hasattr(OFFICIAL, DATASET2METRIC[task])
    assert (OFFICIAL_DIR / 'README.md').exists() and Path(OFFICIAL.__file__).parent == OFFICIAL_DIR
