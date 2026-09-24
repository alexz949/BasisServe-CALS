"""Official LongBench-v1 scoring for one sample.

``longbench_official/`` holds verbatim copies of the upstream ``metrics.py`` and
prompt / generation-length tables (see its README for the revision and hashes).
``longbench_score`` reproduces the per-sample part of ``scorer`` from upstream
``eval.py``; upstream reports ``100 * mean`` over samples, this keeps the
per-sample score in [0, 1].
"""
import importlib.util
import json
from pathlib import Path

OFFICIAL_DIR = Path(__file__).resolve().parent / 'longbench_official'
LONGBENCH_REPO_REVISION = '2e00731f8d0bff23dc4325161044d0ed8af94c1e'

# Metric function name per dataset, exactly as ``dataset2metric`` in upstream eval.py.
DATASET2METRIC = {
    'narrativeqa': 'qa_f1_score',
    'qasper': 'qa_f1_score',
    'multifieldqa_en': 'qa_f1_score',
    'multifieldqa_zh': 'qa_f1_zh_score',
    'hotpotqa': 'qa_f1_score',
    '2wikimqa': 'qa_f1_score',
    'musique': 'qa_f1_score',
    'dureader': 'rouge_zh_score',
    'gov_report': 'rouge_score',
    'qmsum': 'rouge_score',
    'multi_news': 'rouge_score',
    'vcsum': 'rouge_zh_score',
    'trec': 'classification_score',
    'triviaqa': 'qa_f1_score',
    'samsum': 'rouge_score',
    'lsht': 'classification_score',
    'passage_retrieval_en': 'retrieval_score',
    'passage_count': 'count_score',
    'passage_retrieval_zh': 'retrieval_zh_score',
    'lcc': 'code_sim_score',
    'repobench-p': 'code_sim_score',
}

# Upstream eval.py keeps only the first non-empty line of the prediction for these tasks.
FIRST_LINE_TASKS = ('trec', 'triviaqa', 'samsum', 'lsht')


def load_official_metrics():
    spec = importlib.util.spec_from_file_location('longbench_official_metrics', OFFICIAL_DIR / 'metrics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


official_metrics = load_official_metrics()
OFFICIAL_PROMPT = json.loads((OFFICIAL_DIR / 'dataset2prompt.json').read_text(encoding='utf-8'))
OFFICIAL_MAXLEN = json.loads((OFFICIAL_DIR / 'dataset2maxlen.json').read_text(encoding='utf-8'))
assert set(DATASET2METRIC) == set(OFFICIAL_PROMPT) == set(OFFICIAL_MAXLEN)


def longbench_score(task, prediction, answers, all_classes=None):
    metric = getattr(official_metrics, DATASET2METRIC[task])
    if task in FIRST_LINE_TASKS:
        prediction = prediction.lstrip('\n').split('\n')[0]
    score = 0.0
    for ground_truth in answers:
        score = max(score, metric(prediction, ground_truth, all_classes=all_classes))
    return float(score)
