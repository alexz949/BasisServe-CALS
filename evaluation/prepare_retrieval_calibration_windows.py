"""Generic synthetic long-range retrieval calibration windows (default 128K; --sequence-length overrides).

Each window is a natural-text C4 haystack with synthetic records inserted at
controlled token positions across eight positional strata, followed by a tail of
retrieval questions with deterministic answers. Records, questions and answers
use independently written templates and fresh random identifiers; nothing here
imports or copies an evaluation generator. Segments are tokenized separately and
joined at the token level, matching the project's separator-free packing, so
every window is exactly the requested sequence length with only C4 filler ever trimmed.

Task families over 16 windows: 8 multi-key retrieval, 4 multi-value retrieval,
2 variable tracking, 2 aggregation.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import string
import sys

from datasets import load_dataset
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.v96kl_common import read_json, write_json, save_tensors, sha256

SEQUENCE_LENGTH = 131072
STRATA = 8
TAIL_MINIMUM, TAIL_MAXIMUM = 9216, 13312
INTERACTION_CAP = 96
FILLER_MINIMUM, FILLER_MAXIMUM = 256, 2048
DATASET_REVISION = '1588ec454efa1a09f29cd18ddd04fe05fc8653a2'
FAMILIES = ['multikey'] * 8 + ['multivalue'] * 4 + ['tracking'] * 2 + ['aggregation'] * 2

RECORD_TEMPLATES = (
    'The registry entry for {k} is {v}.',
    'Identifier {k} has authorization code {v}.',
    'Record: asset {k} maps to token {v}.',
    'For reference, {k} was assigned the code {v}.',
    'Ledger line: {k} carries {v}.',
    'Note that the credential attached to {k} reads {v}.',
)
ATTRIBUTE_TEMPLATES = (
    'Entity {e}: its {a} is {v}.',
    'The {a} recorded for entity {e} is {v}.',
    'Filed under {e}, the {a} field holds {v}.',
    'Entity {e} lists {v} as its {a}.',
)
POINTER_TEMPLATES = (
    'Pointer {a} references {b}.',
    'Alias {a} forwards to {b}.',
    'The label {a} resolves to {b}.',
)
TERMINAL_TEMPLATES = (
    'Pointer {a} holds the value {v}.',
    'Alias {a} stores {v}.',
    'The label {a} terminates at {v}.',
)
MARKER_TEMPLATES = (
    'Marker {m} observed.',
    'Log entry: {m} was recorded here.',
    'Signal {m} flagged.',
)
ATTRIBUTES = ('primary code', 'secondary code', 'checksum', 'revision tag')
QUERY_MULTIKEY = (
    'Retrieve the codes associated with {keys}, in that order.',
    'Report the values recorded for {keys}, keeping that order.',
    'Which codes belong to {keys}? List them in order.',
    'Provide the entries for {keys}, in the order given.',
)
QUERY_SINGLE = (
    'Which code belongs to {keys}?',
    'What is the entry recorded for {keys}?',
)
QUERY_MULTIVALUE = (
    'For entity {e}, provide the {attrs}, in that order.',
    'List the {attrs} of entity {e}, in the order given.',
    'What are the {attrs} recorded for entity {e}? Keep that order.',
)
QUERY_TRACKING = (
    'Starting from {a}, what value is finally reached?',
    'Follow {a} to its end. Which value does it reach?',
    'What value does the chain beginning at {a} arrive at?',
)
QUERY_HOP = (
    'What does {a} reference directly?',
    'Which label does {a} point to?',
)
QUERY_COUNT = (
    'How many times was {m} observed?',
    'Report the number of occurrences of {m}.',
)
QUERY_MOST = (
    'Which of {ms} appears most often?',
    'Among {ms}, which marker was recorded the most times?',
)
QUERY_COUNTS = (
    'Report the counts for {ms}, in that order.',
    'Give the occurrence counts of {ms}, keeping that order.',
)


def join_names(items):
    return ', '.join(items[:-1]) + ' and ' + items[-1] if len(items) > 1 else items[0]


class Identifiers:
    """Opaque high-entropy identifiers, unique across a window and absent from its haystack."""

    def __init__(self, rng, haystack_text):
        self.rng, self.haystack, self.used = rng, haystack_text, set()

    def draw(self, kind):
        while True:
            if kind == 'key':
                value = ''.join(self.rng.choices(string.ascii_uppercase, k=4)) + '-' + ''.join(self.rng.choices(string.digits, k=4))
            elif kind == 'value':
                value = ''.join(self.rng.choices(string.ascii_uppercase, k=2)) + '-' + ''.join(self.rng.choices('0123456789ABCDEF', k=6))
            elif kind == 'entity':
                value = 'ENT-' + ''.join(self.rng.choices(string.ascii_uppercase + string.digits, k=5))
            elif kind == 'pointer':
                value = 'P' + ''.join(self.rng.choices(string.ascii_uppercase, k=2)) + ''.join(self.rng.choices(string.digits, k=3))
            else:
                value = 'TAG-' + ''.join(self.rng.choices(string.ascii_uppercase + string.digits, k=3))
            if value not in self.used and value not in self.haystack:
                self.used.add(value)
                return value


def plan_records(family, rng, ids):
    """Return (records, questions). Records carry text, stratum and support role;
    questions carry text, answer, kind and the record indices they depend on."""
    records, questions = [], []

    def add_record(text, stratum, **fields):
        records.append(dict(text=text, stratum=stratum, **fields))
        return len(records) - 1

    if family == 'multikey':
        pairs = [(ids.draw('key'), ids.draw('value')) for _ in range(64)]
        for index, (key, value) in enumerate(pairs):
            add_record(rng.choice(RECORD_TEMPLATES).format(k=key, v=value), index % STRATA, key=key, value=value)
        while len(questions) < INTERACTION_CAP:
            if rng.random() < 0.2:
                pick = [rng.randrange(len(pairs))]
                text = rng.choice(QUERY_SINGLE).format(keys=pairs[pick[0]][0])
                kind = 'single-key'
            else:
                count = rng.randint(2, 5)
                strata = rng.sample(range(STRATA), count)
                pick = [rng.choice([i for i in range(len(pairs)) if i % STRATA == s]) for s in strata]
                keys = [pairs[i][0] for i in pick]
                text = rng.choice(QUERY_MULTIKEY).format(keys=join_names(keys))
                kind = 'multi-key'
            questions.append(dict(text=text, answer=answer_for(kind, [records[i] for i in pick]), kind=kind, support=pick))
    elif family == 'multivalue':
        entities = [ids.draw('entity') for _ in range(24)]
        table = {}
        for index, entity in enumerate(entities):
            attributes = rng.sample(ATTRIBUTES, rng.randint(3, 4))
            strata = rng.sample(range(STRATA), len(attributes))
            table[entity] = {}
            for attribute, stratum in zip(attributes, strata):
                value = ids.draw('value')
                table[entity][attribute] = (value, add_record(
                    rng.choice(ATTRIBUTE_TEMPLATES).format(e=entity, a=attribute, v=value),
                    stratum, entity=entity, attribute=attribute, value=value))
        while len(questions) < INTERACTION_CAP:
            entity = rng.choice(entities)
            attributes = rng.sample(list(table[entity]), rng.randint(2, len(table[entity])))
            text = rng.choice(QUERY_MULTIVALUE).format(e=entity, attrs=join_names(attributes))
            support = [table[entity][a][1] for a in attributes]
            questions.append(dict(text=text, answer=answer_for('multi-value', [records[i] for i in support]),
                                  kind='multi-value', support=support))
    elif family == 'tracking':
        chains = []
        for _ in range(20):
            length = rng.randint(3, 5)
            names = [ids.draw('pointer') for _ in range(length)]
            value = ids.draw('value')
            strata = rng.sample(range(STRATA), length)
            indices = []
            for step in range(length):
                if step + 1 < length:
                    text = rng.choice(POINTER_TEMPLATES).format(a=names[step], b=names[step + 1])
                    indices.append(add_record(text, strata[step], pointer=names[step], target=names[step + 1]))
                else:
                    text = rng.choice(TERMINAL_TEMPLATES).format(a=names[step], v=value)
                    indices.append(add_record(text, strata[step], pointer=names[step], value=value))
            chains.append(dict(names=names, value=value, records=indices))
        while len(questions) < INTERACTION_CAP:
            chain = rng.choice(chains)
            if rng.random() < 0.25:
                hop = rng.randrange(len(chain['names']) - 1)
                text = rng.choice(QUERY_HOP).format(a=chain['names'][hop])
                questions.append(dict(text=text, answer=answer_for('tracking-hop', [records[chain['records'][hop]]]),
                                      kind='tracking-hop', support=[chain['records'][hop]]))
            else:
                text = rng.choice(QUERY_TRACKING).format(a=chain['names'][0])
                questions.append(dict(text=text, answer=answer_for('tracking', [records[i] for i in chain['records']]),
                                      kind='tracking', support=list(chain['records'])))
    else:
        markers = [ids.draw('marker') for _ in range(6)]
        counts = {marker: rng.randint(3, 9) for marker in markers}
        occurrences = collections.defaultdict(list)
        for marker in markers:
            for _ in range(counts[marker]):
                occurrences[marker].append(add_record(
                    rng.choice(MARKER_TEMPLATES).format(m=marker), rng.randrange(STRATA), marker=marker))
        while len(questions) < INTERACTION_CAP:
            choice = rng.random()
            if choice < 0.4:
                marker = rng.choice(markers)
                text = rng.choice(QUERY_COUNT).format(m=marker)
                questions.append(dict(text=text, answer=answer_for('aggregation-count', [records[i] for i in occurrences[marker]]),
                                      kind='aggregation-count', support=list(occurrences[marker])))
            elif choice < 0.7:
                subset = rng.sample(markers, 3)
                best = max(subset, key=lambda m: counts[m])
                if sum(counts[m] == counts[best] for m in subset) > 1:
                    continue
                text = rng.choice(QUERY_MOST).format(ms=join_names(subset))
                support = [i for m in subset for i in occurrences[m]]
                questions.append(dict(text=text, answer=answer_for('aggregation-most', [records[i] for i in support]),
                                      kind='aggregation-most', support=support))
            else:
                subset = rng.sample(markers, rng.randint(2, 3))
                text = rng.choice(QUERY_COUNTS).format(ms=join_names(subset))
                support = [i for m in subset for i in occurrences[m]]
                questions.append(dict(text=text, answer=answer_for('aggregation-counts', [records[i] for i in support]),
                                      kind='aggregation-counts', support=support))
    return records, questions


def answer_for(kind, support):
    """Deterministic answer from the supporting records alone; also validation B."""
    if kind == 'single-key':
        return f"The code recorded for {support[0]['key']} is {support[0]['value']}."
    if kind == 'multi-key':
        return ('The codes are ' + ', '.join(r['value'] for r in support) + '; they belong to '
                + ', '.join(r['key'] for r in support) + ' respectively.')
    if kind == 'multi-value':
        return ('The values are ' + ', '.join(r['value'] for r in support) + ', which are the '
                + ', '.join(r['attribute'] for r in support) + f" of entity {support[0]['entity']} in that order.")
    if kind == 'tracking-hop':
        return f"{support[0]['pointer']} references {support[0]['target']} directly."
    if kind == 'tracking':
        chain = ' -> '.join(r['pointer'] for r in support)
        return f"Starting from {support[0]['pointer']}, the chain runs {chain} and ends at the value {support[-1]['value']}."
    counts = collections.Counter(r['marker'] for r in support)
    order = []
    for r in support:
        if r['marker'] not in order:
            order.append(r['marker'])
    if kind == 'aggregation-count':
        return f'{order[0]} was observed {counts[order[0]]} times in the records above.'
    if kind == 'aggregation-most':
        best = max(order, key=lambda m: counts[m])
        return ('Of ' + ', '.join(order) + f', the marker {best} appears most often, with {counts[best]} occurrences.')
    return ('The counts are ' + ', '.join(str(counts[m]) for m in order) + ' for '
            + ', '.join(order) + ' respectively.')


class Haystack:
    """Streams C4 filler excerpts disjoint from the excluded document hashes."""

    def __init__(self, tokenizer, seed, excluded):
        self.tokenizer, self.excluded, self.seen = tokenizer, set(excluded), set()
        self.stream = iter(load_dataset('allenai/c4', 'en', split='train', streaming=True,
                                        revision=DATASET_REVISION).shuffle(seed=seed, buffer_size=10000))
        self.rng = random.Random(seed)
        self.index = -1

    def excerpt(self):
        while True:
            self.index += 1
            row = next(self.stream)
            document = hashlib.sha256(row['text'].encode()).hexdigest()
            if document in self.excluded or document in self.seen:
                continue
            ids = self.tokenizer(row['text'], add_special_tokens=False)['input_ids']
            if len(ids) < FILLER_MINIMUM:
                continue
            self.seen.add(document)
            length = min(len(ids), self.rng.randint(FILLER_MINIMUM, FILLER_MAXIMUM))
            start = self.rng.randint(0, len(ids) - length)
            return document, self.index, ids[start:start + length]


def build_window(sequence_id, family, seed, tokenizer, haystack):
    rng = random.Random(seed)
    encode = lambda text: tokenizer(text, add_special_tokens=False)['input_ids']  # noqa: E731
    separator = encode('\n\n')
    # Reserve the haystack first so identifiers can be checked against its text.
    body_target = SEQUENCE_LENGTH - TAIL_MINIMUM
    fillers, filler_text, documents = [], [], []
    filled = 0
    while filled < body_target + FILLER_MAXIMUM:
        document, stream_index, ids = haystack.excerpt()
        fillers.append(ids)
        filler_text.append(tokenizer.decode(ids))
        documents.append(dict(document_sha256=document, stream_index=stream_index, tokens=len(ids)))
        filled += len(ids) + len(separator)
    ids = Identifiers(rng, '\n'.join(filler_text))
    records, questions = plan_records(family, rng, ids)
    # Tail: header, then question/answer pairs until the tail budget is met.
    tail_lines = [encode('\n\nRetrieval questions about the records above follow. Each answer is exact.\n')]
    kept = []
    tail_tokens = len(tail_lines[0])
    for question in questions:
        line = encode(f"\nQuestion: {question['text']}\nAnswer: {question['answer']}\n")
        if tail_tokens + len(line) > TAIL_MAXIMUM:
            break
        tail_lines.append(line)
        kept.append(question)
        tail_tokens += len(line)
        if tail_tokens >= TAIL_MINIMUM and len(kept) >= 48:
            break
    assert tail_tokens <= TAIL_MAXIMUM and 48 <= len(kept) <= INTERACTION_CAP
    body_budget = SEQUENCE_LENGTH - tail_tokens
    # Records get a target position inside their stratum of the body; the body is
    # then assembled by walking filler excerpts and dropping each record in as soon
    # as its target is passed, so it lands in its stratum, never the tail.
    stratum_width = body_budget / STRATA
    for record in records:
        record['target_position'] = int((record['stratum'] + rng.random()) * stratum_width)
        record['ids'] = encode('\n' + record['text'] + '\n')
    order = sorted(range(len(records)), key=lambda i: records[i]['target_position'])
    body, pending, used_fillers = [], list(order), 0
    reserved = sum(len(records[i]['ids']) for i in order)
    for ids_ in fillers:
        while pending and len(body) >= records[pending[0]]['target_position']:
            i = pending.pop(0)
            records[i]['token_start'] = len(body)
            body.extend(records[i]['ids'])
            records[i]['token_end'] = len(body)
            reserved -= len(records[i]['ids'])
        if len(body) + len(ids_) + len(separator) + reserved >= body_budget:
            break
        body.extend(ids_)
        body.extend(separator)
        used_fillers += 1
    while pending:
        i = pending.pop(0)
        records[i]['token_start'] = len(body)
        body.extend(records[i]['ids'])
        records[i]['token_end'] = len(body)
    # Close the body exactly with C4 filler only: trim the next excerpt, drawing
    # fresh excerpts if the reserved pool is exhausted.
    while len(body) < body_budget:
        if used_fillers == len(fillers):
            document, stream_index, extra = haystack.excerpt()
            fillers.append(extra)
            documents.append(dict(document_sha256=document, stream_index=stream_index, tokens=len(extra)))
        piece = separator + fillers[used_fillers]
        used_fillers += 1
        body.extend(piece[:body_budget - len(body)])
    assert len(body) == body_budget
    tail_start = len(body)
    positions = []
    for line, question in zip(tail_lines[1:], kept):
        cursor = tail_start + sum(len(x) for x in tail_lines[:1 + len(positions)])
        supports = [records[i]['token_start'] for i in question['support']]
        positions.append(dict(text=question['text'], answer=question['answer'], kind=question['kind'],
                              query_token_position=cursor, support_token_positions=supports,
                              distances=[cursor - s for s in supports], support_records=question['support']))
    tokens = body + [t for line in tail_lines for t in line]
    assert len(tokens) == SEQUENCE_LENGTH
    metadata = dict(
        sequence_id=sequence_id, seed=seed, task_family=family, tail_start=tail_start,
        tail_tokens=tail_tokens, interactions=len(kept), c4_documents=documents[:used_fillers],
        records=[{k: v for k, v in r.items() if k != 'ids'} for r in records], queries=positions)
    return torch.tensor(tokens, dtype=torch.int32), metadata


def validate(windows, rows, tokenizer):
    report = dict(length_ok=True, answers_ok=True, leakage_ok=True, collision_ok=True,
                  distance_histogram=collections.Counter(), records_per_stratum=collections.Counter(),
                  task_mix=collections.Counter(), interactions=[], tail_tokens=[])
    for window, row in zip(windows, rows):
        report['length_ok'] &= int(window.shape[0]) == SEQUENCE_LENGTH
        text = tokenizer.decode(window.tolist())
        records = row['records']
        for query in row['queries']:
            regenerated = answer_for(query['kind'], [records[i] for i in query['support_records']])
            if regenerated != query['answer']:
                report['answers_ok'] = False
                report.setdefault('answer_mismatches', []).append(
                    dict(sequence=row['sequence_id'], kind=query['kind'], stored=query['answer'], regenerated=regenerated))
            report['task_mix'][query['kind']] += 1
            for distance in query['distances']:
                bucket = 'short<16K' if distance < 16384 else 'medium<48K' if distance < 49152 else 'long<96K' if distance < 98304 else 'very-long>=96K'
                report['distance_histogram'][bucket] += 1
        for record in records:
            report['records_per_stratum'][record['stratum']] += 1
            report['length_ok'] &= record['token_end'] <= row['tail_start']
        values = [r['value'] for r in records if 'value' in r]
        keys = [r.get('key') or r.get('entity') or r.get('pointer') for r in records if 'value' in r or 'target' in r]
        if row['task_family'] != 'multivalue':
            report['collision_ok'] &= len(values) == len(set(values))
        if row['task_family'] == 'multikey':
            report['collision_ok'] &= len(keys) == len(set(keys))
        for query in row['queries']:
            for i in query['support_records']:
                value = records[i].get('value')
                if value is None:
                    continue
                report['leakage_ok'] &= value not in query['text']
                expected = sum(1 for r in records if r.get('value') == value) + sum(1 for q in row['queries'] if value in q['answer'])
                report['leakage_ok'] &= text.count(value) == expected
        if row['task_family'] == 'aggregation':
            counts = collections.Counter(r['marker'] for r in records)
            body = tokenizer.decode(window[:row['tail_start']].tolist())
            report['leakage_ok'] &= all(body.count(m) == n for m, n in counts.items())
        report['interactions'].append(row['interactions'])
        report['tail_tokens'].append(row['tail_tokens'])
    report['distance_histogram'] = dict(report['distance_histogram'])
    report['records_per_stratum'] = dict(sorted(report['records_per_stratum'].items()))
    report['task_mix'] = dict(report['task_mix'])
    return report


def main():
    global SEQUENCE_LENGTH, TAIL_MINIMUM, TAIL_MAXIMUM
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exclude-manifest', type=Path, action='append', default=[],
                   help='C4 window manifests whose documents must not appear in the haystack')
    p.add_argument('--seed', type=int, default=20260922)
    p.add_argument('--haystack-seed', type=int, default=20260923)
    p.add_argument('--limit', type=int, help='Smoke test: build only the first N windows')
    p.add_argument('--families', help='Smoke test: comma-separated family list overriding the 16-window mix')
    p.add_argument('--sequence-length', type=int, default=SEQUENCE_LENGTH,
                   help='window length in tokens; the tail range scales with it (9216-13312 at 131072)')
    args = p.parse_args()
    scale = args.sequence_length / SEQUENCE_LENGTH
    TAIL_MINIMUM, TAIL_MAXIMUM = int(TAIL_MINIMUM * scale), int(TAIL_MAXIMUM * scale)
    SEQUENCE_LENGTH = args.sequence_length
    families = args.families.split(',') if args.families else FAMILIES
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    excluded = set()
    for manifest in args.exclude_manifest:
        excluded.update(r['document_sha256'] for r in read_json(manifest)['records'])
    haystack = Haystack(tokenizer, args.haystack_seed, excluded)
    windows, rows = [], []
    for sequence_id, family in enumerate(families[:args.limit]):
        window, row = build_window(sequence_id, family, args.seed * 1000 + sequence_id, tokenizer, haystack)
        windows.append(window)
        rows.append(row)
        print(f'window {sequence_id} {family}: tail {row["tail_tokens"]} tokens, {row["interactions"]} interactions, '
              f'{len(row["records"])} records, {len(row["c4_documents"])} C4 documents', flush=True)
    report = validate(windows, rows, tokenizer)
    assert report['length_ok'] and report['answers_ok'] and report['leakage_ok'] and report['collision_ok'], report
    args.output.mkdir(parents=True, exist_ok=True)
    packed = torch.stack(windows)
    save_tensors(args.output / 'windows.safetensors', dict(input_ids=packed))
    with (args.output / 'metadata.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    write_json(args.output / 'summary.json', report)
    write_json(args.output / 'manifest.json', dict(
        status='complete', sha256=sha256(args.output / 'windows.safetensors'),
        metadata_sha256=sha256(args.output / 'metadata.jsonl'), shape=list(packed.shape),
        method='generic synthetic long-range retrieval calibration: C4 haystack excerpts joined at the token '
               'level with a two-newline separator; synthetic records inserted at seeded positions across '
               'eight equal strata of the body; a tail of question/answer lines; identifiers drawn from a '
               'seeded RNG and rejected if present in the haystack; only C4 filler is trimmed to reach '
               f'exactly {SEQUENCE_LENGTH} tokens', sequence_length=SEQUENCE_LENGTH,
        families=families[:args.limit], seed=args.seed, haystack_seed=args.haystack_seed,
        window_seed_rule='seed * 1000 + sequence_id',
        model=str(args.model.resolve()), model_config_sha256=sha256(args.model / 'config.json'),
        tokenizer_sha256=sha256(args.model / 'tokenizer.json'),
        dataset='allenai/c4', dataset_config='en', dataset_split='train', dataset_revision=DATASET_REVISION,
        excluded_manifests=[str(m.resolve()) for m in args.exclude_manifest], excluded_documents=len(excluded),
        tail_range=[TAIL_MINIMUM, TAIL_MAXIMUM], strata=STRATA, validation=report,
        source_sha256=sha256(Path(__file__))))
    print('retrieval calibration windows complete', tuple(packed.shape), flush=True)


if __name__ == '__main__':
    main()
