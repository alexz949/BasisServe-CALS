"""Observe LRQK after selection; exact-QK diagnostic scores never feed routing."""
import json
from pathlib import Path
import sys

import torch
from basisserve.core.c1_lrqk import LRQKState
from evaluation import eval_qwen35_longbench_v2 as evaluator


def main():
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    output.mkdir(parents=True, exist_ok=True)
    original_init, original_decode = LRQKState.__init__, LRQKState.decode
    trace = (output / 'selection_audit.jsonl').open('a', buffering=1)
    state_count = 0

    def init(self, q, k, config, layer=0):
        nonlocal state_count
        original_init(self, q, k, config, layer=layer)
        self.audit_layer = (3, 7, 11, 15, 19, 23, 27, 31)[state_count % 8]
        self.audit_stage = state_count // 8
        state_count += 1

    @torch.inference_mode()
    def decode(self, q, k, v, scale):
        result = original_decode(self, q, k, v, scale)
        if self.steps not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1023):
            return result
        # Read-only diagnostic AFTER the method has fixed its selected set and output.
        batch, heads, _, dim = q.shape
        groups, length = k.shape[1:3]
        scores = (q[:, :, 0].float().reshape(batch, groups, heads//groups, dim)
                  @ k.float().transpose(-1, -2)).reshape(batch, heads, length)
        oracle = scores.topk(min(2048, length), dim=-1).indices
        selected_mask = torch.zeros_like(scores, dtype=torch.bool)
        selected_mask.scatter_(-1, self.selected, True)
        recall = selected_mask.gather(-1, oracle).float().mean(-1)
        recent_start = max(0, length-self.config.recent)
        recent_count = (self.selected >= recent_start).sum(-1)
        record = dict(layer=self.audit_layer, stage=self.audit_stage, step=self.steps, length=length,
            statistics=self.statistics(groups), q_dtype=str(q.dtype), code_dtype=str(self.ak.dtype),
            exact_top2048_recall_including_recent=recall.cpu().tolist(),
            recent_count=recent_count.cpu().tolist(),
            note='sampled decode steps; exact scores computed only after original decode returns')
        trace.write(json.dumps(record)+'\n')
        return result

    LRQKState.__init__, LRQKState.decode = init, decode
    evaluator.main()
    trace.close()


if __name__ == '__main__':
    main()
