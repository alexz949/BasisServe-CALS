"""Compose frozen compact V with the existing single-process Private AG."""
from contextlib import ExitStack
from pathlib import Path

import torch

from basisserve.core.qwen35_gated_v_runtime import GatedVRuntime, factor_hash
from basisserve.core.qwen35_gdn_private_ag_runtime import Qwen35PrivateAGRuntime
from basisserve.core.qwen35_full_attention_private_ag_runtime import Qwen35FullAttentionPrivateAGRuntime


class HybridOutputRuntime:
    def __init__(self, model, v_bank, output_bank):
        self.model, self.v_bank = model, v_bank
        self.output_bank = torch.load(output_bank, weights_only=True, map_location='cpu') if isinstance(output_bank, (str, Path)) else output_bank
        assert self.output_bank['format'] == 'basisserve.qwen35.hybrid_output_bank.v1'
        assert self.output_bank['upstream_v_factor_sha256'] == factor_hash(v_bank)
        self.stack = ExitStack()

    def __enter__(self):
        # An inner ExitStack rolls back every installed adapter if a later
        # installation fails; ownership transfers only after all succeed.
        with ExitStack() as pending:
            pending.enter_context(GatedVRuntime(self.model, self.v_bank))
            pending.enter_context(Qwen35PrivateAGRuntime(self.model, self.output_bank['gdn']))
            pending.enter_context(Qwen35FullAttentionPrivateAGRuntime(self.model, self.output_bank['full_attention']))
            self.stack = pending.pop_all()
        return self

    def __exit__(self, *args):
        self.stack.close()

