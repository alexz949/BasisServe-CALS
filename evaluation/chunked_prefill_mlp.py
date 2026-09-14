"""Bound tokenwise MLP intermediates while preserving dispatched forwards."""
import torch


class ChunkedTokenwise(torch.nn.Module):
    def __init__(self, inner, chunk_size=1024):
        super().__init__()
        assert chunk_size > 0
        self.inner = inner
        self.chunk_size = chunk_size

    def forward(self, hidden_states, *tokenwise_inputs):
        if hidden_states.shape[1] <= self.chunk_size:
            return self.inner(hidden_states, *tokenwise_inputs)
        output = torch.empty_like(hidden_states)
        for start in range(0, hidden_states.shape[1], self.chunk_size):
            stop = start + self.chunk_size
            output[:, start:stop] = self.inner(hidden_states[:, start:stop],
                *(value[:, start:stop] for value in tokenwise_inputs))
        return output


def install_chunked_prefill_mlps(model):
    for layer in model.model.layers:
        layer.mlp = ChunkedTokenwise(layer.mlp)


def install_chunked_prefill_norms(model):
    for layer in model.model.layers:
        layer.input_layernorm = ChunkedTokenwise(layer.input_layernorm)
        layer.post_attention_layernorm = ChunkedTokenwise(layer.post_attention_layernorm)
        layer.self_attn.q_norm = ChunkedTokenwise(layer.self_attn.q_norm)
        layer.self_attn.k_norm = ChunkedTokenwise(layer.self_attn.k_norm)
    model.model.norm = ChunkedTokenwise(model.model.norm)
