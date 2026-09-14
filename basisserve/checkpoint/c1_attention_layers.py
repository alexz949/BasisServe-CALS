"""Locate attention modules without treating recurrent blocks as attention."""


def c1_attention_layers(model):
    kind = model.config.model_type
    assert kind in ('llama', 'qwen3', 'nemotron_h')
    if kind == 'nemotron_h':
        assert len(model.model.layers) == len(model.config.layers_block_type)
        result = []
        for index, (block, block_type) in enumerate(zip(
                model.model.layers, model.config.layers_block_type, strict=True)):
            assert block.block_type == block_type
            if block_type == 'full_attention':
                result.append((index, block.mixer))
    else:
        result = [(index, block.self_attn) for index, block in enumerate(model.model.layers)]
    assert result
    return result
