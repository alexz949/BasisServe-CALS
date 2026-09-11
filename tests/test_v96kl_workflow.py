"""Check layerwise teacher semantics and complete LongBench scoring coverage."""
import torch
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.masking_utils import create_causal_mask

from evaluation.eval_v96kl_longbench import scorer_module, score_sample
from evaluation.prepare_v96kl_data import TASKS
from evaluation.v96kl_common import save_tensors, sha256


def test_layerwise_bf16_teacher_matches_complete_forward():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                      num_attention_heads=4,num_key_value_heads=2,head_dim=8,
                      max_position_embeddings=64,attention_dropout=0.0)
    config._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(config).bfloat16().eval()
    tokens=torch.randint(0,64,(1,37))
    with torch.inference_mode():
        expected=model.model(input_ids=tokens,use_cache=False).last_hidden_state
        hidden=model.model.embed_tokens(tokens)
        positions=torch.arange(tokens.shape[1])[None]
        cos,sin=model.model.rotary_emb(torch.empty(1,dtype=torch.float32),positions)
        mask=create_causal_mask(config=model.config,input_embeds=hidden,attention_mask=None,
            cache_position=positions[0],past_key_values=None,position_ids=positions)
        for layer in model.model.layers:
            hidden=layer(hidden,position_ids=positions,position_embeddings=(cos.bfloat16(),sin.bfloat16()),
                         attention_mask=mask,use_cache=False)
        actual=model.model.norm(hidden)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_every_official_longbench_task_is_included():
    scorer=scorer_module()
    assert len(TASKS)==21 and set(TASKS)==set(scorer.dataset2metric)


def test_official_first_line_preprocessing_is_preserved():
    scorer=scorer_module()
    row=dict(task='trec',answers=['animal'],all_classes=['animal','plant'])
    prediction='\nanimal\nplant'
    score=score_sample(scorer,row,prediction)
    assert score==1.0
    assert 100*score==scorer.scorer('trec',[prediction],[row['answers']],row['all_classes'])


def test_interrupted_tensor_record_can_resume_without_overwriting(tmp_path):
    path=tmp_path/'windows.safetensors'
    values={'input_ids':torch.arange(32,dtype=torch.int32)}
    save_tensors(path,values)
    before=sha256(path)
    save_tensors(path,values)
    assert sha256(path)==before
    with pytest.raises(AssertionError):
        save_tensors(path,{'input_ids':values['input_ids']+1})
    assert sha256(path)==before
