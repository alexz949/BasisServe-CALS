from evaluation.diagnose_qwen35_gsm8k_generation import aligned_stop_settings


def test_distinct_native_and_chat_eos_and_task_stops_are_preserved():
    stops = aligned_stop_settings(248044, 248046, '<|im_end|>', ['Question:', '</s>', '<|im_end|>'])
    assert stops['eos_token_id'] == [248044, 248046]
    assert stops['stop_strings'] == ['Question:', '</s>', '<|im_end|>']
    assert '</think>' not in stops['stop_strings']
