import json

from benchmarks.system.run_tp8_joint_batch_sweep import (
    ARMS, BATCHES, MODELS, batches_by_context, command_for, failure_status,
)


def test_grid_has_requested_non_power_of_two_batches():
    assert BATCHES == (1, 2, 4, 6, 8, 10, 12, 14, 16)
    assert len(MODELS) * len(ARMS) * len(BATCHES) * 2 == 72


def test_commands_use_same_prompt_for_each_pair(tmp_path):
    for model in MODELS:
        commands = [command_for(model, arm, 130048, 14, tmp_path, 16, 128) for arm in ARMS]
        for command in commands:
            assert command[command.index('--batch') + 1] == '14'
            assert command[command.index('--measure-steps') + 1] == '128'
            assert '--nproc-per-node=8' in command
        token_files = [c[c.index('--tokens') + 1] for c in commands]
        assert token_files[0] == token_files[1]


def test_failure_classification_keeps_oom_phase(tmp_path):
    (tmp_path / 'rank0.log').write_text(json.dumps({'status': 'model_ready'}) + '\n')
    result = failure_status('torch.OutOfMemoryError: CUDA out of memory', tmp_path)
    assert result['status'] == 'gpu_oom'
    assert result['possible_failure_phases'] == ['prefill']
    assert failure_status('unrelated failure', tmp_path)['status'] == 'failed'


def test_qwen8_post_grid_and_inputs(tmp_path):
    batches = batches_by_context([65536, 130048], BATCHES, 8)
    assert batches['65536'] == list(BATCHES)
    assert batches['130048'] == [1, 2, 4, 6, 8]
    assert sum(map(len, batches.values())) * len(ARMS) == 28
    commands = [command_for('qwen8', arm, 130048, 8, tmp_path, 16, 128) for arm in ARMS]
    for command in commands:
        assert 'models--Qwen--Qwen3-8B/snapshots/' in command[command.index('--model') + 1]
        assert 'post_uniform_v96' in command[command.index('--router-root') + 1]
        assert '--prompt-manifest' not in command
    assert commands[0][commands[0].index('--tokens') + 1] == commands[1][commands[1].index('--tokens') + 1]
