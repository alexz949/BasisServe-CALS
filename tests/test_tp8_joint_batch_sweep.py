import json

from benchmarks.system.run_tp8_joint_batch_sweep import (
    ARMS, BATCHES, MODELS, command_for, failure_status,
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
