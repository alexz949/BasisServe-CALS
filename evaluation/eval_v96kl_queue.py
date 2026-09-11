"""Distribute unchanged V96-KL evaluation samples with per-sample OS locks."""
import ctypes
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import eval_v96kl_longbench as evaluation

libc = ctypes.CDLL(None, use_errno=True)
original_inputs = evaluation.inputs


class QueueRows(list):
    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.claims()
        return super().__getitem__(key)

    def claims(self):
        for row in self:
            result = self.output / self.arm / 'evaluate' / f"sample_{row['index']:05d}.json"
            if result.exists():
                continue
            with (self.locks / f"{row['index']:05d}.lock").open('a') as lock:
                if libc.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB) != 0:
                    continue
                if not result.exists():
                    print('CLAIM', row['index'], row['task'], flush=True)
                    yield row
                libc.flock(lock.fileno(), fcntl.LOCK_UN)


def queue_inputs(args):
    manifest, rows, tokens, protocol = original_inputs(args)
    assert args.stage == 'evaluate' and args.arm == 'lrqk'
    queued = QueueRows(rows)
    queued.output = args.output
    queued.arm = args.arm
    queued.locks = args.output / 'lrqk' / 'queue_locks'
    queued.locks.mkdir(parents=True, exist_ok=True)
    return manifest, queued, tokens, protocol


if __name__ == '__main__':
    evaluation.inputs = queue_inputs
    evaluation.main()
