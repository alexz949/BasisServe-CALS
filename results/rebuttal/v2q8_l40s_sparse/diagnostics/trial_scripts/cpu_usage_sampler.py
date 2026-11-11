"""Every 5 s: how many logical CPUs the whole host keeps busy, how many of that are our benchmark processes
(bench_v2q8_tp.py), and the same for the 24 CPUs a GPU-0 job is pinned to. JSON lines to the file given as argv[1];
stops when the tmux session given as argv[2] is gone."""
import json, os, subprocess, sys, time

OUT, SESSION = sys.argv[1], sys.argv[2]
TICK = os.sysconf("SC_CLK_TCK")
PINNED = list(range(0, 24, 2)) + list(range(96, 120, 2))


def host():
    rows = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            parts = line.split()
            values = list(map(int, parts[1:9]))
            rows[int(parts[0][3:])] = (sum(values), values[3] + values[4])
    return rows


def ours():
    total, names = 0.0, []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            command = open(f"/proc/{pid}/cmdline").read().replace("\x00", " ")
            if "bench_v2q8_tp.py" not in command or "torch.distributed.run" in command:
                continue
            fields = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            total += (int(fields[11]) + int(fields[12])) / TICK
            names.append(command.split("--arm ")[1].split()[0] + ":" + command.split("--mode ")[1].split()[0])
        except (OSError, IndexError):
            continue
    return total, names


before, mine, clock = host(), ours()[0], time.time()
with open(OUT, "a") as handle:
    while subprocess.run(["tmux", "has-session", "-t", SESSION], capture_output=True).returncode == 0:
        time.sleep(5)
        after, (now_mine, names), now = host(), *[ours()], time.time()
        busy = {cpu: 1 - (after[cpu][1] - before[cpu][1]) / max(after[cpu][0] - before[cpu][0], 1) for cpu in after}
        handle.write(json.dumps(dict(
            time_utc=time.strftime("%H:%M:%S", time.gmtime(now)), jobs=names,
            host_busy_cpus=round(sum(busy.values()), 1), our_busy_cpus=round(max(now_mine - mine, 0) / (now - clock), 2),
            pinned_set_busy_cpus=round(sum(busy[cpu] for cpu in PINNED), 1))) + "\n")
        handle.flush()
        before, mine, clock = after, now_mine, now
