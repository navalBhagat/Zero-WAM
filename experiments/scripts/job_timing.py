"""Write timing for completed AND interrupted jobs; seconds exclude queue wait."""
import json
from pathlib import Path
import sys


def record(values):
    status, start, server_start, ready, client_start, client_end, end = map(int, values)
    def delta(a, b):
        return b-a if a > 0 and b >= a else None
    return dict(exit_code=status, allocated_seconds=delta(start, end),
                pre_server_seconds=delta(start, server_start),
                server_startup_seconds=delta(server_start, ready),
                client_seconds=delta(client_start, client_end),
                client_started=bool(client_start), client_completed=bool(client_end),
                start_epoch=start, end_epoch=end,
                note='Approximate job-script wall time; sacct ElapsedRaw is authoritative. '
                     'Client includes expert checks, resets, rollout, and video export.')


if __name__ == '__main__':
    path = Path(sys.argv[1])
    path.write_text(json.dumps(record(sys.argv[2:]), indent=2)+'\n')
