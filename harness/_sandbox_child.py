"""Child entrypoint: runs ONE tool call under resource limits, emits JSON.

Reads {"name", "arguments", "cpu_seconds", "mem_mb"} on stdin, applies RLIMITs,
dispatches the tool, and writes the result dict to stdout as JSON. Invoked by
harness.sandbox inside an `unshare -rn` namespace; keep this file dependency-free
beyond the standard library and the local tools package.
"""
from __future__ import annotations

import json
import resource
import sys
from pathlib import Path

# Make the project importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.tools import dispatch  # noqa: E402


def _apply_limits(cpu_seconds: int, mem_mb: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    nbytes = mem_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
    except (ValueError, OSError):
        pass  # some platforms disallow AS limits; CPU limit still applies
    # No new files on disk beyond what's open.
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    except (ValueError, OSError):
        pass


def main() -> None:
    req = json.loads(sys.stdin.read())
    _apply_limits(int(req.get("cpu_seconds", 5)), int(req.get("mem_mb", 256)))
    out = dispatch(req["name"], req.get("arguments", {}))
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
