"""Run a single tool call inside an egress-fenced, resource-limited subprocess.

This mirrors a "sacrificial box" execution model: tool code runs in a child
process placed in its own user+network namespace (rootless `unshare -rn`), so
it has no network reach, under CPU/memory/wall-clock limits. The parent only
passes in a tool name + JSON arguments and reads back a JSON result.

Design rule (honesty discipline): the sandbox never *claims* an isolation it
did not actually achieve. Each run reports `net_isolation_enforced` as a
verified fact — if the namespace could not be created, it says so and the
caller decides whether to proceed, rather than silently running unfenced.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CHILD = Path(__file__).with_name("_sandbox_child.py")


@dataclass
class SandboxResult:
    ok: bool
    result: dict[str, Any] | None
    net_isolation_enforced: bool      # verified, not assumed
    sandboxed: bool                   # did we run in a child namespace at all?
    timed_out: bool
    returncode: int | None
    stderr: str
    notes: list[str] = field(default_factory=list)


def _unshare_available() -> bool:
    if not shutil.which("unshare"):
        return False
    try:
        r = subprocess.run(
            ["unshare", "-rn", "true"],
            capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def run_tool_sandboxed(
    name: str,
    arguments: dict[str, Any],
    *,
    network: bool = False,
    cpu_seconds: int = 5,
    mem_mb: int = 256,
    wall_timeout: int = 10,
    require_isolation: bool = True,
) -> SandboxResult:
    """Execute one tool call in a fenced child process.

    network=False (default) places the child in a fresh network namespace with
    no interfaces up except loopback -> no outbound reach. If isolation cannot
    be established and require_isolation is True, the call fails closed rather
    than running the tool unfenced.
    """
    notes: list[str] = []
    payload = json.dumps({
        "name": name, "arguments": arguments,
        "cpu_seconds": cpu_seconds, "mem_mb": mem_mb,
    })

    use_ns = (not network) and _unshare_available()
    if not network and not use_ns:
        notes.append("network namespace unavailable; isolation NOT enforced")
        if require_isolation:
            return SandboxResult(
                ok=False, result=None, net_isolation_enforced=False,
                sandboxed=False, timed_out=False, returncode=None,
                stderr="refused: network isolation required but unavailable",
                notes=notes,
            )

    if use_ns:
        # -r: map current user to root inside the namespace (rootless)
        # -n: new network namespace (no external interfaces)
        cmd = ["unshare", "-rn", sys.executable, "-I", str(_CHILD)]
    else:
        cmd = [sys.executable, "-I", str(_CHILD)]

    try:
        proc = subprocess.run(
            cmd, input=payload, capture_output=True, text=True,
            timeout=wall_timeout,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            ok=False, result=None, net_isolation_enforced=use_ns,
            sandboxed=use_ns or not network, timed_out=True, returncode=None,
            stderr=f"wall-clock timeout after {wall_timeout}s", notes=notes,
        )

    if proc.returncode != 0:
        return SandboxResult(
            ok=False, result=None, net_isolation_enforced=use_ns,
            sandboxed=True, timed_out=False, returncode=proc.returncode,
            stderr=proc.stderr.strip()[:500], notes=notes,
        )

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return SandboxResult(
            ok=False, result=None, net_isolation_enforced=use_ns,
            sandboxed=True, timed_out=False, returncode=0,
            stderr="child produced non-JSON output", notes=notes,
        )

    return SandboxResult(
        ok=True, result=result, net_isolation_enforced=use_ns,
        sandboxed=use_ns or not network, timed_out=False, returncode=0,
        stderr="", notes=notes,
    )


def verify_fence() -> dict[str, Any]:
    """Self-check: prove the fence blocks outbound network, and report honestly.

    Runs a tool that *attempts* a network connection inside the sandbox and
    confirms it fails. Returns a dict suitable for logging alongside results.
    """
    res = run_tool_sandboxed(
        "attempt_network", {"host": "1.1.1.1", "port": 53},
        network=False, require_isolation=False,
    )
    blocked = bool(res.result and res.result.get("blocked") is True)
    return {
        "net_isolation_enforced": res.net_isolation_enforced,
        "outbound_blocked_verified": blocked,
        "sandboxed": res.sandboxed,
        "notes": res.notes,
    }


if __name__ == "__main__":
    print(json.dumps(verify_fence(), indent=2))
