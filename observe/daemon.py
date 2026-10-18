"""The out-of-process controller daemon.

The daemon attaches to one serving process, or to every worker process of a tensor-parallel engine
(one --pid for each worker). It reads the request context and the attention-share statistic of the
compiler pass (through CUDA IPC), runs the decide stage (``observe.controller``), and commits complete
policy generations to the shared-memory mailbox of each process (``observe.ranks``). A hook in the
serving process applies them between steps. The daemon never writes serving memory. A stopped,
killed, or wedged daemon needs no cleanup: its last generation carries a lease, and when the lease
passes every request returns to full reads.

With no --pid the daemon attaches to the newest live serving process that has a control mailbox;
--follow reattaches after an engine restart instead of exiting. --health-file writes a one-line
JSON heartbeat each second. --duration 0 runs until killed. Deployment: deploy/ekv-daemon.service,
docs/OPERATIONS.md. Attaching to another process's GPU buffers needs the same user.

Usage:  CUDA_VISIBLE_DEVICES=<gpu> python -m observe.daemon \
            [--pid <engine pid> [<worker pid> ...]] [--policy sum|maxmin|uniform|recency] [--budget 0.5]
            [--interval 0.02] [--duration 30] [--lease-seconds 0.5] [--follow]
            [--target-step-ms 60 [--reload-aware DIR]] [--health-file /run/ekv/health.json] [--verbose]

--reload-aware attaches the eBPF copy collector (``observe.ebpf_attach.EbpfSession``, needs sudo) to an engine
with vLLM's native CPU KV offload, and the step-time target then leaves out every step during which a KV
reload from host memory was in flight (``observe.reloads``).
"""

from __future__ import annotations

import argparse
import faulthandler
import glob
import json
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from observe.controller import Controller, Settings, TileShareSource
from observe.cuda import Cuda
from observe.ebpf_attach import EbpfSession
from observe.publication import MailboxWriter, mailbox_path, request_key
from observe.ranks import RankedShareSource, RankedWriter
from observe.reloads import ReloadTracker, read_kv_cache
from observe.tile_shares import Aggregate, share_descriptor

OBJECTIVES = {"sum": "sum", "maxmin": "maxmin", "uniform": "uniform", "recency": "uniform"}


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists and is not ours


def discover_pid() -> int | None:
    """Return the live serving process whose control mailbox is newest."""
    best, best_mtime = None, -1.0
    for path in glob.glob(f"{os.environ.get('EKV_CONTROL_DIR', '/dev/shm')}/ekv_control_*.shm"):
        try:
            pid = int(path.rsplit("_", 1)[1].split(".")[0])
        except ValueError:
            continue
        if alive(pid) and os.path.getmtime(path) > best_mtime:
            best, best_mtime = pid, os.path.getmtime(path)
    return best


def share_source(pids: list[int], aggregate: Aggregate) -> TileShareSource | RankedShareSource:
    """The attention-share reader for ``pids``, from the descriptor that each process's compiler pass published."""
    sources = [TileShareSource(Cuda(), share_descriptor(pid), pid, aggregate) for pid in pids]
    return sources[0] if len(sources) == 1 else RankedShareSource(sources)


def make_controller(
    args: argparse.Namespace,
    writer: MailboxWriter | RankedWriter,
    pids: list[int],
    reloads: ReloadTracker | None,
) -> Controller:
    settings = Settings(
        budget_fraction=args.budget,
        objective=OBJECTIVES[args.policy],  # ty: ignore[invalid-argument-type]  # a key of OBJECTIVES
        recent_blocks=args.recent_blocks,
        lease_seconds=args.lease_seconds,
        refresh_blocks=args.refresh_blocks,
        share_decay=args.share_decay,
        target_step_ms=args.target_step_ms,
        min_budget_fraction=args.min_budget,
        release_after=args.release_after,
        reload_aware=bool(args.reload_aware),
        cap_above_blocks=args.cap_above_blocks,
    )
    weight = None
    if args.weights:
        # The front end knows its own request ids. A request that is not listed has weight 1.
        with open(args.weights) as f:
            table = {request_key(request_id): float(w) for request_id, w in json.load(f).items()}
        weight = lambda key: table.get(key, 1.0)
    if args.policy == "recency":
        return Controller(
            writer,
            None,
            settings,
            weight=weight,
            scores=lambda n: np.arange(n, dtype=float),
            reloads=reloads,
        )
    return Controller(writer, share_source(pids, args.aggregate), settings, weight=weight, reloads=reloads)


def attach_reloads(pids: list[int], capture_dir: Path) -> tuple[EbpfSession, ReloadTracker]:
    """Attach the eBPF copy collector to the engine, and read its reloads from host memory.

    Raises if the engine has not published its GPU KV cache, or has more than one process: a tensor-parallel
    engine moves KV in each worker, and this reads one process."""
    if len(pids) != 1:
        raise ValueError(f"the reload-aware target reads one engine process, not {len(pids)}")
    kv_cache = read_kv_cache(pids[0])
    capture_dir.mkdir(parents=True, exist_ok=True)
    session = EbpfSession(pids[0], capture_dir, alloc=False, copy=True)
    session.__enter__()
    return session, ReloadTracker(session.kv_copy, kv_cache)


def serve(args: argparse.Namespace) -> int:
    end = None if args.duration <= 0 else time.time() + args.duration
    commits = 0
    while end is None or time.time() < end:
        found = discover_pid()
        pids = args.pid or ([found] if found else [])
        missing = [pid for pid in pids if not mailbox_path(pid).exists()]
        if not pids or missing:
            if not args.follow:
                print(f"[daemon] no control mailbox for pid {missing or '(none found)'}", file=sys.stderr)
                return 1
            time.sleep(0.5)
            continue
        session, reloads = (
            attach_reloads(pids, Path(args.reload_aware)) if args.reload_aware else (None, None)
        )
        writers = [MailboxWriter(mailbox_path(pid)) for pid in pids]
        writer = writers[0] if len(writers) == 1 else RankedWriter(writers)
        controller = make_controller(args, writer, pids, reloads)
        print(
            f"[daemon] attached to pid={pids} policy={args.policy} budget={args.budget} "
            f"reloads={'eBPF ' + str(session.kv_copy) if session else 'not read'}",
            flush=True,
        )
        last_health = 0.0
        trace = open(args.trace, "w") if args.trace else None  # noqa: SIM115  # closed in the finally clause
        tick_us: deque[float] = deque(maxlen=100_000)  # time of each tick, not counting the sleep
        wall_start, cpu_start = time.monotonic(), time.process_time()
        next_due = wall_start
        try:
            while (end is None or time.time() < end) and all(alive(pid) for pid in pids):
                tick_start = time.monotonic()
                controller.tick()
                tick_us.append((time.monotonic() - tick_start) * 1e6)
                commits += 1
                if args.health_file and time.time() - last_health >= 1.0:
                    last_health = time.time()
                    usage = measure_usage(controller, tick_us, wall_start, cpu_start)
                    write_health(args.health_file, pids, args.policy, controller, writer, usage)
                if args.verbose and controller.ticks % 50 == 0:
                    print(f"[daemon] tick={controller.ticks} measured={len(controller.shares)}", flush=True)
                if trace is not None and controller.allocation and controller.ticks % args.trace_every == 0:
                    trace.write(
                        json.dumps(
                            {"tick": controller.ticks, "time": time.time(), "rows": controller.allocation}
                        )
                        + "\n"
                    )
                # A fixed period: the sleep is what is left of the interval after the tick. A tick that takes
                # longer than the interval is followed by the next one at once, and does not make up the lost time.
                next_due += args.interval
                delay = next_due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_due = time.monotonic()
        except KeyboardInterrupt:
            pass
        finally:
            writer.close()
            if trace is not None:
                trace.close()
            if session is not None:
                session.detach()
        final = measure_usage(controller, tick_us, wall_start, cpu_start)
        if reloads is not None:
            reloads.poll()  # the collector wrote its last records when it stopped
            final["reloads_read"] = reloads.read_count
            final["reload_bytes_read"] = reloads.read_bytes
        print(
            f"EKV_DAEMON pid={pids} policy={args.policy} ticks={controller.ticks} {json.dumps(final)}",
            flush=True,
        )
        if not args.follow:
            break
        args.pid = []  # rediscover on the next attach
    return 0 if commits else 1


def measure_usage(
    controller: Controller, tick_us: deque[float], wall_start: float, cpu_start: float
) -> dict[str, float]:
    """What this controller process has used since it attached: measured CPU time, the real period, the time
    of a tick without its sleep, and the controller's own latency summary."""
    wall, cpu = time.monotonic() - wall_start, time.process_time() - cpu_start
    ticks = np.asarray(tick_us) if tick_us else np.zeros(1)
    return {
        "wall_s": round(wall, 3),
        "cpu_s": round(cpu, 3),
        "cpu_percent": round(100 * cpu / wall, 2) if wall > 0 else 0.0,
        "period_ms_mean": round(1e3 * wall / max(controller.ticks, 1), 3),
        "tick_us_p50": round(float(np.percentile(ticks, 50)), 1),
        "tick_us_p99": round(float(np.percentile(ticks, 99)), 1),
        **{k: round(v, 3) for k, v in controller.summary().items()},
    }


def write_health(
    path: str,
    pids: list[int],
    policy: str,
    controller: Controller,
    writer: MailboxWriter | RankedWriter,
    usage: dict[str, float],
) -> None:
    record = {
        "ts": time.time(),
        "daemon_pid": os.getpid(),
        "engine_pids": pids,
        "policy": policy,
        "ticks": controller.ticks,
        "measured_requests": len(controller.shares),
        **writer.box.status(),
        **usage,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f)
    os.replace(tmp, path)  # atomic: a reader never sees a torn file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pid",
        type=int,
        nargs="+",
        default=[],
        help="serving process, or each worker of a tensor-parallel engine (omit to discover the newest)",
    )
    parser.add_argument("--policy", default="sum", choices=sorted(OBJECTIVES))
    parser.add_argument("--budget", type=float, default=0.5, help="fraction of measured blocks to keep")
    parser.add_argument("--recent-blocks", type=int, default=4, help="newest blocks always kept")
    parser.add_argument(
        "--aggregate",
        choices=["mean", "max"],
        default="mean",
        help="score a block by its average share over heads, or by the strongest head that attends it",
    )
    parser.add_argument("--interval", type=float, default=0.02)
    parser.add_argument("--duration", type=float, default=30.0, help="seconds to run; 0 = until killed")
    parser.add_argument(
        "--lease-seconds",
        type=float,
        default=0.5,
        help="a generation not renewed within this time lapses to full reads",
    )
    parser.add_argument(
        "--refresh-blocks",
        type=int,
        default=0,
        help="audit a request again when it has this many unmeasured blocks; 0 audits it once",
    )
    parser.add_argument(
        "--target-step-ms",
        type=float,
        default=None,
        help="hold the median engine step time near this value by moving the budget (starting from "
        "--budget, between --min-budget and 1); omit to keep --budget fixed",
    )
    parser.add_argument(
        "--min-budget",
        type=float,
        default=Settings.min_budget_fraction,
        help="the smallest budget fraction the step-time target may reach",
    )
    parser.add_argument(
        "--cap-above-blocks",
        type=int,
        default=0,
        help="apply the budget only to requests longer than this many blocks; shorter requests read in full. "
        "0 applies it to every request",
    )
    parser.add_argument(
        "--share-decay",
        type=float,
        default=Settings.share_decay,
        help="weight of the older shares when a new audit of a request arrives, in [0, 1)",
    )
    parser.add_argument(
        "--release-after",
        type=int,
        default=0,
        help="ask the engine to free a block left out of this many audits in a row (at least 2); "
        "0 never frees. The engine must run with EKV_RELEASE=1.",
    )
    parser.add_argument(
        "--reload-aware",
        default="",
        metavar="DIR",
        help="attach the eBPF copy collector (sudo) to the engine, write its capture in DIR, and leave out of "
        "--target-step-ms every step during which a KV reload from host memory was in flight (vLLM's native "
        "CPU KV offload, kv_offloading_size)",
    )
    parser.add_argument("--follow", action="store_true", help="reattach after an engine restart")
    parser.add_argument("--health-file", default="", help="write a 1 Hz JSON heartbeat here")
    parser.add_argument(
        "--weights",
        default="",
        help="JSON file {client request id: weight}; a heavier request keeps more blocks under 'sum'",
    )
    parser.add_argument(
        "--trace",
        default="",
        help="write the allocation (request key, blocks, blocks kept) as JSON lines here",
    )
    parser.add_argument("--trace-every", type=int, default=50, help="ticks between trace records")
    parser.add_argument("--verbose", action="store_true")
    # SIGUSR1 writes the stack of every thread to stderr, also while a thread waits in native code, where Python
    # cannot run a signal handler. An operator finds where a daemon is stuck with `kill -USR1 <pid>`.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    sys.exit(serve(parser.parse_args()))


if __name__ == "__main__":
    main()
