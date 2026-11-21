#!/usr/bin/env python3
"""Queue four split speed/count gate runs behind the live baseline scheduler.

One frequency occupies one physical GPU. A free card is taken only when every
live baseline scheduler will not launch: it still has CUDA work, but the number
of alive CUDA tasks is already at ``max_gpus``. A card freed by an exit in that
state is not refilled. If a baseline queue cannot be read, this scheduler waits.

State and logs:
    <training_results_3s>/operations/mstt_state_aux_gate_split_mlp_p5_p15/
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_root import new_run_id, xuancheng26_training_3s_root  # noqa: E402
from MSTT.config import load_config  # noqa: E402
from MSTT.run_mstt_state_aux_gate_temporal_aux_p5_p15 import (  # noqa: E402
    GPU_LOCK_DIR,
    IDLE_MEMORY_MIB,
    PYTHON,
    RESULTS,
    UTIL_LIMIT,
    FileLock,
    GpuClaim,
    atomic_json,
    child_env,
    pid_alive,
    process_command,
    query_gpus,
    read_json,
    stamp,
)
from MSTT.train import resolve_output_root  # noqa: E402
from MSTT.trainer import run_directory  # noqa: E402
from MSTT.utils.threads import MAX_THREADS, THREAD_ENV_KEYS  # noqa: E402
from reimplementation.common.data.sampling_seeds import configure_sampling  # noqa: E402

OPS = RESULTS / "operations" / "mstt_state_aux_gate_split_mlp_p5_p15"
LOCK_PATH = OPS / "scheduler.lock"
STATE_PATH = OPS / "state.json"
LOG_PATH = OPS / "scheduler.log"
PID_PATH = OPS / "scheduler.pid"
BASELINE_OPS = RESULTS / "operations"
CONFIG_REL = "MSTT/configs/all_sp_c_state_aux_gate_split_mlp_d128_projected_all.json"
EXPERIMENT_ID = "state_aux_gate_split_mlp_d128_projected_all"
FREQUENCIES = (
    ("30min", "variable_30min_p5_p15"),
    ("1h", "variable_1h_p5_p15"),
    ("2h", "variable_2h_p5_p15"),
    ("day", "variable_day_p5_p15"),
)
POLL_SECONDS = 30
BOUND = frozenset({"starting", "running"})


def baseline_blocks_claim(alive_cuda: int, max_gpus: int, pending_cuda: int) -> bool:
    """True when a live baseline scheduler still wants a GPU on its next poll.

    ``max_gpus`` counts alive CUDA tasks. At or above that cap a pending CUDA
    task stays queued, so a card released by an exit can be taken. Below the
    cap, any pending CUDA task means the baseline scheduler may launch.
    """
    if min(alive_cuda, max_gpus, pending_cuda) < 0:
        raise ValueError(f"negative baseline counts alive={alive_cuda} max={max_gpus} pending={pending_cuda}")
    return pending_cuda > 0 and alive_cuda < max_gpus


def log(message: str) -> None:
    OPS.mkdir(parents=True, exist_ok=True)
    line = f"[{stamp()}] {message}\n"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def planned_output(dataset_id: str, run_id: str) -> str:
    config = load_config(ROOT / CONFIG_REL)
    config["dataset_id"] = dataset_id
    config["run_id"] = run_id
    config["explicit_path_keys"] = ["dataset_id"]
    config = configure_sampling(config, model_name="MSTT")
    root = resolve_output_root(config)
    config["output_root"] = root.as_posix()
    return run_directory(root, "p5_p15", "AllSpC2All", 42, config).as_posix()


def dataset_ready(dataset_id: str) -> str | None:
    root = xuancheng26_training_3s_root() / "datasets" / dataset_id
    manifest_path = root / "dataset_manifest.json"
    payload, error = read_json(manifest_path)
    if error or not isinstance(payload, dict):
        return f"unreadable dataset manifest {manifest_path}: {error}"
    keys = [str(item) for item in payload.get("observation_keys") or []]
    if keys != ["p5_p15"]:
        return f"{dataset_id} observation_keys={keys!r}"
    windows = root / "sampling_seeds" / "seed_42" / "flow" / "window_data" / "p5_p15"
    if not windows.is_dir() or next(windows.glob("*/*/all_sp_c.npz"), None) is None:
        return f"{dataset_id} has no seed-42 all_sp_c.npz under {windows}"
    return None


def fresh_state() -> dict:
    tasks = []
    for frequency, dataset_id in FREQUENCIES:
        run_id = new_run_id("MSTT", dataset_id)
        problem = dataset_ready(dataset_id)
        tasks.append({
            "frequency": frequency,
            "dataset_id": dataset_id,
            "experiment_id": EXPERIMENT_ID,
            "run_id": run_id,
            "status": "blocked" if problem else "pending",
            "gpu_index": None,
            "gpu_uuid": None,
            "pid": None,
            "supervisor_pid": None,
            "log": str(OPS / "logs" / f"{run_id}.log"),
            "output_dir": planned_output(dataset_id, run_id),
            "command": None,
            "error": problem,
            "started_at": None,
            "finished_at": None,
        })
    return {
        "campaign": "state_aux_gate_split_mlp_p5_p15",
        "code_root": str(ROOT),
        "config": CONFIG_REL,
        "experiment_id": EXPERIMENT_ID,
        "max_threads": MAX_THREADS,
        "priority": "baseline CUDA tasks below max_gpus are launched first",
        "tasks": tasks,
        "created_at": stamp(),
        "updated_at": stamp(),
    }


def load_state() -> dict:
    if STATE_PATH.is_file():
        payload, error = read_json(STATE_PATH)
        if error or not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
            raise SystemExit(f"unreadable scheduler state: {error or payload!r}")
        return payload
    state = fresh_state()
    save_state(state)
    return state


def save_state(state: dict) -> None:
    state["updated_at"] = stamp()
    state["max_threads"] = MAX_THREADS
    atomic_json(STATE_PATH, state)


def run_completed(path: str | None) -> bool:
    if not path:
        return False
    directory = Path(path)
    meta, error = read_json(directory / "run_metadata.json")
    return (
        not error
        and isinstance(meta, dict)
        and meta.get("status") == "completed"
        and (directory / "test_sampling_metrics.json").is_file()
    )


def train_command(task: dict) -> list[str]:
    return [
        str(PYTHON), "-u", "MSTT/train.py",
        "--config", CONFIG_REL,
        "--task", "AllSpC2All",
        "--dataset-id", task["dataset_id"],
        "--penetration", "p5_p15",
        "--gpu", "0",
        "--run-id", task["run_id"],
    ]


def adopt(state: dict) -> None:
    for task in state["tasks"]:
        if run_completed(task.get("output_dir")):
            task["status"] = "completed"
            task["pid"] = None
            task["supervisor_pid"] = None
            task["error"] = None
            continue
        if task["status"] == "failed":
            continue
        pid = task.get("pid")
        if task["status"] in BOUND and pid_alive(pid) and task["run_id"] in process_command(int(pid)):
            task["status"] = "running"
            continue
        supervisor = task.get("supervisor_pid")
        if task["status"] == "starting" and pid_alive(supervisor):
            continue
        if task["status"] == "running":
            task["status"] = "failed"
            task["finished_at"] = stamp()
            task["error"] = "training process exited before test metrics were written"
            continue
        if task["status"] == "starting":
            task["status"] = "pending"
            task["pid"] = None
            task["supervisor_pid"] = None
            task["gpu_uuid"] = None
            task["gpu_index"] = None
            task["error"] = "supervisor exited before training started"


def next_pending(state: dict):
    for task in state["tasks"]:
        if task["status"] == "pending":
            return task
    return None


def _cuda_counts(tasks: list) -> tuple[int, int, set[str]]:
    alive = 0
    pending = 0
    reserved: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict) or task.get("cpu"):
            continue
        status = task.get("status")
        if status == "pending":
            pending += 1
            continue
        if status != "running":
            continue
        pid = int(task.get("pid") or 0)
        if not pid_alive(pid):
            continue
        alive += 1
        uuid = str(task.get("gpu_uuid") or "")
        if uuid:
            reserved.add(uuid)
    return alive, pending, reserved


def live_baseline_snapshot(ops_root: Path | None = None) -> tuple[bool, str, set[str]]:
    """Return whether to wait, why, and UUIDs already held by live baseline tasks."""
    root = BASELINE_OPS if ops_root is None else ops_root
    reserved: set[str] = set()
    notes: list[str] = []
    blocked = False
    saw_live = False
    if not root.is_dir():
        return True, f"missing baseline operations directory {root}", reserved
    for campaign in sorted(path for path in root.iterdir() if path.is_dir()):
        lock = campaign / "scheduler.lock"
        if not lock.is_file():
            continue
        owner = lock.read_text(encoding="utf-8", errors="replace").strip()
        if not owner.isdigit() or not pid_alive(int(owner)):
            continue
        if "run_variable_baselines.py" not in process_command(int(owner)):
            continue
        saw_live = True
        payload, error = read_json(campaign / "tasks.json")
        control, control_error = read_json(campaign / "control.json")
        if error or not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
            return True, f"{campaign.name} tasks unreadable: {error or payload!r}", reserved
        max_gpus = None if not isinstance(control, dict) else control.get("max_gpus")
        if control_error or isinstance(max_gpus, bool) or not isinstance(max_gpus, int):
            return True, f"{campaign.name} control unreadable: {control_error or control!r}", reserved
        alive, pending, held = _cuda_counts(payload["tasks"])
        reserved.update(held)
        if baseline_blocks_claim(alive, int(max_gpus), pending):
            blocked = True
            notes.append(
                f"{campaign.name} alive_cuda={alive} < max_gpus={max_gpus} pending_cuda={pending}"
            )
        else:
            notes.append(
                f"{campaign.name} alive_cuda={alive} max_gpus={max_gpus} pending_cuda={pending} "
                "baseline_will_not_launch"
            )
    if not saw_live:
        return False, "no live baseline scheduler", reserved
    return blocked, "; ".join(notes), reserved


def external_reserved_uuids() -> tuple[set[str], str | None]:
    """UUIDs held by other live experiment schedulers, or a reason to wait."""
    reserved: set[str] = set()
    root = RESULTS / "operations"
    if not root.is_dir():
        return reserved, f"missing {root}"
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if directory == OPS:
            continue
        state_path = directory / "state.json"
        if not state_path.is_file():
            continue
        owner = ""
        lock = directory / "scheduler.lock"
        pid_file = directory / "scheduler.pid"
        if lock.is_file():
            owner = lock.read_text(encoding="utf-8", errors="replace").strip()
        elif pid_file.is_file():
            owner = pid_file.read_text(encoding="utf-8", errors="replace").strip()
        if not owner.isdigit() or not pid_alive(int(owner)):
            continue
        payload, error = read_json(state_path)
        if error or not isinstance(payload, dict):
            return reserved, f"{directory.name} state unreadable: {error}"
        for task in payload.get("tasks") or []:
            if not isinstance(task, dict) or task.get("status") not in BOUND:
                continue
            uuid = str(task.get("gpu_uuid") or "")
            if uuid and (pid_alive(task.get("pid")) or pid_alive(task.get("supervisor_pid"))):
                reserved.add(uuid)
    if GPU_LOCK_DIR.is_dir():
        for path in sorted(GPU_LOCK_DIR.glob("*.lock")):
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                note = json.loads(text)
            except json.JSONDecodeError:
                return reserved, f"unreadable gpu lock {path.name}"
            supervisor_pid = note.get("supervisor_pid") if isinstance(note, dict) else None
            if pid_alive(supervisor_pid) and int(supervisor_pid) != os.getpid():
                reserved.add(path.name[:-5] if path.name.endswith(".lock") else path.name)
    return reserved, None


def claimed_uuids(state: dict) -> set[str]:
    found = set()
    for task in state["tasks"]:
        if task["status"] in BOUND and task.get("gpu_uuid"):
            found.add(str(task["gpu_uuid"]))
    return found


def select_free(rows: list[dict], apps: dict, reserved: set[str]) -> list[dict]:
    free = []
    for row in rows:
        uuid = row["uuid"]
        if row.get("indeterminate") or apps.get(uuid) or uuid in reserved:
            continue
        if float(row.get("memory_used", 1e9)) > IDLE_MEMORY_MIB:
            continue
        if float(row.get("utilization", 100)) > UTIL_LIMIT:
            continue
        free.append(row)
    free.sort(key=lambda row: int(row["index"]))
    return free


def run_supervisor(payload_path: Path) -> int:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    claim = GpuClaim(GPU_LOCK_DIR, payload["uuid"])
    if not claim.try_acquire({"supervisor_pid": os.getpid(), "run_id": payload["run_id"], "started_at": stamp()}):
        return 2
    try:
        rows, apps = query_gpus()
        blocked, reason, baseline_reserved = live_baseline_snapshot()
        others, other_error = external_reserved_uuids()
    except (OSError, subprocess.SubprocessError, RuntimeError, json.JSONDecodeError, ValueError) as exc:
        claim.release()
        Path(payload["log"]).parent.mkdir(parents=True, exist_ok=True)
        Path(payload["log"]).write_text(f"supervisor recheck failed: {exc}\n", encoding="utf-8")
        return 5
    if blocked or other_error:
        claim.release()
        Path(payload["log"]).parent.mkdir(parents=True, exist_ok=True)
        Path(payload["log"]).write_text(
            f"supervisor yielded: {other_error or reason}\n",
            encoding="utf-8",
        )
        return 4
    reserved = baseline_reserved | others
    row = next((item for item in rows if item["uuid"] == payload["uuid"]), None)
    busy = (
        row is None
        or row.get("indeterminate")
        or apps.get(payload["uuid"])
        or payload["uuid"] in reserved
        or float(row.get("memory_used", 1e9)) > IDLE_MEMORY_MIB
    )
    if busy:
        claim.release()
        return 4
    log_handle = Path(payload["log"]).open("w", encoding="utf-8")
    env = child_env(payload["uuid"])
    if any(env[key] != "15" for key in THREAD_ENV_KEYS) or env["MSTT_NUM_THREADS"] != "15":
        log_handle.write("refusing to start without a 15-thread cap\n")
        log_handle.close()
        claim.release()
        return 6
    process = subprocess.Popen(
        payload["command"],
        cwd=payload["cwd"],
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    atomic_json(Path(payload["pid_file"]), {
        "pid": process.pid,
        "supervisor_pid": os.getpid(),
        "gpu_uuid": payload["uuid"],
        "gpu_index": payload["gpu_index"],
        "max_threads": MAX_THREADS,
    })
    code = process.wait()
    log_handle.close()
    claim.release()
    return int(code)


def start_task(state: dict, task: dict, gpu: dict) -> bool:
    log_path = Path(task["log"])
    pid_path = OPS / "logs" / f"{task['run_id']}.pid.json"
    payload_path = OPS / "payloads" / f"{task['run_id']}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = train_command(task)
    payload = {
        "uuid": gpu["uuid"],
        "gpu_index": gpu["index"],
        "run_id": task["run_id"],
        "command": command,
        "cwd": str(ROOT),
        "log": str(log_path),
        "pid_file": str(pid_path),
    }
    atomic_json(payload_path, payload)
    supervisor = subprocess.Popen(
        [str(PYTHON), "-u", str(Path(__file__).resolve()), "supervise", str(payload_path)],
        cwd=str(ROOT),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    task.update({
        "status": "starting",
        "gpu_uuid": gpu["uuid"],
        "gpu_index": gpu["index"],
        "supervisor_pid": supervisor.pid,
        "pid": None,
        "command": command,
        "started_at": stamp(),
        "error": None,
    })
    save_state(state)
    deadline = time.time() + 20
    while time.time() < deadline:
        if pid_path.is_file():
            info, error = read_json(pid_path)
            if error or not isinstance(info, dict) or not isinstance(info.get("pid"), int):
                break
            task.update({
                "status": "running",
                "pid": int(info["pid"]),
                "supervisor_pid": info.get("supervisor_pid", supervisor.pid),
            })
            save_state(state)
            log(
                f"launch {task['frequency']} dataset={task['dataset_id']} pid={task['pid']} "
                f"gpu={gpu['index']}:{gpu['uuid']} threads={MAX_THREADS} run_id={task['run_id']} log={log_path}"
            )
            return True
        if supervisor.poll() is not None:
            task.update({
                "status": "pending",
                "pid": None,
                "supervisor_pid": None,
                "gpu_uuid": None,
                "gpu_index": None,
                "error": f"supervisor exited {supervisor.returncode} before training started",
            })
            save_state(state)
            log(f"rollback {task['frequency']} supervisor_exit={supervisor.returncode}")
            return False
        time.sleep(0.05)
    log(f"supervisor still starting {task['run_id']} pid={supervisor.pid}")
    return False


def reap(state: dict) -> None:
    for task in state["tasks"]:
        if task["status"] not in BOUND:
            continue
        if task["status"] == "starting" and not task.get("pid"):
            info, error = read_json(OPS / "logs" / f"{task['run_id']}.pid.json")
            if not error and isinstance(info, dict) and isinstance(info.get("pid"), int):
                task["pid"] = int(info["pid"])
                task["supervisor_pid"] = info.get("supervisor_pid", task.get("supervisor_pid"))
                task["status"] = "running"
        if run_completed(task.get("output_dir")):
            task["status"] = "completed"
            task["pid"] = None
            task["supervisor_pid"] = None
            task["finished_at"] = stamp()
            log(f"completed {task['frequency']} run_id={task['run_id']}")
            continue
        pid = task.get("pid")
        supervisor = task.get("supervisor_pid")
        if pid_alive(pid):
            continue
        if task["status"] == "starting" and pid_alive(supervisor):
            continue
        log_path = Path(task["log"]) if task.get("log") else None
        tail = ""
        if log_path and log_path.is_file():
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        task["status"] = "failed"
        task["pid"] = None
        task["supervisor_pid"] = None
        task["finished_at"] = stamp()
        task["error"] = tail[-800:] or "training process exited before test metrics were written"
        log(f"failed {task['frequency']} run_id={task['run_id']}")


def print_status() -> int:
    if not STATE_PATH.is_file():
        print(f"no queue at {STATE_PATH}")
        return 1
    state, error = read_json(STATE_PATH)
    if error or not isinstance(state, dict):
        print(f"unreadable state: {error}")
        return 1
    print(f"campaign={state.get('campaign')} updated={state.get('updated_at')} pid_file={PID_PATH}")
    print(f"log={LOG_PATH}")
    for task in state.get("tasks") or []:
        print(
            f"{task.get('dataset_id')} status={task.get('status')} run_id={task.get('run_id')} "
            f"gpu={task.get('gpu_index')} pid={task.get('pid')} log={task.get('log')}"
        )
    return 0


def serve() -> int:
    lock = FileLock(LOCK_PATH)
    if not lock.try_acquire(f"{os.getpid()}\n"):
        holder = lock.read_holder()
        print(f"scheduler already running: {holder or LOCK_PATH}")
        return 0
    PID_PATH.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    state = load_state()
    adopt(state)
    save_state(state)
    log(f"scheduler start pid={os.getpid()} threads={MAX_THREADS} experiment={EXPERIMENT_ID}")
    for task in state["tasks"]:
        log(
            f"queued {task['frequency']} status={task['status']} run_id={task['run_id']} "
            f"output={task['output_dir']} error={task.get('error')}"
        )
    while True:
        reap(state)
        save_state(state)
        pending = next_pending(state)
        if pending is None and all(task["status"] in {"completed", "failed", "blocked"} for task in state["tasks"]):
            if any(task["status"] == "blocked" for task in state["tasks"]):
                log("waiting; a task is blocked and no pending task can start")
                time.sleep(POLL_SECONDS)
                continue
            log("scheduler idle; every frequency is completed or failed")
            save_state(state)
            return 0
        if pending is None:
            time.sleep(POLL_SECONDS)
            continue
        try:
            blocked, reason, baseline_reserved = live_baseline_snapshot()
            others, other_error = external_reserved_uuids()
            if blocked or other_error:
                log(f"waiting {pending['frequency']}; {other_error or reason}")
                time.sleep(POLL_SECONDS)
                continue
            rows, apps = query_gpus()
            reserved = baseline_reserved | others | claimed_uuids(state)
            free = select_free(rows, apps, reserved)
        except (OSError, subprocess.SubprocessError, RuntimeError, json.JSONDecodeError, ValueError) as exc:
            log(f"waiting {pending['frequency']}; GPU or baseline query failed: {exc}")
            time.sleep(POLL_SECONDS)
            continue
        if not free:
            log(f"waiting {pending['frequency']}; {reason}; no idle unreserved GPU")
            time.sleep(POLL_SECONDS)
            continue
        if not start_task(state, pending, free[0]):
            time.sleep(POLL_SECONDS)
            continue
        time.sleep(1)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["supervise"]:
        if len(args) != 2:
            raise SystemExit("usage: run_mstt_state_aux_gate_split_mlp_p5_p15.py supervise PAYLOAD")
        return run_supervisor(Path(args[1]))
    if args == ["status"]:
        return print_status()
    if args not in ([], ["serve"]):
        raise SystemExit("usage: run_mstt_state_aux_gate_split_mlp_p5_p15.py [serve|status|supervise PAYLOAD]")
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
