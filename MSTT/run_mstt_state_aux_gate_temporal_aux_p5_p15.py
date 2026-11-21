#!/usr/bin/env python3
"""Launch four AllSpC2All runs: speed gate plus the first-11-step auxiliary loss.

One frequency occupies one GPU. A second invocation keeps the existing run ids
and does not start another copy of a frequency that is already queued, running,
or finished. Each training process is capped at 15 threads.

State, the scheduler lock, and logs:
    <training_results_3s>/operations/mstt_state_aux_gate_temporal_aux11_p5_p15/
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_root import new_run_id, xuancheng26_training_results_3s_root  # noqa: E402
from MSTT.config import load_config  # noqa: E402
from MSTT.train import resolve_output_root  # noqa: E402
from MSTT.trainer import run_directory  # noqa: E402
from MSTT.utils.threads import MAX_THREADS, THREAD_ENV_KEYS  # noqa: E402
from reimplementation.common.data.sampling_seeds import configure_sampling  # noqa: E402

PYTHON = Path("/home/tzhang174/miniforge3/envs/trafficflow/bin/python3")
RESULTS = xuancheng26_training_results_3s_root()
OPS = RESULTS / "operations" / "mstt_state_aux_gate_temporal_aux11_p5_p15"
LOCK_PATH = OPS / "scheduler.lock"
STATE_PATH = OPS / "state.json"
LOG_PATH = OPS / "scheduler.log"
PID_PATH = OPS / "scheduler.pid"
GPU_LOCK_DIR = RESULTS / "operations" / "gpu_uuid_locks"
BASELINE_OPS = RESULTS / "operations"
CONFIG_REL = "MSTT/configs/experiment_state_aux_gate_temporal_aux_history11_projected.json"
EXPERIMENT_ID = "state_aux_gate_d128_temporal_aux11_l05_projected_all"
FREQUENCIES = (
    ("30min", "variable_30min_p5_p15"),
    ("1h", "variable_1h_p5_p15"),
    ("2h", "variable_2h_p5_p15"),
    ("day", "variable_day_p5_p15"),
)
POLL_SECONDS = 30
IDLE_MEMORY_MIB = 512.0
UTIL_LIMIT = 30.0
BOUND = frozenset({"starting", "running"})
CST = timezone(timedelta(hours=8))


def stamp() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)


def log(message: str) -> None:
    OPS.mkdir(parents=True, exist_ok=True)
    line = f"[{stamp()}] {message}\n"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def pid_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return Path(f"/proc/{pid}").is_dir()


def process_command(pid: int) -> str:
    path = Path(f"/proc/{pid}/cmdline")
    try:
        return path.read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def child_env(uuid: str, *, base: dict | None = None) -> dict[str, str]:
    """One visible GPU and a hard cap of 15 threads. Never leave the pool at 64."""
    if MAX_THREADS != 15:
        raise RuntimeError(f"thread cap must stay 15, got {MAX_THREADS}")
    env = dict(os.environ if base is None else base)
    env["CUDA_VISIBLE_DEVICES"] = uuid
    env["PYTHONUNBUFFERED"] = "1"
    env["MSTT_NUM_THREADS"] = str(MAX_THREADS)
    for key in THREAD_ENV_KEYS:
        env[key] = str(MAX_THREADS)
    return env


def planned_output(dataset_id: str, run_id: str) -> str:
    config = load_config(ROOT / CONFIG_REL)
    config["dataset_id"] = dataset_id
    config["run_id"] = run_id
    config["explicit_path_keys"] = ["dataset_id"]
    config = configure_sampling(config, model_name="MSTT")
    root = resolve_output_root(config)
    config["output_root"] = root.as_posix()
    return run_directory(root, "p5_p15", "AllSpC2All", 42, config).as_posix()


def fresh_state() -> dict:
    tasks = []
    for frequency, dataset_id in FREQUENCIES:
        run_id = new_run_id("MSTT", dataset_id)
        tasks.append({
            "frequency": frequency,
            "dataset_id": dataset_id,
            "experiment_id": EXPERIMENT_ID,
            "run_id": run_id,
            "status": "pending",
            "gpu_index": None,
            "gpu_uuid": None,
            "pid": None,
            "supervisor_pid": None,
            "log": str(OPS / "logs" / f"{run_id}.log"),
            "output_dir": planned_output(dataset_id, run_id),
            "error": None,
            "started_at": None,
        })
    return {
        "campaign": "state_aux_gate_temporal_aux11_p5_p15",
        "code_root": str(ROOT),
        "config": CONFIG_REL,
        "max_threads": MAX_THREADS,
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


class FileLock:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        self.path = path
        self.handle = path.open("a+", encoding="utf-8")
        self.held = False

    def try_acquire(self, note: str) -> bool:
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        self.held = True
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(note)
        self.handle.flush()
        return True

    def read_holder(self) -> str:
        self.handle.seek(0)
        return self.handle.read().strip()


class GpuClaim:
    def __init__(self, directory: Path, uuid: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{uuid.replace('/', '_')}.lock"
        self.handle = self.path.open("a+", encoding="utf-8")
        self.held = False

    def try_acquire(self, note: dict) -> bool:
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        self.held = True
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps(note, ensure_ascii=False) + "\n")
        self.handle.flush()
        return True

    def release(self) -> None:
        try:
            if self.held:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
                self.held = False
        finally:
            self.handle.close()


def query_gpus() -> tuple[list[dict], dict[str, list[int]]]:
    devices = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True, timeout=20,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        check=False, capture_output=True, text=True, timeout=20,
    )
    if devices.returncode != 0 or apps.returncode != 0:
        detail = (devices.stderr or apps.stderr or devices.stdout or apps.stdout).strip()
        raise RuntimeError(f"nvidia-smi failed: {detail}")
    rows = []
    for line in devices.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid, used, total, util = [part.strip() for part in line.split(",")]
        if used in {"[N/A]", "[Not Supported]"} or util in {"[N/A]", "[Not Supported]"}:
            rows.append({"index": int(index), "uuid": uuid, "indeterminate": True})
            continue
        rows.append({
            "index": int(index), "uuid": uuid, "memory_used": float(used),
            "memory_total": float(total), "utilization": float(util), "indeterminate": False,
        })
    by_uuid: dict[str, list[int]] = {row["uuid"]: [] for row in rows}
    for line in apps.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid_text = [part.strip() for part in line.split(",", 1)]
        if not pid_text.isdigit():
            raise RuntimeError(f"unreadable compute pid in {line!r}")
        by_uuid.setdefault(uuid, []).append(int(pid_text))
    return rows, by_uuid


def baseline_reserved_uuids() -> set[str]:
    reserved = set()
    if not BASELINE_OPS.is_dir():
        return reserved
    for campaign in BASELINE_OPS.iterdir():
        tasks_path = campaign / "tasks.json"
        lock = campaign / "scheduler.lock"
        if not tasks_path.is_file() or not lock.is_file():
            continue
        owner = lock.read_text(encoding="utf-8", errors="replace").strip()
        if not owner.isdigit() or not pid_alive(int(owner)):
            continue
        if "run_variable_baselines.py" not in process_command(int(owner)):
            continue
        payload, error = read_json(tasks_path)
        if error or not isinstance(payload, dict):
            continue
        for task in payload.get("tasks") or []:
            if task.get("status") != "running" or task.get("cpu"):
                continue
            uuid = str(task.get("gpu_uuid") or "")
            pid = int(task.get("pid") or 0)
            if uuid and pid_alive(pid):
                reserved.add(uuid)
    return reserved


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


def claimed_uuids(state: dict) -> set[str]:
    found = set()
    for task in state["tasks"]:
        if task["status"] in BOUND and task.get("gpu_uuid"):
            found.add(str(task["gpu_uuid"]))
    return found


def run_completed(path: str | None) -> bool:
    if not path:
        return False
    directory = Path(path)
    meta, error = read_json(directory / "run_metadata.json")
    return not error and isinstance(meta, dict) and meta.get("status") == "completed" and (directory / "test_sampling_metrics.json").is_file()


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
        pid = task.get("pid")
        if task["status"] in BOUND and pid_alive(pid) and task["run_id"] in process_command(int(pid)):
            task["status"] = "running"
            continue
        supervisor = task.get("supervisor_pid")
        if task["status"] == "starting" and pid_alive(supervisor):
            continue
        if task["status"] in BOUND:
            task["status"] = "pending"
            task["pid"] = None
            task["supervisor_pid"] = None
            task["gpu_uuid"] = None
            task["gpu_index"] = None
            task["error"] = "training process was not alive at scheduler start"


def next_pending(state: dict):
    for task in state["tasks"]:
        if task["status"] == "pending":
            return task
    return None


def run_supervisor(payload_path: Path) -> int:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    claim = GpuClaim(GPU_LOCK_DIR, payload["uuid"])
    if not claim.try_acquire({"supervisor_pid": os.getpid(), "run_id": payload["run_id"], "started_at": stamp()}):
        return 2
    try:
        rows, apps = query_gpus()
        reserved = baseline_reserved_uuids()
    except (OSError, subprocess.SubprocessError, RuntimeError, json.JSONDecodeError) as exc:
        claim.release()
        Path(payload["log"]).parent.mkdir(parents=True, exist_ok=True)
        Path(payload["log"]).write_text(f"supervisor recheck failed: {exc}\n", encoding="utf-8")
        return 5
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
    payload = {
        "uuid": gpu["uuid"],
        "run_id": task["run_id"],
        "command": train_command(task),
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
        task["error"] = tail[-800:] or "training process exited before test metrics were written"
        log(f"failed {task['frequency']} run_id={task['run_id']}")


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
    log(f"scheduler start pid={os.getpid()} threads={MAX_THREADS}")
    while True:
        reap(state)
        save_state(state)
        pending = next_pending(state)
        if pending is None and all(task["status"] in {"completed", "failed"} for task in state["tasks"]):
            log("scheduler idle; every frequency is completed or failed")
            save_state(state)
            return 0
        if pending is None:
            time.sleep(POLL_SECONDS)
            continue
        try:
            rows, apps = query_gpus()
            reserved = baseline_reserved_uuids() | claimed_uuids(state)
            free = select_free(rows, apps, reserved)
        except (OSError, subprocess.SubprocessError, RuntimeError, json.JSONDecodeError) as exc:
            log(f"GPU query failed: {exc}")
            time.sleep(POLL_SECONDS)
            continue
        if not free:
            log(f"waiting {pending['frequency']}; no idle unreserved GPU")
            time.sleep(POLL_SECONDS)
            continue
        start_task(state, pending, free[0])
        time.sleep(1)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["supervise"]:
        if len(args) != 2:
            raise SystemExit("usage: run_mstt_state_aux_gate_temporal_aux_p5_p15.py supervise PAYLOAD")
        return run_supervisor(Path(args[1]))
    if args not in ([], ["serve"]):
        raise SystemExit("usage: run_mstt_state_aux_gate_temporal_aux_p5_p15.py [serve|supervise PAYLOAD]")
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
