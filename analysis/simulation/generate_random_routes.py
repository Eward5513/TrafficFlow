"""
为 20 天日需求数据批量生成随机背景流，每天一份、随机种子各不相同。

产物放在 `data/random_trip/` 下，文件名为 `<编号>_<种子>`（编号 1..20 不补零）：

- `<编号>_<种子>.trips.xml` —— randomTrips.py 生成的原始 trip；
- `<编号>_<种子>.rou.xml`   —— duarouter `--validate` 排好的可行路由，3 万条；
- `<编号>_<种子>.vtype.xml` —— 当天独享的 vType，避免并行 duarouter 抢写同一文件；
- `logs/<编号>_<种子>.log`  —— 该次调用的完整输出。

种子每天随机抽取（默认取系统熵，`--meta-seed` 可让抽取本身复现），抽到的值
直接写进文件名永久留档。`merge_basic_routes.py` 从这些文件名读取每天的编号与
种子，因此两个脚本之间不靠公式约定，改动这里的命名规则会同时影响下游。若某天
的 rou.xml 不完整，会先删掉该编号下所有残余文件（任意种子的 trips / rou / log /
.bak），再重新生成。

依赖 SUMO：需要 `SUMO_HOME` 指向 SUMO 安装目录，且 `duarouter` 可执行
（`--validate` 会调用它）。本脚本只负责生成随机流，不跑仿真。

用法：

    python3 analysis/simulation/generate_random_routes.py            # 20 天，4 并发
    python3 analysis/simulation/generate_random_routes.py --days 1   # 先试一天
    python3 analysis/simulation/generate_random_routes.py --jobs 1   # 串行
"""

from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR.parent
NET_FILE = ANALYSIS_DIR / "road_network" / "net_tls.net.xml"
DATA_DIR = BASE_DIR / "data"
RANDOM_DIR = DATA_DIR / "random_trip"
LOG_DIR = RANDOM_DIR / "logs"

# ---- 硬编码配置 ----
DAY_COUNT = 20
# 每天的背景流条数；--period 由此反推，保证恰好这么多条
TRIP_COUNT = 30000
BEGIN_SECONDS = 0
END_SECONDS = 86400
MIN_DISTANCE = 2000
VEHICLE_CLASS = "passenger"
# 车辆 id 前缀必须保持以 random 开头，用来和匹配轨迹的 VIN 区分。
VEHICLE_PREFIX = "random"
# 车辆 type 必须是这个 id；merge_basic_routes.py 合并日需求时也会手动补同一条。
# 不能把共享文件传给 --vtype：randomTrips --validate 会把它同时交给 duarouter
# 的 additional-files 和 vtype-output，并行任务会把同一个 XML 写坏。
VTYPE_ID = "random_passenger"
VTYPE_CLASS = VEHICLE_CLASS
VTYPE_SUFFIX = ".vtype.xml"
SEED_MIN = 1
SEED_MAX = 999999
DEFAULT_JOBS = 4

STEM_PATTERN = re.compile(r"^(\d+)_(\d+)$")
ROUTE_SUFFIX = ".rou.xml"
TRIPS_SUFFIX = ".trips.xml"


@dataclass(slots=True)
class DayJob:
    index: int
    seed: int

    @property
    def stem(self) -> str:
        return f"{self.index}_{self.seed}"

    @property
    def route_file(self) -> Path:
        return RANDOM_DIR / f"{self.stem}{ROUTE_SUFFIX}"

    @property
    def trips_file(self) -> Path:
        return RANDOM_DIR / f"{self.stem}{TRIPS_SUFFIX}"

    @property
    def log_file(self) -> Path:
        return LOG_DIR / f"{self.stem}.log"

    @property
    def vtype_file(self) -> Path:
        return RANDOM_DIR / f"{self.stem}{VTYPE_SUFFIX}"


def find_random_trips_tool() -> Path:
    """定位 $SUMO_HOME/tools/randomTrips.py，找不到就直接退出。"""
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise SystemExit(
            "SUMO_HOME is not set. Point it at the SUMO installation, for example "
            "export SUMO_HOME=/usr/share/sumo"
        )
    tool = Path(sumo_home) / "tools" / "randomTrips.py"
    if not tool.is_file():
        raise SystemExit(f"randomTrips.py not found under SUMO_HOME: {tool}")
    return tool


def looks_complete(route_file: Path) -> bool:
    """粗判一份 rou.xml 是否写完整（结尾有 </routes>），避免把中断的产物当成已完成。"""
    if not route_file.is_file():
        return False
    size = route_file.stat().st_size
    if size == 0:
        return False
    with route_file.open("rb") as handle:
        handle.seek(max(0, size - 4096))
        return b"</routes>" in handle.read()


def write_vtype_file(path: Path) -> None:
    """写出当天独享的 vType 文件，供 duarouter 当 additional / vtype-output。"""
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<additional>\n"
        f'    <vType id="{VTYPE_ID}" vClass="{VTYPE_CLASS}"/>\n'
        "</additional>\n",
        encoding="utf-8",
    )


def remove_day_remnants(random_dir: Path, log_dir: Path, index: int) -> list[Path]:
    """删掉某编号下所有失败或中断留下的产物（任意种子）。

    包括半截 rou.xml、孤立的 trips.xml / vtype.xml、log，以及 randomTrips 留下的 .bak。
    """
    removed: list[Path] = []
    prefix = f"{index}_"
    for directory in (random_dir, log_dir):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(f"{prefix}*")):
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


def discover_existing_days(random_dir: Path) -> dict[int, int]:
    """扫已生成的随机流，返回 {编号: 种子}。

    未写完整的文件会被忽略并给出警告，让本次运行重新生成该编号。
    """
    found: dict[int, int] = {}
    if not random_dir.is_dir():
        return found

    for route_file in sorted(random_dir.glob(f"*_*{ROUTE_SUFFIX}")):
        match = STEM_PATTERN.match(route_file.name[: -len(ROUTE_SUFFIX)])
        if match is None:
            continue
        index, seed = int(match.group(1)), int(match.group(2))
        if not looks_complete(route_file):
            print(f"[warn] ignoring incomplete random flow: {route_file.name}")
            continue
        if index in found:
            raise SystemExit(
                f"day {index} has more than one random flow file in {random_dir}; "
                "remove the extra one so the index maps to a single seed"
            )
        found[index] = seed
    return found


def draw_seeds(
    indices: list[int],
    used_seeds: set[int],
    meta_seed: int | None,
) -> dict[int, int]:
    """给缺失的编号各抽一个未被占用的种子。"""
    rng = random.Random(meta_seed)
    taken = set(used_seeds)
    seeds: dict[int, int] = {}
    for index in indices:
        seed = rng.randint(SEED_MIN, SEED_MAX)
        while seed in taken:
            seed = rng.randint(SEED_MIN, SEED_MAX)
        taken.add(seed)
        seeds[index] = seed
    return seeds


def build_command(tool: Path, job: DayJob) -> list[str]:
    period = (END_SECONDS - BEGIN_SECONDS) / TRIP_COUNT
    return [
        sys.executable,
        str(tool),
        "-n",
        str(NET_FILE),
        "-o",
        str(job.trips_file),
        "-r",
        str(job.route_file),
        "--begin",
        str(BEGIN_SECONDS),
        "--end",
        str(END_SECONDS),
        "--period",
        f"{period:g}",
        "--validate",
        "--min-distance",
        str(MIN_DISTANCE),
        "--vehicle-class",
        VEHICLE_CLASS,
        "--prefix",
        VEHICLE_PREFIX,
        "--vtype",
        str(job.vtype_file),
        "--seed",
        str(job.seed),
    ]


def run_job(tool: Path, job: DayJob) -> int:
    write_vtype_file(job.vtype_file)
    command = build_command(tool, job)
    with job.log_file.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"Running: {' '.join(command)}\n\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=BASE_DIR,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    return completed.returncode


def count_vehicles(route_file: Path) -> int:
    """流式统计 rou.xml 里的 <vehicle 个数，跨块边界不漏计。"""
    needle = b"<vehicle "
    overlap = len(needle) - 1
    total = 0
    tail = b""
    with route_file.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            buffer = tail + chunk
            total += buffer.count(needle)
            tail = buffer[-overlap:]
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one random background flow per day with a distinct random seed, "
            "written to data/random_trip/<index>_<seed>.rou.xml."
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DAY_COUNT,
        help=f"Number of days to cover, indices 1..N. Defaults to {DAY_COUNT}.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=(
            "How many randomTrips.py processes to run at once. Each one starts a "
            f"duarouter that loads the whole network. Defaults to {DEFAULT_JOBS}."
        ),
    )
    parser.add_argument(
        "--meta-seed",
        type=int,
        default=None,
        help=(
            "Seed for drawing the per-day seeds. Omit for system entropy; the drawn "
            "seeds are recorded in the output filenames either way."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.days < 1:
        raise SystemExit("--days must be at least 1")
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    tool = find_random_trips_tool()
    if not NET_FILE.exists():
        raise SystemExit(f"network file not found: {NET_FILE}")

    existing = discover_existing_days(RANDOM_DIR)
    wanted = list(range(1, args.days + 1))
    missing = [index for index in wanted if index not in existing]
    for index in wanted:
        if index in existing:
            print(f"Skipped day {index}, seed {existing[index]} already generated")

    if not missing:
        print("All requested days already exist, nothing to do.")
        return

    for index in missing:
        removed = remove_day_remnants(RANDOM_DIR, LOG_DIR, index)
        if removed:
            print(f"Removed {len(removed)} remnant file(s) for day {index}:")
            for path in removed:
                print(f"  {path.relative_to(BASE_DIR)}")

    seeds = draw_seeds(missing, set(existing.values()), args.meta_seed)
    jobs = [DayJob(index=index, seed=seeds[index]) for index in missing]

    RANDOM_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating {len(jobs)} day(s) of {TRIP_COUNT} random trips "
        f"with {args.jobs} parallel job(s)"
    )
    for job in jobs:
        print(f"  day {job.index}: seed {job.seed} -> {job.route_file.name}")

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        return_codes = list(executor.map(lambda job: run_job(tool, job), jobs))

    failed: list[DayJob] = []
    for job, return_code in zip(jobs, return_codes):
        if return_code != 0:
            failed.append(job)
            print(
                f"Failed day {job.index} (seed {job.seed}) with code {return_code}, "
                f"log: {job.log_file.relative_to(BASE_DIR)}"
            )
        elif not looks_complete(job.route_file):
            failed.append(job)
            print(
                f"Failed day {job.index} (seed {job.seed}): route file is incomplete, "
                f"log: {job.log_file.relative_to(BASE_DIR)}"
            )
        else:
            print(
                f"Finished day {job.index} (seed {job.seed}): "
                f"{count_vehicles(job.route_file)} vehicles in {job.route_file.name}"
            )

    if failed:
        indices = ", ".join(str(job.index) for job in failed)
        raise SystemExit(f"randomTrips failed for day(s): {indices}")


if __name__ == "__main__":
    main()
