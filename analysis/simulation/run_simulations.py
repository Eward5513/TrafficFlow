"""
跑完整的 20 天 SUMO 仿真，缺什么数据就先补什么，每天一个输出文件夹。

这是整条仿真链路的唯一入口，跑之前会自己把上游数据补齐，不用再手工一步步调那两个
生成脚本：

1. `data/random_trip/` 缺哪天的背景随机流，就调 `generate_random_routes.py` 补；
2. `data/route_demand/` 缺哪天的日需求，就调 `merge_basic_routes.py` 补；
3. 再拿日需求加 `analysis/road_network/net_tls.net.xml` 跑 SUMO。

最终真正调 SUMO 时，每天就是在当天输出目录里执行一条命令。以第 1 天（种子 650083）
为例，等价于：

    python3 analysis/simulation/run_simulations.py --days 1

脚本会先补齐上游数据、写出 `data/simulation/1_650083/simulation.sumocfg`，然后在该
目录里跑：

    cd analysis/simulation/data/simulation/1_650083
    sumo -c simulation.sumocfg

`simulation.sumocfg` 里加载的路径（均相对该文件夹）：

    路网  ../../../../road_network/net_tls.net.xml
    需求  ../../route_demand/1_650083.rou.xml
    信号  ../../../../road_network/tls_schedule.add.xml
    时间  begin=0, end=86400（24 小时）

两个上游脚本本身都会跳过已经生成好的天，所以重复调用不会重做已有的数据；本脚本也
只在真的缺文件时才去调它们。第 1 步需要 SUMO 环境（`--validate` 会拉起
`duarouter`），因此 `SUMO_HOME` 必须指向 SUMO 安装目录。

仿真还会挂上 `analysis/road_network/tls_schedule.add.xml`（offpeak / evening_peak 两
套配时加一张 WAUT 分时切换表），所以跑出来的是带分时信号方案的全天交通。

产物放在 `data/simulation/<编号>_<种子>/` 下，每天一个文件夹：

- `simulation.sumocfg` —— 本脚本生成的配置，路径全部相对该文件夹；
- `fcd.csv.gz`         —— 逐秒轨迹，SUMO 按 `.gz` 后缀直接压缩写出；
- `tripinfo.xml`       —— 每辆车的出行汇总；
- `vehroute.xml`       —— 实际走过的路径，含进出边时刻；
- `summary.xml`        —— 逐仿真步的总量曲线；
- `statistics.xml`     —— 全局统计；
- `simulation.log`     —— 该次 SUMO 调用的完整输出。

编号与种子沿用文件名词干，SUMO 的 `--seed` 就取当天的种子，因此同一天重跑结果
一致、不同天互相独立。仿真窗口固定为 0..86400 秒（24 小时），只记录这一时段的
数据；`--end` 可改成更短的秒数做冒烟测试。

已经跑完（五个输出文件都在且非空）的天默认跳过，`--force` 才会重跑并覆盖。脚本
不会删除任何目录。

`--days` 默认是 20（完整跑 1..20 天）。传单个数字 N 代表跑 1..N 天，例如 `--days 1`
就只跑第 1 天，非常适合先单独跑一天看看效果；也可以传多个数字如 `--days 1 2 3`
指定跑这几天。

注意背景随机流那一步只能按 `1..N` 连续生成，如果指定跑第 5 天，脚本仍然会把第
1..5 天的随机流补齐（已有的会跳过），但日需求和仿真只针对指定的那几天。

用法：

    python3 analysis/simulation/run_simulations.py                 # 完整 20 天，4 并发
    python3 analysis/simulation/run_simulations.py --days 1        # 只跑第 1 天看效果
    python3 analysis/simulation/run_simulations.py --days 3        # 跑前 3 天
    python3 analysis/simulation/run_simulations.py --days 1 2 3    # 指定跑第 1、2、3 天
    python3 analysis/simulation/run_simulations.py --jobs 1        # 串行
    python3 analysis/simulation/run_simulations.py --generate-only # 备齐数据和 sumocfg，不跑 SUMO
    python3 analysis/simulation/run_simulations.py --days 1 --end 3600 --force  # 冒烟测试
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR.parent

# ---- 硬编码配置 ----
NET_FILE = ANALYSIS_DIR / "road_network" / "net_tls.net.xml"
# 分时信号方案 + WAUT 切换表；不挂它就只有路网里的 programID="0"
TLS_SCHEDULE_FILE = ANALYSIS_DIR / "road_network" / "tls_schedule.add.xml"
DATA_DIR = BASE_DIR / "data"
# 背景随机流目录，同时也是编号与种子的唯一来源
RANDOM_DIR = DATA_DIR / "random_trip"
# 日需求目录，仿真真正加载的 route 文件
DEMAND_DIR = DATA_DIR / "route_demand"
# 每天一个子文件夹的仿真输出根目录
SIMULATION_DIR = DATA_DIR / "simulation"

# 缺数据时按需拉起的两个上游脚本
GENERATE_RANDOM_SCRIPT = BASE_DIR / "generate_random_routes.py"
MERGE_DEMAND_SCRIPT = BASE_DIR / "merge_basic_routes.py"

# 默认的天数，与 generate_random_routes.py 的 DAY_COUNT 保持一致
DAY_COUNT = 20

CONFIG_NAME = "simulation.sumocfg"
LOG_NAME = "simulation.log"
# 按 .gz 后缀让 SUMO 直接写压缩流，全天逐秒 FCD 不压缩要几十 GB
FCD_NAME = "fcd.csv.gz"
TRIPINFO_NAME = "tripinfo.xml"
VEHROUTE_NAME = "vehroute.xml"
SUMMARY_NAME = "summary.xml"
STATISTICS_NAME = "statistics.xml"
OUTPUT_NAMES = (FCD_NAME, TRIPINFO_NAME, VEHROUTE_NAME, SUMMARY_NAME, STATISTICS_NAME)

BEGIN_SECONDS = 0
END_SECONDS = 86400
TIME_TO_TELEPORT = 300
FCD_PERIOD = 1
FCD_ATTRIBUTES = "id,x,y,angle,type,speed,pos,lane"
# 全部车辆都装 rerouting device，仿真过程中按路况动态改路
REROUTING_PROBABILITY = 1.0
REROUTING_PERIOD = 60
REROUTING_ADAPTATION_INTERVAL = 10
REROUTING_ADAPTATION_STEPS = 30
# 每个进程单线程，靠 --jobs 并行不同的天
SUMO_THREADS = 1
DEFAULT_JOBS = 4
DEFAULT_SUMO_BINARY = "sumo"

ROUTE_SUFFIX = ".rou.xml"
STEM_PATTERN = re.compile(r"^(\d+)_(\d+)$")


@dataclass(slots=True)
class DayRun:
    """一天的仿真：编号、种子、以及当天的需求 route 文件。"""

    index: int
    seed: int
    route_file: Path

    @property
    def stem(self) -> str:
        return f"{self.index}_{self.seed}"

    @property
    def output_dir(self) -> Path:
        return SIMULATION_DIR / self.stem

    @property
    def config_file(self) -> Path:
        return self.output_dir / CONFIG_NAME

    @property
    def log_file(self) -> Path:
        return self.output_dir / LOG_NAME

    @property
    def output_files(self) -> tuple[Path, ...]:
        return tuple(self.output_dir / name for name in OUTPUT_NAMES)


def looks_complete(route_file: Path) -> bool:
    """粗判一份 rou.xml 是否写完整（结尾有 </routes>），避免拿中断的需求去跑仿真。"""
    if not route_file.is_file():
        return False
    size = route_file.stat().st_size
    if size == 0:
        return False
    with route_file.open("rb") as handle:
        handle.seek(max(0, size - 4096))
        return b"</routes>" in handle.read()


def scan_seeds(directory: Path) -> dict[int, int]:
    """扫 `<编号>_<种子>.rou.xml`，返回 {编号: 种子}。

    没写完整的文件当作不存在，这样中断留下的半截产物会被上游脚本重做。
    """
    found: dict[int, int] = {}
    if not directory.is_dir():
        return found

    for route_file in sorted(directory.glob(f"*_*{ROUTE_SUFFIX}")):
        match = STEM_PATTERN.match(route_file.name[: -len(ROUTE_SUFFIX)])
        if match is None:
            continue
        index, seed = int(match.group(1)), int(match.group(2))
        if not looks_complete(route_file):
            print(f"[warn] ignoring incomplete route file: {route_file.name}")
            continue
        if found.get(index, seed) != seed:
            raise SystemExit(
                f"day {index} maps to more than one seed in {directory} "
                f"({found[index]} and {seed}); keep exactly one file per day index"
            )
        found[index] = seed
    return found


def run_upstream_script(script: Path, arguments: list[str]) -> None:
    """同步跑一个上游生成脚本，输出直接透传，失败就整体退出。"""
    if not script.is_file():
        raise SystemExit(f"upstream script not found: {script}")

    command = [sys.executable, str(script), *arguments]
    print(f"\n$ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=BASE_DIR)
    if completed.returncode != 0:
        raise SystemExit(f"{script.name} failed with code {completed.returncode}")


def ensure_random_trips(indices: list[int], jobs: int, generate: bool) -> dict[int, int]:
    """补齐背景随机流，返回 {编号: 种子}。

    `generate_random_routes.py` 只按 `1..N` 连续生成，所以这里按最大编号补齐；已经
    生成好的天由它自己跳过，不会重抽种子。
    """
    needed = max(indices)
    seeds = scan_seeds(RANDOM_DIR)
    missing = [index for index in range(1, needed + 1) if index not in seeds]
    if missing:
        if not generate:
            raise SystemExit(
                f"missing random background flow for day(s) {missing} in {RANDOM_DIR}; "
                "drop --no-upstream or run generate_random_routes.py first"
            )
        print(
            f"Missing random background flow for day(s) {missing}; "
            f"generating days 1..{needed}"
        )
        run_upstream_script(
            GENERATE_RANDOM_SCRIPT, ["--days", str(needed), "--jobs", str(jobs)]
        )
        seeds = scan_seeds(RANDOM_DIR)
        still_missing = [index for index in indices if index not in seeds]
        if still_missing:
            raise SystemExit(
                f"random background flow still missing for day(s) {still_missing} "
                f"after running {GENERATE_RANDOM_SCRIPT.name}"
            )
    return seeds


def demand_file(index: int, seed: int) -> Path:
    return DEMAND_DIR / f"{index}_{seed}{ROUTE_SUFFIX}"


def ensure_route_demand(
    indices: list[int],
    seeds: dict[int, int],
    generate: bool,
) -> None:
    """补齐这几天的日需求。

    只为缺的天调一次 `merge_basic_routes.py`：它每次都要重新解析并清洗 16 万条匹配
    轨迹，按天分开调会把那份开销重复很多遍。
    """
    missing = [
        index
        for index in indices
        if not looks_complete(demand_file(index, seeds[index]))
    ]
    if not missing:
        return

    if not generate:
        raise SystemExit(
            f"missing route demand for day(s) {missing} in {DEMAND_DIR}; "
            "drop --no-upstream or run merge_basic_routes.py first"
        )
    print(f"Missing route demand for day(s) {missing}; building them")
    run_upstream_script(
        MERGE_DEMAND_SCRIPT, ["--only", *[str(index) for index in missing]]
    )

    still_missing = [
        index
        for index in missing
        if not looks_complete(demand_file(index, seeds[index]))
    ]
    if still_missing:
        raise SystemExit(
            f"route demand still missing for day(s) {still_missing} "
            f"after running {MERGE_DEMAND_SCRIPT.name}"
        )


def outputs_complete(day_dir: Path) -> bool:
    return all(
        (day_dir / name).is_file() and (day_dir / name).stat().st_size > 0
        for name in OUTPUT_NAMES
    )


def scan_finished_days() -> dict[int, int]:
    """扫 `data/simulation/`，返回已经跑完的 {编号: 种子}。

    先认已完成的天，才能在补上游数据之前就把它们摘掉：日需求动辄 200 MB 一天，
    为一个早就跑完的天重新造一遍纯属浪费。
    """
    finished: dict[int, int] = {}
    if not SIMULATION_DIR.is_dir():
        return finished

    for day_dir in sorted(SIMULATION_DIR.iterdir()):
        if not day_dir.is_dir():
            continue
        match = STEM_PATTERN.match(day_dir.name)
        if match is None or not outputs_complete(day_dir):
            continue
        finished[int(match.group(1))] = int(match.group(2))
    return finished


def relative_to_config(target: Path, config_dir: Path) -> str:
    """把输入路径写成相对 sumocfg 所在目录的 POSIX 路径。

    SUMO 按配置文件位置解析相对路径，这样仓库整体搬家也不会失效。
    """
    return Path(os.path.relpath(target, config_dir)).as_posix()


def append_section(root: ET.Element, tag: str, values: list[tuple[str, str]]) -> None:
    section = ET.SubElement(root, tag)
    for name, value in values:
        ET.SubElement(section, name).set("value", value)


def build_config(day: DayRun, use_tls_schedule: bool, end: float) -> Path:
    """生成当天的 sumocfg，写在当天的输出文件夹里。"""
    config_dir = day.output_dir
    config_dir.mkdir(parents=True, exist_ok=True)

    input_values = [
        ("net-file", relative_to_config(NET_FILE, config_dir)),
        ("route-files", relative_to_config(day.route_file, config_dir)),
    ]
    if use_tls_schedule:
        input_values.append(
            ("additional-files", relative_to_config(TLS_SCHEDULE_FILE, config_dir))
        )

    time_values = [
        ("begin", str(BEGIN_SECONDS)),
        ("end", f"{end:g}"),
        ("time-to-teleport", str(TIME_TO_TELEPORT)),
    ]

    root = ET.Element("configuration")
    append_section(root, "input", input_values)
    append_section(
        root,
        "processing",
        [
            ("seed", str(day.seed)),
            ("threads", str(SUMO_THREADS)),
            ("device.rerouting.probability", str(REROUTING_PROBABILITY)),
            ("device.rerouting.period", str(REROUTING_PERIOD)),
            ("device.rerouting.adaptation-interval", str(REROUTING_ADAPTATION_INTERVAL)),
            ("device.rerouting.adaptation-steps", str(REROUTING_ADAPTATION_STEPS)),
        ],
    )
    append_section(root, "time", time_values)
    append_section(
        root,
        "output",
        [
            ("fcd-output", FCD_NAME),
            ("fcd-output.geo", "true"),
            ("device.fcd.period", str(FCD_PERIOD)),
            ("fcd-output.attributes", FCD_ATTRIBUTES),
            ("output.column-header", "plain"),
            ("output.column-separator", ","),
            ("tripinfo-output", TRIPINFO_NAME),
            ("vehroute-output", VEHROUTE_NAME),
            ("vehroute-output.exit-times", "true"),
            ("vehroute-output.sorted", "true"),
            ("vehroute-output.write-unfinished", "true"),
            ("summary-output", SUMMARY_NAME),
            ("statistic-output", STATISTICS_NAME),
            # statistics.xml 里的 vehicleTripStatistics 需要它才会写出来
            ("duration-log.statistics", "true"),
        ],
    )
    # 全天仿真的逐步日志会把 log 撑到几百 MB，且没有分析价值
    append_section(root, "report", [("no-step-log", "true")])

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(day.config_file, encoding="utf-8", xml_declaration=False)
    return day.config_file


def run_day(day: DayRun, sumo_binary: str) -> int:
    """在当天的输出文件夹里调 SUMO，返回退出码。"""
    command = [sumo_binary, "-c", CONFIG_NAME]
    with day.log_file.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"Running: {' '.join(command)}\n")
        log_handle.write(f"Working directory: {day.output_dir}\n\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=day.output_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single entry point for the daily simulations: generate any missing "
            "random background flow and route demand, then run one full SUMO "
            "simulation per day into data/simulation/<index>_<seed>/."
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        nargs="+",
        default=[DAY_COUNT],
        metavar="N",
        help=(
            f"Days to cover. Pass a single number N to run days 1..N (defaults "
            f"to {DAY_COUNT}, e.g. --days 1 to test day 1), or list specific day "
            "indices (e.g. --days 1 2 3)."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=(
            "How many SUMO processes to run at once, also passed to "
            "generate_random_routes.py. Each one loads the whole network and "
            f"writes its own FCD stream. Defaults to {DEFAULT_JOBS}."
        ),
    )
    parser.add_argument(
        "--sumo-binary",
        default=DEFAULT_SUMO_BINARY,
        help=f"SUMO executable to call. Defaults to {DEFAULT_SUMO_BINARY!r}.",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help=(
            "Stop the simulation at this second instead of the default "
            f"{END_SECONDS} s (24 h). Only for smoke tests; it truncates the day."
        ),
    )
    parser.add_argument(
        "--no-tls-schedule",
        action="store_true",
        help=(
            "Do not load road_network/tls_schedule.add.xml, so every junction stays "
            "on the network's programID=0 all day."
        ),
    )
    parser.add_argument(
        "--no-upstream",
        action="store_true",
        help=(
            "Fail instead of generating missing random background flow or route "
            "demand. Use it when the upstream data must not be rebuilt."
        ),
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help=(
            "Prepare the upstream data and write the per-day sumocfg files without "
            "starting SUMO."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run days whose outputs already exist, overwriting them.",
    )
    return parser.parse_args()


def resolve_indices(days_arg: list[int]) -> list[int]:
    """把 --days 参数解析成要处理的天编号列表。

    - 单个整数 N：代表 1..N 天（例如 --days 1 跑第 1 天，默认 20 跑 1..20 天）；
    - 多个整数：代表指定的这几天（例如 --days 1 2 3）。
    """
    if not days_arg:
        raise SystemExit("--days must specify at least one day")

    if any(day < 1 for day in days_arg):
        raise SystemExit("--days indices must be at least 1")

    if len(days_arg) == 1:
        return list(range(1, days_arg[0] + 1))

    return sorted(set(days_arg))


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if args.end is not None and args.end <= BEGIN_SECONDS:
        raise SystemExit(f"--end must be greater than {BEGIN_SECONDS}")
    simulation_end = args.end if args.end is not None else END_SECONDS

    indices = resolve_indices(args.days)
    use_tls_schedule = not args.no_tls_schedule
    required = [NET_FILE]
    if use_tls_schedule:
        required.append(TLS_SCHEDULE_FILE)
    for path in required:
        if not path.exists():
            raise SystemExit(f"required input not found: {path}")

    generate = not args.no_upstream
    print(f"Target day(s): {indices}")

    # --force 要重跑，--generate-only 只备料，两种情况都不摘已完成的天
    if args.force or args.generate_only:
        targets = indices
    else:
        finished = scan_finished_days()
        for index in indices:
            if index in finished:
                print(
                    f"Skipped day {index} (seed {finished[index]}), "
                    "simulation output already exists"
                )
        targets = [index for index in indices if index not in finished]
    if not targets:
        print("\nNothing to do.")
        return

    print("\n== Checking random background flow ==")
    seeds = ensure_random_trips(targets, args.jobs, generate)
    for index in targets:
        print(f"  day {index}: seed {seeds[index]}")

    print("\n== Checking route demand ==")
    ensure_route_demand(targets, seeds, generate)

    print("\n== Preparing simulations ==")
    pending = [
        DayRun(
            index=index,
            seed=seeds[index],
            route_file=demand_file(index, seeds[index]),
        )
        for index in targets
    ]
    for day in pending:
        config_file = build_config(day, use_tls_schedule, simulation_end)
        print(f"Generated {config_file.relative_to(BASE_DIR)}")

    if args.generate_only:
        print("\n--generate-only: no simulation started.")
        return

    print(
        f"\nRunning {len(pending)} day(s) with {args.jobs} parallel job(s) "
        f"using {args.sumo_binary!r}"
    )
    for day in pending:
        print(f"  day {day.index}: seed {day.seed} -> {day.output_dir.name}/")

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        return_codes = list(
            executor.map(lambda day: run_day(day, args.sumo_binary), pending)
        )

    print("\n== Summary ==")
    failed: list[DayRun] = []
    for day, return_code in zip(pending, return_codes):
        log_path = day.log_file.relative_to(BASE_DIR)
        if return_code != 0:
            failed.append(day)
            print(
                f"Failed day {day.index} (seed {day.seed}) with code {return_code}, "
                f"log: {log_path}"
            )
            continue
        missing = [path.name for path in day.output_files if not path.is_file()]
        if missing:
            failed.append(day)
            print(
                f"Failed day {day.index} (seed {day.seed}): missing output "
                f"{', '.join(missing)}, log: {log_path}"
            )
            continue
        print(f"Finished day {day.index} (seed {day.seed}) -> {day.output_dir}")

    if failed:
        indices = ", ".join(str(day.index) for day in failed)
        raise SystemExit(f"SUMO failed for day(s): {indices}")


if __name__ == "__main__":
    main()
