#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

# ============================================================
# 固定参数区（你只需要改这里）
# ============================================================

# 你的路网文件（必须存在）
NET_FILE = Path("net.net.xml").resolve()

# 输出目录与文件名
OUT_DIR = Path(".").resolve()
ROUTES_FILE = OUT_DIR / "routes.rou.xml"
FCD_FILE = OUT_DIR / "fcd.xml"

# 仿真时间与随机行程生成强度
BEGIN = 0.0
END = 3600.0
TRIP_PERIOD = 1.0      # randomTrips.py 的 -p
SEED = 42

# 仿真步长（越小轨迹越细）
STEP_LENGTH = 0.1

# 关键控制：禁 teleport、碰撞不 teleport
TIME_TO_TELEPORT = -1          # -1 表示禁用 teleport
COLLISION_ACTION = "warn"      # 避免 teleport（teleport 会破坏 FIFO）

# 固定 SUMO_HOME（如你没设环境变量，这里可以直接写死）
# APT 安装一般为 /usr/share/sumo
SUMO_HOME = Path(os.environ.get("SUMO_HOME", "/usr/share/sumo")).resolve()

# 固定使用 sumo（不允许 GUI）
SUMO_BINARY = "sumo"

# 在生成 trips 时附加的属性。
# 注意：不要在 <trip> 上设置 sigma/speedDev —— 一些 SUMO 版本的 routes_file.xsd 不允许 trip 带这些属性，
# 会导致 duarouter 解析 trips.trips.xml 报错（attribute 'sigma' is not declared for element 'trip'）。
# 随机性控制应通过 <vType sigma="0" speedDev="0"> 来做（见 patch_routes_for_stability）。
TRIP_ATTRIBUTES = 'departLane="best" departPos="base" departSpeed="max"'

# ============================================================
# 实现逻辑（一般不需要改）
# ============================================================

def require_exists(p: Path, desc: str):
    if not p.exists():
        raise FileNotFoundError(f"{desc} not found: {p}")

def find_tools(sumo_home: Path) -> Path:
    tools = sumo_home / "tools"
    require_exists(tools, "SUMO tools directory")
    require_exists(tools / "randomTrips.py", "randomTrips.py")
    return tools

def check_binaries():
    # duarouter 通常由 sumo 包提供，randomTrips 会调用它
    if shutil.which("duarouter") is None:
        raise FileNotFoundError(
            "duarouter not found in PATH. Please install SUMO binaries "
            "(e.g., apt-get install sumo sumo-tools) or add SUMO bin to PATH."
        )
    if shutil.which(SUMO_BINARY) is None:
        raise FileNotFoundError(
            f"{SUMO_BINARY} not found in PATH. Please install SUMO (apt-get install sumo) "
            "or add SUMO bin to PATH."
        )

def run_random_trips(tools_dir: Path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(tools_dir / "randomTrips.py"),
        "-n", str(NET_FILE),
        "--begin", str(BEGIN),
        "--end", str(END),
        "-p", str(TRIP_PERIOD),
        "--seed", str(SEED),
        "--route-file", str(ROUTES_FILE),
        "--trip-attributes", TRIP_ATTRIBUTES
    ]

    print("[1/3] Generating routes via randomTrips.py")
    print("      ", " ".join(cmd))
    subprocess.run(cmd, check=True)

def patch_routes_for_stability(routes_file: Path):
    require_exists(routes_file, "Routes file")
    tree = ET.parse(routes_file)
    root = tree.getroot()

    # 1) 确保存在一个固定 vType（如果没有就创建）
    vtype_id = "car_stable"
    vtypes = root.findall("vType")
    has_target_vtype = any(vt.get("id") == vtype_id for vt in vtypes)

    if not has_target_vtype:
        vt = ET.Element("vType", {
            "id": vtype_id,
            "sigma": "0",
            "speedDev": "0",
        })
        # 尽量插到文件前部（美观+兼容性更好）
        root.insert(0, vt)
    else:
        for vt in vtypes:
            if vt.get("id") == vtype_id:
                vt.set("sigma", "0")
                vt.set("speedDev", "0")

    # 2) 给 vehicle/flow 补齐出发属性，并强制绑定到这个 vType
    for tag in ("vehicle", "flow"):
        for v in root.findall(tag):
            v.set("type", vtype_id)  # 强制所有车使用同一个稳定 vType
            if "departLane" not in v.attrib:
                v.set("departLane", "best")
            if "departPos" not in v.attrib:
                v.set("departPos", "base")
            if "departSpeed" not in v.attrib:
                v.set("departSpeed", "max")

    tree.write(routes_file, encoding="utf-8", xml_declaration=True)
    print(f"[1/3] Patched routes: {routes_file}")

    """
    补丁目的：
      - 强制 vType: sigma=0, speedDev=0（减少随机驾驶行为）
      - 强制 vehicle/flow: departLane/begin/pos/speed 等尽量齐全（减少禁换道后卡死概率）
    """
    require_exists(routes_file, "Routes file")
    tree = ET.parse(routes_file)
    root = tree.getroot()

    # patch vType randomness
    for vt in root.findall("vType"):
        vt.set("sigma", "0")
        vt.set("speedDev", "0")

    # patch vehicles & flows with depart settings if absent
    for tag in ("vehicle", "flow"):
        for v in root.findall(tag):
            if "departLane" not in v.attrib:
                v.set("departLane", "best")
            if "departPos" not in v.attrib:
                v.set("departPos", "base")
            if "departSpeed" not in v.attrib:
                v.set("departSpeed", "max")

    tree.write(routes_file, encoding="utf-8", xml_declaration=True)
    print(f"[1/3] Patched routes: {routes_file}")

def run_sumo_with_traci(tools_dir: Path):
    # 确保能 import traci
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import traci  # type: ignore

    cmd = [
        SUMO_BINARY,
        "-n", str(NET_FILE),
        "-r", str(ROUTES_FILE),
        "--begin", str(BEGIN),
        "--end", str(END),
        "--step-length", str(STEP_LENGTH),
        "--fcd-output", str(FCD_FILE),
        "--time-to-teleport", str(TIME_TO_TELEPORT),
        "--collision.action", str(COLLISION_ACTION),
        # 关键：不要加 --lateral-resolution（避免 sublane 单车道超越）
    ]

    print("[2/3] Running sumo (no GUI) via TraCI")
    print("      ", " ".join(cmd))
    traci.start(cmd)

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # 对本步新出发车辆：禁用自主换道（512）
            for vid in traci.simulation.getDepartedIDList():
                traci.vehicle.setLaneChangeMode(vid, 512)

    finally:
        traci.close()

    print(f"[2/3] FCD saved: {FCD_FILE}")

def main():
    require_exists(NET_FILE, "Network file")
    check_binaries()

    tools_dir = find_tools(SUMO_HOME)

    run_random_trips(tools_dir)
    patch_routes_for_stability(ROUTES_FILE)
    run_sumo_with_traci(tools_dir)

    print("[3/3] Done.")
    print("Routes:", ROUTES_FILE)
    print("FCD:   ", FCD_FILE)

if __name__ == "__main__":
    main()
