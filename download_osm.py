#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# 你的 poly（按你给的坐标顺序原样保留）
POLY = "31.2036766 121.1236249 31.2036766 121.3644594 31.3642665 121.3644594 31.3642665 121.1236249"

# 输出文件（OSM XML，SUMO 直接可用）
OUT_FILE = "map.osm.xml"

# Overpass QL：道路 ways + 转向限制 relations + 闭包补全（nodes / referenced ways）
QUERY = f"""
[out:xml][timeout:3000];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified"](poly:"{POLY}");
  relation["type"="restriction"](poly:"{POLY}");
);
(._; >;);
out body;
"""

def main() -> None:
    resp = requests.post(
        OVERPASS_URL,
        data={"data": QUERY},
        timeout=3000,  # HTTP 连接超时；不是 Overpass 的 [timeout:*]
    )
    resp.raise_for_status()

    with open(OUT_FILE, "wb") as f:
        f.write(resp.content)

    print(f"Saved OSM XML to: {OUT_FILE}")

if __name__ == "__main__":
    main()
