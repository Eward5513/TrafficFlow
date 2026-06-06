#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# 你的 poly（按你给的坐标顺序原样保留）
POLY = "31.2036766 121.1236249 31.2036766 121.3644594 31.3642665 121.3644594 31.3642665 121.1236249"

# 输出文件
OUT_XML_FILE = "basemap.osm.xml"
OUT_GEOJSON_FILE = "basemap.geojson"

# Overpass QL：道路 ways + 转向限制 relations（过滤条件）
FILTER = f"""
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified"](poly:"{POLY}");
  relation["type"="restriction"](poly:"{POLY}");
)
"""


def build_xml_query() -> str:
    # 闭包补全 nodes / referenced ways，保证 OSM XML 在下游工具中可直接使用
    return f"""
[out:xml][timeout:3000];
{FILTER};
(._; >;);
out body;
"""


def build_geojson_query() -> str:
    # GeoJSON 只要几何信息，不做闭包补全以减少数据量
    return f"""
[out:json][timeout:3000];
{FILTER};
out body geom;
"""


def post_overpass(query: str, retries: int = 3, retry_backoff_s: int = 2) -> requests.Response:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=3000,  # HTTP 请求超时；不是 Overpass 的 [timeout:*]
            )
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(retry_backoff_s * attempt)

    raise RuntimeError("Unexpected retry loop exit")


def overpass_json_to_geojson(overpass_json: dict) -> dict:
    features = []

    for element in overpass_json.get("elements", []):
        element_type = element.get("type")
        tags = element.get("tags", {})

        if element_type == "way":
            geometry = element.get("geometry", [])
            if len(geometry) < 2:
                continue
            coordinates = [[pt["lon"], pt["lat"]] for pt in geometry]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coordinates},
                    "properties": {
                        "osm_type": "way",
                        "osm_id": element.get("id"),
                        "tags": tags,
                    },
                }
            )
            continue

        if element_type == "relation":
            member_lines = []
            for member in element.get("members", []):
                member_geom = member.get("geometry", [])
                if len(member_geom) >= 2:
                    member_lines.append([[pt["lon"], pt["lat"]] for pt in member_geom])

            if not member_lines:
                continue

            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": member_lines,
                    },
                    "properties": {
                        "osm_type": "relation",
                        "osm_id": element.get("id"),
                        "tags": tags,
                    },
                }
            )

    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    # 1) OSM XML
    xml_resp = post_overpass(build_xml_query())
    with open(OUT_XML_FILE, "wb") as f:
        f.write(xml_resp.content)

    # 2) GeoJSON
    json_resp = post_overpass(build_geojson_query())
    overpass_data = json_resp.json()
    geojson_data = overpass_json_to_geojson(overpass_data)
    with open(OUT_GEOJSON_FILE, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, ensure_ascii=False, indent=2)

    print(f"Saved OSM XML to: {OUT_XML_FILE}")
    print(f"Saved GeoJSON to: {OUT_GEOJSON_FILE}")
    print(f"GeoJSON features: {len(geojson_data['features'])}")


if __name__ == "__main__":
    main()
