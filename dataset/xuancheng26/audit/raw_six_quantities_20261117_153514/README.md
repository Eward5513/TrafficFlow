# 六个原始量审计

阅读顺序：`REPORT.md`，然后 `definitions.json`，再按表格目录抽查。

本次范围：2023_04_01 至 2023_04_30，29 天，1406 条路，288 个 5 分钟段。全量一次；固定 p05–p70 与波动 p5_p15、p10_p20、p5_p35，种子 42–46。

运行命令见报告末尾。扫描脚本 `/tmp/xuancheng26_raw_audit_20261117/audit_scan.py`，汇总脚本 `/tmp/xuancheng26_raw_audit_20261117/audit_aggregate.py`。缓存 `/tmp/xuancheng26_raw_audit_20261117/cache`。cache_key `a5eb60206d91bbd3ebafbbc322d52dd94abdfcc220dbb11fbf9352bc7420e711`。

- `tables/marginal/` 边际分布
- `tables/temporal/` 时间
- `tables/roads/` 道路
- `tables/joint/` 联合
- `tables/speed_count/` C 与速度
- `tables/sampling/` 抽样对照
- `figures/` 图
- `quality/` 核对
- `logs/` 扫描进度
