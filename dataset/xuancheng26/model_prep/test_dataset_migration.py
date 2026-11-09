"""Fixture checks for the fixed_uniform sampling_seeds migration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from migrate_xuancheng_training_datasets import MigrationError, migrate


def _build(root: Path) -> None:
    shared = root / "shared"
    shared.mkdir(parents=True)
    (shared / "split_manifest.json").write_text('{"train":["d"]}\n', encoding="utf-8")
    (shared / "node_mapping.csv").write_text("edge_id\nA\n", encoding="utf-8")
    (shared / "state_normalization.json").write_text("{}\n", encoding="utf-8")
    (shared / "state_window_manifest.json").write_text("{}\n", encoding="utf-8")
    (shared / "validation_summary.json").write_text("{}\n", encoding="utf-8")
    graph = root / "graph"
    graph.mkdir()
    (graph / "g.txt").write_text("graph\n", encoding="utf-8")
    seed = root / "sampling_seeds" / "seed_42"
    window = seed / "flow" / "window_data"
    day = window / "p10" / "train" / "d"
    day.mkdir(parents=True)
    np.savez(day / "inflow.npz", x=np.arange(4))
    (seed / "file.csv").write_text("vehicle_id\nv\n", encoding="utf-8")
    sampled = seed / "sampled_vehicle" / "p10"
    sampled.mkdir(parents=True)
    (sampled / "d.csv").write_text("vehicle_id\nv\n", encoding="utf-8")
    flow = seed / "flow" / "partial_flow" / "p10" / "d"
    flow.mkdir(parents=True)
    (flow / "inflow.csv").write_text("time_bin\n0\n", encoding="utf-8")
    (window / "split_manifest.json").symlink_to("../../../../shared/split_manifest.json")
    (window / "internal.txt").symlink_to("../../file.csv")
    (seed / "graph").symlink_to("../../graph")
    outside = root / "notes"
    outside.mkdir()
    (outside / "into").symlink_to("../sampling_seeds/seed_42/file.csv")


def test_migrate_is_idempotent_and_preserves_targets(tmp_path: Path) -> None:
    root = tmp_path / "training_data_3s"
    _build(root)
    preview = migrate(root, execute=False)
    assert preview["status"] == "dry_run"
    assert preview["counts"] == {"vehicle": 1, "state_csv": 1, "window_npz": 1}
    report = migrate(root, execute=True)
    assert report["status"] == "migrated"
    legacy = root / "sampling_seeds"
    dest = root / "datasets" / "fixed_uniform" / "sampling_seeds"
    assert legacy.is_symlink()
    assert legacy.resolve() == dest.resolve()
    assert (dest / "seed_42" / "flow" / "window_data" / "split_manifest.json").resolve() == (
        root / "shared" / "split_manifest.json"
    ).resolve()
    assert (dest / "seed_42" / "graph").resolve() == (root / "graph").resolve()
    assert (dest / "seed_42" / "flow" / "window_data" / "internal.txt").resolve() == (
        dest / "seed_42" / "file.csv"
    ).resolve()
    assert (root / "notes" / "into").resolve() == (dest / "seed_42" / "file.csv").resolve()
    with np.load(legacy / "seed_42" / "flow" / "window_data" / "p10" / "train" / "d" / "inflow.npz") as alias, np.load(
        dest / "seed_42" / "flow" / "window_data" / "p10" / "train" / "d" / "inflow.npz"
    ) as moved:
        assert np.array_equal(alias["x"], moved["x"])
    again = migrate(root, execute=True)
    assert again["status"] == "already_migrated"
    assert json.loads((root / "datasets" / "fixed_uniform" / "dataset_manifest.json").read_text())["dataset_id"] == "fixed_uniform"


def test_failure_after_rename_rolls_back(tmp_path: Path) -> None:
    root = tmp_path / "training_data_3s"
    _build(root)
    original = os.readlink(root / "sampling_seeds" / "seed_42" / "flow" / "window_data" / "split_manifest.json")
    with pytest.raises(MigrationError, match="injected failure after rename"):
        migrate(root, execute=True, fail_after="renamed")
    legacy = root / "sampling_seeds"
    assert legacy.is_dir() and not legacy.is_symlink()
    assert not (root / "datasets" / "fixed_uniform" / "sampling_seeds").exists()
    assert os.readlink(legacy / "seed_42" / "flow" / "window_data" / "split_manifest.json") == original
    assert (legacy / "seed_42" / "flow" / "window_data" / "split_manifest.json").resolve() == (
        root / "shared" / "split_manifest.json"
    ).resolve()
    journal = json.loads((root / "architecture_audit" / "dataset_layout_migration_journal.json").read_text())
    assert journal["status"] == "rolled_back"


def test_conflicting_destination_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "training_data_3s"
    _build(root)
    dest = root / "datasets" / "fixed_uniform" / "sampling_seeds"
    dest.mkdir(parents=True)
    (dest / "occupied").write_text("nope", encoding="utf-8")
    with pytest.raises(MigrationError, match="both"):
        migrate(root, execute=True)
    assert (root / "sampling_seeds" / "seed_42" / "file.csv").is_file()
