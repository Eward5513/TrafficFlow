#!/usr/bin/env python3
"""Move Xuancheng26 sampling_seeds into datasets/fixed_uniform.

The move is a same-filesystem rename plus a compatibility symlink. Relative
links are rewritten from the journal so they keep the same real targets after
the directory drops two levels. Default is dry-run; ``--execute`` performs it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT))

from data_root import FIXED_DATASET_ID, FIXED_OBSERVATION_KEYS, data_root  # noqa: E402

PAYLOAD_MARKERS = ("sampled_vehicle", "partial_flow", "window_data")
BLOCK_MARKERS = (
    "train_r_only",
    "MSTT/train.py",
    "generate_r_level",
    "generate_multi_seed",
    "generate_variable_penetration",
    "migrate_xuancheng_training_datasets.py",
)


class MigrationError(RuntimeError):
    """A layout change that must be rolled back or refused."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk(root: Path):
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in list(dirnames):
            path = current / name
            if path.is_symlink():
                dirnames.remove(name)
                yield path
        for name in filenames:
            yield current / name


def _payload_kind(relative: str) -> str | None:
    parts = relative.split("/")
    if len(parts) >= 4 and parts[1] == "sampled_vehicle" and parts[-1].endswith(".csv"):
        return "vehicle"
    if "partial_flow" in parts and parts[-1].endswith(".csv") and parts[-1] != "generation_summary.csv":
        return "state_csv"
    if "window_data" in parts and parts[-1].endswith(".npz"):
        return "window_npz"
    return None


def _entry(src: Path, path: Path) -> dict:
    stat = path.lstat()
    kind = "symlink" if path.is_symlink() else "dir" if stat.st_mode & 0o170000 == 0o040000 else "file"
    record = {
        "rel": path.relative_to(src).as_posix(),
        "kind": kind,
        "size": stat.st_size,
        "dev": stat.st_dev,
        "ino": stat.st_ino,
    }
    if path.is_symlink():
        raw = os.readlink(path)
        immediate = Path(raw) if raw.startswith("/") else path.parent / raw
        lexical = Path(os.path.normpath(immediate))
        record["link"] = raw
        try:
            record["target_rel"] = lexical.relative_to(src).as_posix()
        except ValueError:
            record["target_rel"] = None
        if not lexical.exists():
            raise MigrationError(f"dangling symlink: {path} -> {raw}")
        record["target_real"] = str(lexical.resolve())
        # A lexical path can sit inside the tree while naming another symlink.
        # Keep that immediate relative path so the chain moves with the tree.
        if record["target_rel"] is not None and ".." in Path(record["target_rel"]).parts:
            record["target_rel"] = None
    return record


def _inventory(src: Path) -> dict:
    entries = []
    counts = {"vehicle": 0, "state_csv": 0, "window_npz": 0}
    for path in _walk(src):
        record = _entry(src, path)
        payload = _payload_kind(record["rel"]) if record["kind"] == "file" else None
        if payload:
            record["payload"] = payload
            counts[payload] += 1
        entries.append(record)
    return {"entries": entries, "counts": counts}


def _blocking(src: Path) -> list[dict]:
    hits = []
    own = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == own:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if not command.strip():
            continue
        blocked = any(marker in command for marker in BLOCK_MARKERS)
        open_hit = None
        try:
            for fd in (entry / "fd").iterdir():
                target = os.readlink(fd)
                if target.startswith(str(src)) or "/sampling_seeds/" in target:
                    open_hit = target
                    break
        except OSError:
            pass
        if blocked or open_hit:
            hits.append({"pid": pid, "command": command[:400], "open_path": open_hit})
    return hits


def _shared_hashes(root: Path) -> dict[str, str]:
    names = (
        "split_manifest.json",
        "node_mapping.csv",
        "state_normalization.json",
        "state_window_manifest.json",
        "validation_summary.json",
    )
    return {name: _sha256(root / "shared" / name) for name in names if (root / "shared" / name).is_file()}


def _reverse_links(root: Path, src: Path) -> list[dict]:
    found = []
    src_resolved = src.resolve()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        if current == src or src in current.parents:
            dirnames.clear()
            continue
        names = list(dirnames) + list(filenames)
        for name in names:
            path = current / name
            if not path.is_symlink():
                continue
            try:
                target = path.resolve()
                stat = target.stat()
            except OSError:
                continue
            if target == src_resolved or src_resolved in target.parents:
                found.append({
                    "path": str(path),
                    "link": os.readlink(path),
                    "target": str(target),
                    "dev": stat.st_dev,
                    "ino": stat.st_ino,
                })
        for name in list(dirnames):
            if (current / name).is_symlink():
                dirnames.remove(name)
    return found


def _spot_npz(src: Path) -> list[dict]:
    spots = []
    for path in _walk(src):
        if path.suffix == ".npz" and path.is_file() and not path.is_symlink():
            spots.append({"rel": path.relative_to(src).as_posix(), "sha256": _sha256(path)})
            if len(spots) == 3:
                break
    return spots


def _desired_link(record: dict, dest: Path) -> str:
    link_path = dest / record["rel"]
    if record.get("target_rel"):
        target = dest / record["target_rel"]
    else:
        target = Path(record["target_real"])
    return os.path.relpath(target, link_path.parent)


def _replace_link(path: Path, text: str) -> None:
    if path.is_symlink() and os.readlink(path) == text:
        return
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(text)


def _apply_links(dest: Path, entries: list[dict]) -> None:
    for record in entries:
        if record["kind"] != "symlink":
            continue
        _replace_link(dest / record["rel"], _desired_link(record, dest))


def _restore_links(dest: Path, entries: list[dict]) -> None:
    for record in entries:
        if record["kind"] != "symlink":
            continue
        path = dest / record["rel"]
        if path.is_symlink() or path.exists():
            _replace_link(path, record["link"])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _layout(root: Path) -> tuple[Path, Path, Path]:
    legacy = root / "sampling_seeds"
    dataset = root / "datasets" / FIXED_DATASET_ID
    dest = dataset / "sampling_seeds"
    return legacy, dataset, dest


def _already(root: Path) -> bool:
    legacy, dataset, dest = _layout(root)
    if not legacy.is_symlink() or not dest.is_dir() or dest.is_symlink():
        return False
    if legacy.resolve() != dest.resolve():
        return False
    shared = dataset / "shared"
    return shared.is_symlink() and shared.resolve() == (root / "shared").resolve() and (dataset / "dataset_manifest.json").is_file()


def _verify(root: Path, journal: dict) -> dict:
    legacy, dataset, dest = _layout(root)
    if not _already(root):
        raise MigrationError("fixed_uniform alias or shared link is not in the completed state")
    broken = []
    for path in _walk(dest):
        if path.is_symlink() and not path.exists():
            broken.append(str(path))
    if broken:
        raise MigrationError(f"broken symlinks after migration: {broken[:8]}")
    current = _inventory(dest)
    expected = journal.get("counts")
    if expected and current["counts"] != expected:
        raise MigrationError(f"payload counts changed: {current['counts']} != {expected}")
    by_rel = {item["rel"]: item for item in journal.get("entries", []) if item["kind"] == "file"}
    mismatches = []
    for item in current["entries"]:
        if item["kind"] != "file":
            continue
        previous = by_rel.get(item["rel"])
        if previous is None:
            continue
        if (item["size"], item["dev"], item["ino"]) != (previous["size"], previous["dev"], previous["ino"]):
            mismatches.append(item["rel"])
    if mismatches:
        raise MigrationError(f"file identity changed for {mismatches[:8]}")
    shared_now = _shared_hashes(root)
    if journal.get("shared_sha256") and shared_now != journal["shared_sha256"]:
        raise MigrationError("shared metadata hash changed during migration")
    for spot in journal.get("spot_npz") or []:
        path = dest / spot["rel"]
        via_alias = legacy / spot["rel"]
        if _sha256(path) != spot["sha256"] or _sha256(via_alias) != spot["sha256"]:
            raise MigrationError(f"spot npz changed: {spot['rel']}")
    for ref in journal.get("reverse_links") or []:
        path = Path(ref["path"])
        if not path.is_symlink() or not path.exists():
            raise MigrationError(f"reverse link no longer resolves: {path}")
        stat = path.resolve().stat()
        if (stat.st_dev, stat.st_ino) != (ref["dev"], ref["ino"]):
            raise MigrationError(f"reverse link no longer reaches the same file: {path}")
    for record in journal.get("entries", []):
        if record["kind"] != "symlink":
            continue
        path = dest / record["rel"]
        if not path.is_symlink() or not path.exists():
            raise MigrationError(f"migrated link is broken: {path}")
        if record.get("target_rel"):
            if path.resolve() != (dest / record["target_rel"]).resolve():
                raise MigrationError(f"internal link retarget failed: {path}")
        elif path.resolve() != Path(record["target_real"]).resolve():
            raise MigrationError(f"external link retarget failed: {path}")
    return {"ok": True, "counts": current["counts"], "alias": os.readlink(legacy), "shared": os.readlink(dataset / "shared")}


def _rollback(root: Path, journal: dict, created: dict) -> None:
    legacy, dataset, dest = _layout(root)
    if created.get("alias") and legacy.is_symlink():
        legacy.unlink()
    if dest.is_dir() and not dest.is_symlink():
        _restore_links(dest, journal.get("entries") or [])
        if legacy.exists() or legacy.is_symlink():
            raise MigrationError(f"cannot restore {legacy}; it still exists")
        os.rename(dest, legacy)
    if created.get("manifest") and (dataset / "dataset_manifest.json").is_file():
        (dataset / "dataset_manifest.json").unlink()
    shared = dataset / "shared"
    if created.get("shared") and shared.is_symlink():
        shared.unlink()


def _manifest(root: Path, legacy_name: str) -> dict:
    return {
        "dataset_id": FIXED_DATASET_ID,
        "kind": "fixed_uniform",
        "observation_keys": list(FIXED_OBSERVATION_KEYS),
        "sampling_seeds": [42, 43, 44, 45, 46],
        "migrated_from": legacy_name,
        "compatibility_alias": "sampling_seeds",
        "shared_alias": "datasets/fixed_uniform/shared -> ../../shared",
        "algorithm_version": "xuancheng26_3s_multi_seed_v2",
        "payload_rewritten": False,
        "note": "Path migration only. CSV and NPZ bytes were not rewritten.",
    }


def migrate(root: Path, *, execute: bool = False, fail_after: str | None = None) -> dict:
    root = Path(root).resolve()
    if fail_after and not str(root).startswith("/tmp"):
        raise MigrationError("fail-after is only allowed for /tmp fixtures")
    if CODE_ROOT == root or CODE_ROOT in root.parents:
        raise MigrationError(f"refusing to migrate inside CODE_ROOT: {root}")
    legacy, dataset, dest = _layout(root)
    journal_path = root / "architecture_audit" / "dataset_layout_migration_journal.json"
    lock_path = root / ".dataset_layout_migration.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        import fcntl

        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blockers = _blocking(legacy if legacy.exists() else dest)
        if blockers and execute:
            raise MigrationError(f"running tasks block migration: {blockers[:6]}")
        if _already(root):
            journal = json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.is_file() else {}
            report = _verify(root, journal)
            report["status"] = "already_migrated"
            report["blockers"] = blockers
            return report
        if legacy.is_dir() and not legacy.is_symlink() and dest.exists():
            raise MigrationError("both the legacy sampling_seeds directory and the fixed_uniform destination exist")
        if legacy.is_symlink():
            raise MigrationError(f"sampling_seeds is an unexpected symlink: {os.readlink(legacy)}")
        if not legacy.is_dir() and dest.is_dir():
            if not journal_path.is_file():
                raise MigrationError("destination exists without a journal; refusing to guess a half migration")
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        elif legacy.is_dir():
            journal = {
                "status": "inventoried",
                "source": str(legacy),
                "destination": str(dest),
                "shared_sha256": _shared_hashes(root),
                "reverse_links": _reverse_links(root, legacy),
                "spot_npz": _spot_npz(legacy),
            }
            inventoried = _inventory(legacy)
            journal.update(inventoried)
            if not execute:
                return {
                    "status": "dry_run",
                    "counts": journal["counts"],
                    "symlinks": sum(1 for item in journal["entries"] if item["kind"] == "symlink"),
                    "blockers": blockers,
                    "spot_npz": [item["rel"] for item in journal["spot_npz"]],
                    "reverse_links": len(journal["reverse_links"]),
                }
            _write_json(journal_path, journal)
        else:
            raise MigrationError(f"missing sampling_seeds at {legacy}")
        created = {"alias": False, "shared": False, "manifest": False}
        try:
            if fail_after == "inventoried":
                raise MigrationError("injected failure after inventory")
            if legacy.is_dir() and not legacy.is_symlink():
                dataset.mkdir(parents=True, exist_ok=True)
                os.rename(legacy, dest)
                journal["status"] = "renamed"
                _write_json(journal_path, journal)
            if fail_after == "renamed":
                raise MigrationError("injected failure after rename")
            _apply_links(dest, journal["entries"])
            journal["status"] = "links_rewritten"
            _write_json(journal_path, journal)
            if fail_after == "links_rewritten":
                raise MigrationError("injected failure after link rewrite")
            if not legacy.exists() and not legacy.is_symlink():
                legacy.symlink_to("datasets/fixed_uniform/sampling_seeds")
                created["alias"] = True
            shared = dataset / "shared"
            if not shared.exists() and not shared.is_symlink():
                shared.symlink_to("../../shared")
                created["shared"] = True
            manifest_path = dataset / "dataset_manifest.json"
            if not manifest_path.exists():
                _write_json(manifest_path, _manifest(root, "sampling_seeds"))
                created["manifest"] = True
            journal["status"] = "alias_ready"
            _write_json(journal_path, journal)
            if fail_after == "alias_ready":
                raise MigrationError("injected failure after alias")
            report = _verify(root, journal)
            journal["status"] = "verified"
            journal["verification"] = report
            _write_json(journal_path, journal)
            report["status"] = "migrated"
            return report
        except Exception:
            _rollback(root, journal, created)
            journal["status"] = "rolled_back"
            journal["error"] = traceback.format_exc(limit=6)
            try:
                _write_json(journal_path, journal)
            except OSError:
                pass
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate sampling_seeds into datasets/fixed_uniform")
    parser.add_argument("--training-root", type=Path, default=data_root() / "xuancheng26/training_data_3s")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fail-after", choices=("inventoried", "renamed", "links_rewritten", "alias_ready"))
    args = parser.parse_args(argv)
    root = args.training_root.expanduser().resolve()
    if args.fail_after and not str(root).startswith("/tmp"):
        print("fail-after is only allowed for /tmp fixtures", file=sys.stderr)
        return 2
    try:
        report = migrate(root, execute=args.execute, fail_after=args.fail_after)
    except BlockingIOError:
        print("migration lock is held", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
