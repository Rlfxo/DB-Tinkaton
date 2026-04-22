"""Full Phase B export — every AC charger in the PLATFORM-OCPP collection.

Workflow:
1. Classify chargers via BootNotification (AC = ELA* or E01AS*).
2. Save the classification to ``{out_dir}/_charger_manifest.csv`` for audit.
3. Stream the three OCPP actions (StartTransaction / StopTransaction /
   MeterValues) per AC charger into ``{out_dir}/{chargerId}.json``.
4. Track progress in ``{out_dir}/_checkpoint.json`` so an interrupted run
   can resume without redoing completed chargers.
5. Record any per-charger failures to ``{out_dir}/_errors.log`` — the
   loop continues so one bad charger does not abort the batch.

CLI flags:
    --classify-only   write manifest, skip data export
    --dry-run         count docs per AC charger, skip data export
    --resume          skip chargers listed in the checkpoint
    --limit N         process at most N AC chargers (testing)
    --output-dir DIR  default ``data/raw/phase_b``
    --since ISO       only classify chargers whose BootNotification
                      timestamps fall within the window (default: full
                      collection range)
    --until ISO       upper bound (exclusive)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from bson.json_util import RELAXED_JSON_OPTIONS, dumps

from tinkaton.mongo import (
    build_action_query,
    connect,
    list_ac_chargers,
    load_mongo_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "db_config.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="data/raw/phase_b")
    p.add_argument("--classify-only", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="count per charger, no data fetched")
    p.add_argument("--resume", action="store_true", help="skip chargers in the checkpoint")
    p.add_argument("--limit", type=int, default=None, help="process at most N AC chargers")
    p.add_argument("--since", default=None, help="ISO timestamp lower bound for classification")
    p.add_argument("--until", default=None, help="ISO timestamp upper bound for classification")
    p.add_argument("--batch-size", type=int, default=2000)
    return p.parse_args()


def write_manifest(out_dir: Path, chargers: list[dict]) -> Path:
    path = out_dir / "_charger_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["charger_id", "vendor", "model", "all_models"])
        for c in chargers:
            writer.writerow(
                [
                    c["id"],
                    c["vendor"] or "",
                    c["model"] or "",
                    "|".join(c.get("models") or []),
                ]
            )
    return path


def load_checkpoint(out_dir: Path) -> set[str]:
    path = out_dir / "_checkpoint.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("completed") or [])


def append_checkpoint(out_dir: Path, charger_id: str) -> None:
    path = out_dir / "_checkpoint.json"
    done = load_checkpoint(out_dir)
    done.add(charger_id)
    path.write_text(
        json.dumps({"completed": sorted(done)}, indent=2),
        encoding="utf-8",
    )


def log_error(out_dir: Path, charger_id: str, exc: Exception) -> None:
    path = out_dir / "_errors.log"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"--- {datetime.utcnow().isoformat()}Z  {charger_id} ---\n")
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        f.write("\n")


def export_one_charger(
    coll, charger_id: str, out_dir: Path, batch_size: int
) -> tuple[int, float]:
    """Fetch and write one charger's 3-action stream. Returns (docs, mb)."""
    query = build_action_query(charger_id)
    cursor = coll.find(query).sort("timestamp", 1).batch_size(batch_size)
    out_path = out_dir / f"{charger_id}.json"
    docs: list[dict] = []
    for doc in cursor:
        docs.append(doc)
    if not docs:
        return 0, 0.0
    with out_path.open("w", encoding="utf-8") as f:
        f.write(dumps(docs, json_options=RELAXED_JSON_OPTIONS))
    mb = out_path.stat().st_size / 1024 / 1024
    return len(docs), mb


def main() -> int:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_mongo_config(CONFIG_PATH)
    client = connect(cfg)
    coll = client[cfg.database][cfg.collection]

    print("classifying AC chargers via BootNotification ...", flush=True)
    t0 = time.monotonic()
    chargers = list_ac_chargers(coll, since=args.since, until=args.until)
    t1 = time.monotonic()
    print(f"  found {len(chargers)} AC chargers  ({t1 - t0:.1f}s)")

    manifest_path = write_manifest(out_dir, chargers)
    print(f"  manifest → {manifest_path.relative_to(ROOT)}")

    if args.classify_only:
        return 0

    done = load_checkpoint(out_dir) if args.resume else set()
    if done:
        print(f"  resume: skipping {len(done)} already-completed chargers")

    todo = [c for c in chargers if c["id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"  to process: {len(todo)}")
    if not todo:
        return 0

    cumulative_docs = 0
    cumulative_mb = 0.0
    start = time.monotonic()

    for i, charger in enumerate(todo, start=1):
        cid = charger["id"]
        q = build_action_query(cid)
        try:
            n = coll.count_documents(q)
        except Exception as exc:
            log_error(out_dir, cid, exc)
            print(f"[{i:>3}/{len(todo)}] {cid}: count_documents FAILED — logged")
            continue

        if args.dry_run:
            print(f"[{i:>3}/{len(todo)}] {cid}: {n:,} docs (dry-run)")
            cumulative_docs += n
            continue

        if n == 0:
            print(f"[{i:>3}/{len(todo)}] {cid}: 0 docs — skipping")
            append_checkpoint(out_dir, cid)
            continue

        try:
            t_start = time.monotonic()
            wrote, mb = export_one_charger(coll, cid, out_dir, args.batch_size)
            elapsed = time.monotonic() - t_start
            append_checkpoint(out_dir, cid)
            cumulative_docs += wrote
            cumulative_mb += mb
            eta = "unknown"
            if i > 0:
                avg = (time.monotonic() - start) / i
                remaining = avg * (len(todo) - i)
                eta = f"{remaining / 60:.1f} min"
            print(
                f"[{i:>3}/{len(todo)}] {cid}: {wrote:,} docs  {mb:.2f} MB  "
                f"({elapsed:.1f}s)  cum {cumulative_mb:.1f} MB  ETA {eta}"
            )
        except Exception as exc:
            log_error(out_dir, cid, exc)
            print(f"[{i:>3}/{len(todo)}] {cid}: export FAILED — logged, continuing")
            continue

    total = time.monotonic() - start
    print()
    print(f"done. {cumulative_docs:,} docs  {cumulative_mb:.1f} MB  ({total / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
