"""Export 3-action OCPP logs for a single charger to a JSON file.

Purpose: validate the Phase B export path on one charger before scaling
to all 492 AC chargers.

Actions exported: ``StartTransaction``, ``StopTransaction``,
``MeterValues``. The output uses MongoDB extended JSON in relaxed mode
so that ``tinkaton.loader.load_ocpp_logs`` consumes it unchanged.

Usage:

    uv run python scripts/export_single_charger.py \
        --charger <chargerId> \
        --output-dir data/raw/phase_b/test

    # Dry-run: just count matching docs, no export.
    uv run python scripts/export_single_charger.py \
        --charger <chargerId> --dry-run
"""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path
from urllib.parse import quote_plus

import yaml
from bson.json_util import RELAXED_JSON_OPTIONS, dumps
from pymongo import MongoClient

ACTIONS = ("StartTransaction", "StopTransaction", "MeterValues")

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "db_config.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"Missing {CONFIG_PATH}. Copy configs/db_config.example.yaml and fill in your values."
        )
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def connect(cfg: dict) -> MongoClient:
    mongo = cfg["mongo"]
    username = mongo["username"]
    password = getpass.getpass(f"MongoDB password for {username}: ")
    uri = mongo["uri_template"].format(
        username=quote_plus(username),
        password=quote_plus(password),
    )
    client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    # Force connection to surface auth errors before the real query
    client.admin.command("ping")
    return client


def build_query(charger_id: str) -> dict:
    return {
        "meta.chargerId": charger_id,
        "meta.action": {"$in": list(ACTIONS)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--charger", required=True, help="chargerId to export")
    parser.add_argument(
        "--output-dir",
        default="data/raw/phase_b/test",
        help="directory for the output JSON file (relative to repo root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count matching docs only; skip the export",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="cursor batch size for the fetch loop",
    )
    args = parser.parse_args()

    cfg = load_config()
    client = connect(cfg)
    coll = client[cfg["mongo"]["database"]][cfg["mongo"]["collection"]]

    query = build_query(args.charger)
    n_total = coll.count_documents(query)
    print(f"charger={args.charger}  matching docs: {n_total:,}")

    if args.dry_run or n_total == 0:
        return

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.charger}.json"

    cursor = (
        coll.find(query, no_cursor_timeout=False)
        .sort("timestamp", 1)
        .batch_size(args.batch_size)
    )

    docs: list[dict] = []
    for i, doc in enumerate(cursor, start=1):
        docs.append(doc)
        if i % 5_000 == 0:
            print(f"  fetched {i:,}/{n_total:,}")

    print(f"writing {len(docs):,} docs → {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        f.write(dumps(docs, json_options=RELAXED_JSON_OPTIONS, indent=2))
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"done. file size: {size_mb:,.2f} MB")


if __name__ == "__main__":
    main()
