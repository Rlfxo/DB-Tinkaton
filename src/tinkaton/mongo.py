"""MongoDB connection + query helpers for Phase B export scripts.

Configuration lives in ``configs/db_config.yaml`` (git-ignored). Password
is prompted interactively; never stored on disk.
"""

from __future__ import annotations

import getpass
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

import yaml
from pymongo import MongoClient
from pymongo.collection import Collection

__all__ = [
    "MongoConfig",
    "load_mongo_config",
    "connect",
    "AC_MODEL_PATTERNS",
    "is_ac_model",
    "list_ac_chargers",
    "ACTIONS",
]

ACTIONS = ("StartTransaction", "StopTransaction", "MeterValues")

# AC prefix rules confirmed by domain knowledge (2026-04-21):
#   - ``ELA{kW}*``   : EVAR 단상 AC 시리즈 A/B/C
#   - ``E01AS*``     : EVAR AC Slow 계열 (시리즈 D/E, 지역 변종 포함)
# Everything else (``E01DS``, ``E01DM``, ``FC-SS``, ``WEV-D`` ...) is
# treated as non-AC and excluded.
AC_MODEL_PATTERNS = (re.compile(r"^ELA"), re.compile(r"^E01AS"))


@dataclass(frozen=True)
class MongoConfig:
    uri_template: str
    username: str
    database: str
    collection: str

    @classmethod
    def from_dict(cls, data: dict) -> MongoConfig:
        mongo = data["mongo"]
        return cls(
            uri_template=mongo["uri_template"],
            username=mongo["username"],
            database=mongo["database"],
            collection=mongo["collection"],
        )


def load_mongo_config(path: str | Path) -> MongoConfig:
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Copy configs/db_config.example.yaml and fill in your values."
        )
    return MongoConfig.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def connect(cfg: MongoConfig, *, timeout_ms: int = 10_000) -> MongoClient:
    password = os.environ.get("MONGO_PASSWORD") or ""
    if not password:
        password = getpass.getpass(f"MongoDB password for {cfg.username}: ")
    if not password:
        raise SystemExit(
            "Empty password. Type it at the prompt, or export MONGO_PASSWORD=... "
            "in your shell before running this script."
        )
    uri = cfg.uri_template.format(
        username=quote_plus(cfg.username),
        password=quote_plus(password),
    )
    client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
    client.admin.command("ping")
    return client


def is_ac_model(model: str | None) -> bool:
    if not model:
        return False
    return any(p.match(model) for p in AC_MODEL_PATTERNS)


def list_ac_chargers(
    coll: Collection,
    *,
    since: str | None = None,
    until: str | None = None,
    allow_disk_use: bool = True,
    max_time_ms: int = 1_200_000,
) -> list[dict]:
    """Return ``[{id, vendor, model, models}, ...]`` for AC chargers.

    Uses BootNotification records to resolve ``(vendor, model)`` per
    chargerId, then filters by :func:`is_ac_model`. When the same
    chargerId reported multiple models (firmware updates), the first
    non-empty model drives classification and all observed models are
    returned in ``models`` for auditing.
    """
    match: dict = {"meta.action": "BootNotification"}
    if since or until:
        ts: dict = {}
        if since:
            ts["$gte"] = {"$date": since}
        if until:
            ts["$lt"] = {"$date": until}
        match["timestamp"] = ts

    pipeline = [
        {"$match": match},
        {
            "$group": {
                "_id": "$meta.chargerId",
                "vendors": {"$addToSet": "$meta.payload.chargePointVendor"},
                "models": {"$addToSet": "$meta.payload.chargePointModel"},
            }
        },
    ]
    cursor = coll.aggregate(
        pipeline,
        allowDiskUse=allow_disk_use,
        maxTimeMS=max_time_ms,
    )

    chargers: list[dict] = []
    for doc in cursor:
        models = [m for m in (doc.get("models") or []) if m]
        vendors = [v for v in (doc.get("vendors") or []) if v]
        if not any(is_ac_model(m) for m in models):
            continue
        chargers.append(
            {
                "id": doc["_id"],
                "vendor": vendors[0] if vendors else None,
                "model": next((m for m in models if is_ac_model(m)), models[0]),
                "models": models,
            }
        )
    chargers.sort(key=lambda c: c["id"] or "")
    return chargers


def build_action_query(charger_id: str, actions: Iterable[str] = ACTIONS) -> dict:
    return {
        "meta.chargerId": charger_id,
        "meta.action": {"$in": list(actions)},
    }
