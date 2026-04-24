"""Data cleaning and filtering utilities.

Two layers of cleaning live here:

- :func:`split_by_transaction` — pre-DataFrame file splitter: takes one
  OCPP JSON dump and writes per-transaction files.
- :func:`clean_sessions` — DataFrame filter that drops rows whose
  duration, energy, or completeness fall outside physically plausible
  bounds for an AC Level-2 charger. Defaults match 7 kW single-phase
  operation (I_cap = 31.2 A @ 220 V).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

__all__ = [
    "split_by_transaction",
    "SessionCleanConfig",
    "CleanResult",
    "clean_sessions",
]


def split_by_transaction(input_path: str | Path, output_dir: str | Path | None = None) -> dict:
    """MeterValues JSON을 transactionId + connectorId 기준으로 분류하여 개별 파일로 저장.

    파일명 형식: T{transactionId}C{connectorId}-{YYYYMMDD}.json
    날짜는 해당 트랜잭션의 첫 번째 메시지 timestamp 기준.

    Returns:
        dict with keys 'saved' (list of saved file paths) and 'skipped' (count of
        messages without transactionId).
    """
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent.parent / "processed"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)

    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    skipped = 0

    for record in records:
        payload = record.get("meta", {}).get("payload", {})
        txn_id = payload.get("transactionId")
        conn_id = payload.get("connectorId")

        if txn_id is None:
            skipped += 1
            continue

        groups[(txn_id, conn_id or 0)].append(record)

    saved = []

    def _sort_key(item: tuple) -> tuple:
        return (str(item[0][0]), str(item[0][1]))

    for (txn_id, conn_id), messages in sorted(groups.items(), key=_sort_key):
        messages.sort(key=lambda r: r.get("timestamp", {}).get("$date", ""))

        first_ts = messages[0].get("timestamp", {}).get("$date", "")
        try:
            date_str = datetime.fromisoformat(
                first_ts.replace("Z", "+00:00")
            ).strftime("%Y%m%d")
        except (ValueError, AttributeError):
            date_str = "unknown"

        filename = f"T{txn_id}C{conn_id}-{date_str}.json"
        out_path = output_dir / filename

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

        saved.append(str(out_path))

    return {"saved": saved, "skipped": skipped}


@dataclass(frozen=True)
class SessionCleanConfig:
    """Thresholds for :func:`clean_sessions`.

    A session is rejected when any rule fires. The default values target
    a 7 kW single-phase AC charger (I_cap = 31.2 A, V ≈ 220 V →
    theoretical peak ≈ 6.86 kWh/h).

    ``blacklist_charger_ids`` drops sessions by ``charger_id`` regardless
    of any other metric. The default includes ``003DJKCRUN003``, the
    single charger whose only session had ``mean_current_a = 0``
    (verified by the 2026-04-23 P1 audit, see
    ``reports/p1_anomaly_audit.md``).
    """

    min_duration_min: float = 0.5
    max_duration_hours: float = 48.0
    require_positive_energy: bool = True
    drop_orphan_stop_reason: bool = True
    max_energy_rate_kwh_per_hour: float = 7.5  # 6.86 theoretical + 10% margin
    require_positive_duration: bool = True
    allowed_stop_reasons: tuple[str, ...] | None = None
    blacklist_charger_ids: tuple[str, ...] = ("003DJKCRUN003",)

    def describe(self) -> list[str]:
        lines = [
            f"min_duration_min          = {self.min_duration_min}",
            f"max_duration_hours        = {self.max_duration_hours}",
            f"require_positive_duration = {self.require_positive_duration}",
            f"require_positive_energy   = {self.require_positive_energy}",
            f"drop_orphan_stop_reason   = {self.drop_orphan_stop_reason}",
            f"max_energy_rate_kwh_h     = {self.max_energy_rate_kwh_per_hour}",
        ]
        if self.allowed_stop_reasons is not None:
            lines.append(f"allowed_stop_reasons      = {list(self.allowed_stop_reasons)}")
        if self.blacklist_charger_ids:
            lines.append(f"blacklist_charger_ids     = {list(self.blacklist_charger_ids)}")
        return lines


@dataclass(frozen=True)
class CleanResult:
    """Outcome of :func:`clean_sessions`."""

    clean: pd.DataFrame
    rejected: pd.DataFrame
    summary: dict[str, int] = field(default_factory=dict)


def _evaluate_rejection(
    row: pd.Series, cfg: SessionCleanConfig
) -> str | None:
    """Return a reason string when the row fails any rule, else ``None``."""
    if cfg.blacklist_charger_ids:
        charger_id = row.get("charger_id")
        if charger_id in cfg.blacklist_charger_ids:
            return "charger_blacklisted"

    duration = row.get("duration_min")
    if cfg.require_positive_duration and (pd.isna(duration) or duration <= 0):
        return "non_positive_duration"
    if pd.notna(duration) and duration < cfg.min_duration_min:
        return "duration_below_min"
    if pd.notna(duration) and duration > cfg.max_duration_hours * 60:
        return "duration_above_max"

    energy = row.get("energy_delivered_wh")
    if cfg.require_positive_energy and (pd.isna(energy) or energy < 0):
        return "non_positive_energy"
    if pd.notna(energy) and pd.notna(duration) and duration > 0:
        rate_kwh_h = (energy / 1000.0) / (duration / 60.0)
        if rate_kwh_h > cfg.max_energy_rate_kwh_per_hour:
            return "energy_rate_above_physical_max"

    stop_reason = row.get("stop_reason")
    if cfg.drop_orphan_stop_reason and (pd.isna(stop_reason) or stop_reason in ("", None)):
        return "orphan_no_stop_reason"
    if cfg.allowed_stop_reasons is not None:
        if pd.isna(stop_reason) or stop_reason not in cfg.allowed_stop_reasons:
            return "stop_reason_not_allowed"

    return None


def clean_sessions(
    sessions: pd.DataFrame, config: SessionCleanConfig | None = None
) -> CleanResult:
    """Partition ``sessions`` into kept vs rejected rows with an audit trail.

    Each rejected row carries a ``rejection_reason`` string. The returned
    :class:`CleanResult` holds both frames plus a summary count by reason.
    """
    cfg = config or SessionCleanConfig()
    if sessions.empty:
        return CleanResult(
            clean=sessions.copy(),
            rejected=sessions.copy().assign(rejection_reason=pd.Series(dtype="object")),
            summary={},
        )

    reasons = sessions.apply(lambda r: _evaluate_rejection(r, cfg), axis=1)
    keep_mask = reasons.isna()
    clean = sessions[keep_mask].copy()
    rejected = sessions[~keep_mask].copy()
    rejected["rejection_reason"] = reasons[~keep_mask].values

    summary: dict[str, int] = {
        "input": int(len(sessions)),
        "kept": int(len(clean)),
        "rejected": int(len(rejected)),
    }
    if not rejected.empty:
        for reason, n in rejected["rejection_reason"].value_counts().items():
            summary[f"reject:{reason}"] = int(n)

    return CleanResult(clean=clean, rejected=rejected, summary=summary)
