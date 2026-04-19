"""Data loaders for OCPP server logs from files and MongoDB.

Provides three DataFrame shapes tuned for different analysis styles:

- :func:`logs_to_dataframe` — one row per OCPP message (all actions).
- :func:`meter_values_to_dataframe` — wide form, one row per sampled
  timestamp, measurands as columns (ML/DL feature matrix).
- :func:`meter_values_to_long_dataframe` — long form, one row per
  ``sampledValue`` (aggregation- and per-measurand analysis).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "load_ocpp_logs",
    "logs_to_dataframe",
    "meter_values_to_dataframe",
    "meter_values_to_long_dataframe",
]

_MEASURAND_MAP: dict[str, tuple[str, str]] = {
    "SoC": ("soc", "pct"),
    "Energy.Active.Import.Register": ("energy", "wh"),
    "Energy.Active.Import.Interval": ("energy_interval", "wh"),
    "Energy.Active.Export.Register": ("energy_export", "wh"),
    "Power.Active.Import": ("power", "w"),
    "Power.Offered": ("power_offered", "w"),
    "Voltage": ("voltage", "v"),
    "Current.Import": ("current", "a"),
    "Current.Offered": ("current_offered", "a"),
    "Current.Export": ("current_export", "a"),
    "Frequency": ("frequency", "hz"),
    "Temperature": ("temperature", "c"),
    "RPM": ("rpm", ""),
}


def load_ocpp_logs(path: str | Path) -> list[dict[str, Any]]:
    """Load an OCPP log JSON file, auto-repairing truncated arrays.

    A common export artifact is a missing closing ``]``; this loader
    appends one when the top level starts with ``[`` but does not end
    with ``]``.
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("[") and not text.endswith("]"):
        text = text.rstrip(",") + "]"
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array at top level, got {type(data).__name__}")
    return data


def _server_timestamp(log: dict) -> pd.Timestamp | None:
    ts = log.get("timestamp")
    if isinstance(ts, dict) and "$date" in ts:
        return pd.Timestamp(ts["$date"])
    if ts:
        return pd.Timestamp(ts)
    return None


def _coerce_float(val: Any) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _column_name(measurand: str, phase: str | None, location: str | None) -> str:
    if measurand in _MEASURAND_MAP:
        name, unit = _MEASURAND_MAP[measurand]
    else:
        name = measurand.lower().replace(".", "_")
        unit = ""
    parts: list[str] = [name]
    if phase:
        parts.append(phase.lower().replace("-", ""))
    if location:
        parts.append(location.lower())
    if unit:
        parts.append(unit)
    return "_".join(parts)


def logs_to_dataframe(logs: list[dict]) -> pd.DataFrame:
    """Flatten any OCPP log stream to one row per message.

    Payload is preserved as a dict column so downstream code can opt into
    action-specific parsing.
    """
    rows = []
    for log in logs:
        meta = log.get("meta", {}) or {}
        payload = meta.get("payload") or {}
        rows.append(
            {
                "server_timestamp": _server_timestamp(log),
                "level": log.get("level"),
                "message": log.get("message"),
                "charger_id": meta.get("chargerId"),
                "session_id": (meta.get("sessionInfo") or {}).get("sessionId"),
                "message_id": meta.get("messageId"),
                "message_type": meta.get("messageType"),
                "server_recv_type": meta.get("serverRecvType"),
                "action": meta.get("action"),
                "connector_id": payload.get("connectorId"),
                "transaction_id": payload.get("transactionId"),
                "payload": payload or None,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["server_timestamp"] = pd.to_datetime(df["server_timestamp"], utc=True)
    return df.sort_values("server_timestamp").reset_index(drop=True)


def meter_values_to_long_dataframe(logs: list[dict]) -> pd.DataFrame:
    """Flatten MeterValues to long form — one row per ``sampledValue``.

    Columns: ``timestamp``, ``server_timestamp``, ``charger_id``,
    ``session_id``, ``message_id``, ``connector_id``, ``transaction_id``,
    ``measurand``, ``value``, ``raw_value``, ``unit``, ``phase``,
    ``location``, ``context``.
    """
    rows = []
    for log in logs:
        meta = log.get("meta", {}) or {}
        if meta.get("action") != "MeterValues":
            continue
        payload = meta.get("payload") or {}
        base = {
            "server_timestamp": _server_timestamp(log),
            "charger_id": meta.get("chargerId"),
            "session_id": (meta.get("sessionInfo") or {}).get("sessionId"),
            "message_id": meta.get("messageId"),
            "connector_id": payload.get("connectorId"),
            "transaction_id": payload.get("transactionId"),
        }
        for mv in payload.get("meterValue") or []:
            sample_ts = pd.Timestamp(mv.get("timestamp")) if mv.get("timestamp") else pd.NaT
            for sv in mv.get("sampledValue") or []:
                rows.append(
                    {
                        **base,
                        "timestamp": sample_ts,
                        "measurand": sv.get("measurand"),
                        "value": _coerce_float(sv.get("value")),
                        "raw_value": sv.get("value"),
                        "unit": sv.get("unit"),
                        "phase": sv.get("phase"),
                        "location": sv.get("location"),
                        "context": sv.get("context"),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["server_timestamp"] = pd.to_datetime(df["server_timestamp"], utc=True)
    return df.sort_values(["timestamp", "measurand"]).reset_index(drop=True)


def meter_values_to_dataframe(logs: list[dict]) -> pd.DataFrame:
    """Flatten MeterValues to wide form — one row per sample timestamp.

    Each measurand becomes its own column (e.g. ``soc_pct``, ``voltage_v``,
    ``current_a``, ``energy_wh``, ``temperature_body_c``). Multi-phase
    measurements get a phase suffix (``voltage_l1_v``); multi-location
    measurements get a location suffix. Unknown measurands fall back to
    a snake-cased column name without a unit suffix.

    The result is an ML/DL-ready feature matrix: sorted by sample time,
    UTC-aware timestamps, numeric columns coerced to float.
    """
    records: list[dict] = []
    for log in logs:
        meta = log.get("meta", {}) or {}
        if meta.get("action") != "MeterValues":
            continue
        payload = meta.get("payload") or {}
        base = {
            "charger_id": meta.get("chargerId"),
            "session_id": (meta.get("sessionInfo") or {}).get("sessionId"),
            "message_id": meta.get("messageId"),
            "connector_id": payload.get("connectorId"),
            "transaction_id": payload.get("transactionId"),
            "server_timestamp": _server_timestamp(log),
        }
        for mv in payload.get("meterValue") or []:
            row = dict(base)
            row["timestamp"] = pd.Timestamp(mv.get("timestamp")) if mv.get("timestamp") else pd.NaT
            for sv in mv.get("sampledValue") or []:
                measurand = sv.get("measurand") or "unknown"
                col = _column_name(measurand, sv.get("phase"), sv.get("location"))
                row[col] = _coerce_float(sv.get("value"))
            records.append(row)
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["server_timestamp"] = pd.to_datetime(df["server_timestamp"], utc=True)
    id_cols = [
        "timestamp",
        "server_timestamp",
        "charger_id",
        "session_id",
        "connector_id",
        "transaction_id",
        "message_id",
    ]
    measurement_cols = sorted(c for c in df.columns if c not in id_cols)
    return (
        df.reindex(columns=id_cols + measurement_cols)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
