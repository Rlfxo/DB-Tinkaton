"""Session-level aggregation and feature engineering for OCPP MeterValue data.

Pipeline role: consume the wide DataFrame produced by
:func:`tinkaton.loader.meter_values_to_dataframe` and emit session-level
rows suitable for Phase B EDA, ML training (departure-time prediction),
and LP simulation inputs.

The canonical output schema is defined in HANDOFF_ModelPipeline v2 §5.2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "normalize_measurement_columns",
    "aggregate_sessions",
    "compute_capacity_bound_flag",
    "extract_initial_profile",
    "SessionAggregateConfig",
    "estimate_icap_per_charger",
    "derive_station_id",
    "build_station_clusters",
]


_CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "current_a": ("current_a", "current_outlet_a", "current_import_a"),
    "voltage_v": ("voltage_v", "voltage_outlet_v", "voltage_l1_v"),
    "power_w": ("power_w", "power_outlet_w", "power_active_import_w"),
    "energy_wh": ("energy_wh", "energy_active_import_register_wh"),
    "soc_pct": ("soc_pct", "soc_ev_pct"),
    "temperature_c": ("temperature_c", "temperature_body_c", "temperature_outlet_c"),
    "current_offered_a": ("current_offered_a",),
}


def normalize_measurement_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map charger-specific column names onto a canonical set.

    The OCPP wide schema varies by vendor (e.g. ``current_outlet_a`` vs
    ``current_a``; ``soc_ev_pct`` vs ``soc_pct``). Downstream code depends
    on a canonical set, so we coalesce aliases into the first canonical
    name that has a non-null value, preferring the canonical spelling
    when present. Original columns are retained.
    """
    if df.empty:
        return df
    out = df.copy()
    for canonical, aliases in _CANONICAL_ALIASES.items():
        present = [c for c in aliases if c in out.columns]
        if not present:
            continue
        if canonical in present:
            source = out[canonical]
            for alias in present:
                if alias != canonical:
                    source = source.combine_first(out[alias])
            out[canonical] = source
        else:
            series = out[present[0]]
            for alias in present[1:]:
                series = series.combine_first(out[alias])
            out[canonical] = series
    return out


@dataclass(frozen=True)
class SessionAggregateConfig:
    """Tunable thresholds for session-level aggregation.

    ``i_pwm_assumed_a`` is the PWM cap used when ``current_offered_a`` is
    absent from the data. Phase A sessions have a known cap (31.2 A for
    L7, 18 A for L4, 9 A for L2); Phase B defaults to the 7 kW field cap.
    """

    capacity_bound_ratio: float = 0.95
    capacity_bound_min_minutes: float = 20.0
    initial_profile_window_min: float = 10.0
    min_session_duration_min: float = 0.0
    i_pwm_assumed_a: float = 31.2


def _session_key(row: pd.Series) -> tuple:
    return (row["charger_id"], row.get("transaction_id"), row.get("session_id"))


def _coerce_duration_seconds(ts: pd.Series) -> float:
    if len(ts) < 2:
        return 0.0
    span = ts.max() - ts.min()
    if isinstance(span, pd.Timedelta):
        return span.total_seconds()
    return float(span)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.ptp(x) == 0:
        return 0.0
    m, _ = np.polyfit(x, y, 1)
    return float(m)


def extract_initial_profile(
    session_df: pd.DataFrame,
    window_min: float = 10.0,
    current_col: str = "current_a",
) -> dict[str, float | int]:
    """Summarize the first ``window_min`` minutes of a session's current.

    Returns ``initial_mean_a``, ``initial_std_a``, ``initial_slope_a_per_min``,
    and ``initial_n_samples``. Missing current values yield NaN means but
    preserve the sample count.
    """
    if session_df.empty or current_col not in session_df.columns:
        return {
            "initial_mean_a": float("nan"),
            "initial_std_a": float("nan"),
            "initial_slope_a_per_min": float("nan"),
            "initial_n_samples": 0,
        }
    df = session_df.sort_values("timestamp")
    start = df["timestamp"].iloc[0]
    window_end = start + pd.Timedelta(minutes=window_min)
    window = df[df["timestamp"] <= window_end]
    series = window[current_col].dropna()
    n = int(len(series))
    if n == 0:
        return {
            "initial_mean_a": float("nan"),
            "initial_std_a": float("nan"),
            "initial_slope_a_per_min": float("nan"),
            "initial_n_samples": 0,
        }
    minutes = (window["timestamp"] - start).dt.total_seconds().to_numpy() / 60.0
    values = window[current_col].to_numpy(dtype=float)
    mask = ~np.isnan(values)
    return {
        "initial_mean_a": float(np.nanmean(values)),
        "initial_std_a": float(np.nanstd(values, ddof=0)),
        "initial_slope_a_per_min": _slope(minutes[mask], values[mask]),
        "initial_n_samples": n,
    }


def compute_capacity_bound_flag(
    session_df: pd.DataFrame,
    ratio_threshold: float = 0.95,
    min_minutes: float = 20.0,
    i_pwm_assumed_a: float = 31.2,
) -> tuple[bool, float]:
    """Return ``(flag, longest_bound_window_minutes)``.

    The session is flagged when the binding ratio
    ``I_actual / I_PWM`` meets or exceeds ``ratio_threshold`` for a
    contiguous window of at least ``min_minutes``. When
    ``current_offered_a`` is missing we fall back to ``i_pwm_assumed_a``.
    """
    if session_df.empty or "current_a" not in session_df.columns:
        return False, 0.0
    df = session_df.sort_values("timestamp").reset_index(drop=True)
    if "current_offered_a" in df.columns and df["current_offered_a"].notna().any():
        i_cap = df["current_offered_a"].ffill().bfill().to_numpy(dtype=float)
    else:
        i_cap = np.full(len(df), float(i_pwm_assumed_a))
    i_act = df["current_a"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = np.where(i_cap > 0, i_act / i_cap, np.nan)
    bound = br >= ratio_threshold
    times = df["timestamp"].to_numpy()
    longest_s = 0.0
    run_start: pd.Timestamp | None = None
    for is_bound, t in zip(bound, times, strict=True):
        if is_bound and run_start is None:
            run_start = t
        elif not is_bound and run_start is not None:
            span = (t - run_start) / np.timedelta64(1, "s")
            longest_s = max(longest_s, float(span))
            run_start = None
    if run_start is not None:
        span = (times[-1] - run_start) / np.timedelta64(1, "s")
        longest_s = max(longest_s, float(span))
    longest_min = longest_s / 60.0
    return bool(longest_min >= min_minutes), longest_min


def _energy_delivered_wh(energy: pd.Series) -> float | None:
    series = energy.dropna()
    if series.empty:
        return None
    # OCPP Energy.Active.Import.Register is a lifetime accumulator; session
    # delivery is the end-minus-start span. If the source is non-monotonic
    # (rare vendor quirk or fragment with mixed streams) we fall back to
    # summing positive deltas, which is a more defensible estimate than
    # raw max-minus-min when the readings reset.
    diffs = series.diff().dropna()
    if (diffs < -1.0).any():
        return float(diffs.clip(lower=0).sum())
    return float(series.iloc[-1] - series.iloc[0])


def _circular_encode(value: float, period: int) -> tuple[float, float]:
    angle = 2 * np.pi * value / period
    return float(np.sin(angle)), float(np.cos(angle))


def aggregate_sessions(
    wide_df: pd.DataFrame,
    config: SessionAggregateConfig | None = None,
    charger_metadata: pd.DataFrame | None = None,
    transaction_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate a wide MeterValue DataFrame into session-level rows.

    Sessions are keyed by ``(charger_id, transaction_id)``. Sessions
    missing a ``transaction_id`` fall back to ``session_id``. Rows with
    neither are dropped (they cannot be attributed).

    When ``transaction_events`` is supplied (from
    :func:`tinkaton.loader.transaction_events_to_dataframe`), its
    ``start_ts`` and ``stop_ts`` override the MeterValue-derived
    ``arrival_ts`` and ``plug_out_ts``; its ``meter_start_wh`` /
    ``meter_stop_wh`` override ``energy_delivered_wh``. Sessions that
    appear in ``transaction_events`` but have no MeterValues are kept as
    label-only rows (current/SoC/capacity-bound columns become NaN).

    The ``charger_metadata`` argument, when provided, must have columns
    ``charger_id`` plus any of ``floor`` and ``charger_type``; values are
    merged onto the output.
    """
    cfg = config or SessionAggregateConfig()
    if wide_df.empty and (transaction_events is None or transaction_events.empty):
        return pd.DataFrame()
    if wide_df.empty:
        wide_df = pd.DataFrame(columns=["timestamp", "charger_id", "transaction_id", "session_id"])
    df = normalize_measurement_columns(wide_df)
    df = df.copy()
    df["session_key"] = df["transaction_id"].where(
        df["transaction_id"].notna(), df.get("session_id")
    )
    df = df.dropna(subset=["session_key", "charger_id"])
    has_mv = not df.empty

    records: list[dict] = []
    if not has_mv and (transaction_events is None or transaction_events.empty):
        return pd.DataFrame()
    for (charger_id, key), group in df.groupby(["charger_id", "session_key"], sort=False):
        group = group.sort_values("timestamp")
        arrival = group["timestamp"].iloc[0]
        plug_out = group["timestamp"].iloc[-1]
        duration_s = _coerce_duration_seconds(group["timestamp"])
        duration_min = duration_s / 60.0
        if duration_min < cfg.min_session_duration_min:
            continue
        energy_wh = (
            _energy_delivered_wh(group["energy_wh"]) if "energy_wh" in group.columns else None
        )
        flag, bound_min = compute_capacity_bound_flag(
            group,
            ratio_threshold=cfg.capacity_bound_ratio,
            min_minutes=cfg.capacity_bound_min_minutes,
            i_pwm_assumed_a=cfg.i_pwm_assumed_a,
        )
        profile = extract_initial_profile(
            group, window_min=cfg.initial_profile_window_min
        )

        local_arrival = arrival.tz_convert("Asia/Seoul") if arrival.tzinfo else arrival
        hour = int(local_arrival.hour)
        dow = int(local_arrival.dayofweek)
        month = int(local_arrival.month)
        hour_sin, hour_cos = _circular_encode(hour, 24)
        dow_sin, dow_cos = _circular_encode(dow, 7)
        month_sin, month_cos = _circular_encode(month - 1, 12)

        mean_current = (
            float(group["current_a"].mean()) if "current_a" in group.columns else float("nan")
        )
        mean_voltage = (
            float(group["voltage_v"].mean()) if "voltage_v" in group.columns else float("nan")
        )
        start_soc = (
            float(group["soc_pct"].dropna().iloc[0])
            if "soc_pct" in group.columns and group["soc_pct"].notna().any()
            else float("nan")
        )
        end_soc = (
            float(group["soc_pct"].dropna().iloc[-1])
            if "soc_pct" in group.columns and group["soc_pct"].notna().any()
            else float("nan")
        )

        session_col = "session_id" if "session_id" in group.columns else None
        connector_col = "connector_id" if "connector_id" in group.columns else None
        records.append(
            {
                "charger_id": charger_id,
                "session_key": str(key) if key is not None else None,
                "transaction_id": group["transaction_id"].iloc[0],
                "session_id": group[session_col].iloc[0] if session_col else None,
                "connector_id": group[connector_col].iloc[0] if connector_col else None,
                "arrival_ts": arrival,
                "plug_out_ts": plug_out,
                "duration_min": duration_min,
                "n_samples": int(len(group)),
                "energy_delivered_wh": energy_wh,
                "mean_current_a": mean_current,
                "mean_voltage_v": mean_voltage,
                "start_soc_pct": start_soc,
                "end_soc_pct": end_soc,
                "capacity_bound_flag": flag,
                "capacity_bound_duration_min": bound_min,
                "hour": hour,
                "dayofweek": dow,
                "month": month,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "dow_sin": dow_sin,
                "dow_cos": dow_cos,
                "month_sin": month_sin,
                "month_cos": month_cos,
                **profile,
            }
        )

    out = pd.DataFrame(records)
    if transaction_events is not None and not transaction_events.empty:
        out = _apply_transaction_events(out, transaction_events)
    if out.empty:
        return out
    if charger_metadata is not None and not charger_metadata.empty:
        out = out.merge(charger_metadata, on="charger_id", how="left")
    return out.sort_values(["arrival_ts", "charger_id"]).reset_index(drop=True)


def estimate_icap_per_charger(
    sessions: pd.DataFrame,
    *,
    quantile: float = 0.99,
    min_duration_min: float = 60.0,
    allowed_stop_reasons: tuple[str, ...] = ("Other", "Local", "Remote"),
    min_sessions_per_charger: int = 10,
    fallback_quantile: float = 0.95,
) -> pd.DataFrame:
    """Estimate each charger's I_cap from its observed sustained current.

    The premise is field-level heterogeneity: different charger sites run
    different PWM ceilings, so a global I_cap (e.g. 31.2 A for the 7 kW
    field default) does not normalize binding ratios across the fleet.

    Method: for every session with MeterValues that completed normally
    (``stop_reason`` in ``allowed_stop_reasons``) and lasted at least
    ``min_duration_min`` minutes, take ``mean_current_a`` as the session's
    sustained-current proxy. Per charger, I_cap is the ``quantile`` of
    that distribution. Chargers below ``min_sessions_per_charger`` fall
    back to ``fallback_quantile`` (more permissive) so they still get an
    estimate instead of being dropped.

    Returns one row per charger with: ``charger_id``,
    ``i_cap_observed_a``, ``n_sessions_used``, ``median_current_a``,
    ``max_current_a``, ``estimation_quantile``.
    """
    if sessions.empty:
        return pd.DataFrame(
            columns=[
                "charger_id",
                "i_cap_observed_a",
                "n_sessions_used",
                "median_current_a",
                "max_current_a",
                "estimation_quantile",
            ]
        )

    mask = sessions["has_meter_values"].fillna(False).astype(bool)
    mask &= sessions["mean_current_a"].notna()
    mask &= sessions["duration_min"] >= min_duration_min
    mask &= sessions["stop_reason"].isin(allowed_stop_reasons)
    pool = sessions[mask]

    rows: list[dict] = []
    for charger_id, group in pool.groupby("charger_id"):
        currents = group["mean_current_a"].dropna()
        n = int(len(currents))
        if n == 0:
            continue
        used_quantile = quantile if n >= min_sessions_per_charger else fallback_quantile
        rows.append(
            {
                "charger_id": charger_id,
                "i_cap_observed_a": float(currents.quantile(used_quantile)),
                "n_sessions_used": n,
                "median_current_a": float(currents.median()),
                "max_current_a": float(currents.max()),
                "estimation_quantile": used_quantile,
            }
        )

    return pd.DataFrame(rows).sort_values("charger_id").reset_index(drop=True)


_STATION_ID_RE = re.compile(r"^(.*?)(\d{3})$")


def derive_station_id(charger_id: str | None) -> str | None:
    """Return the station prefix implied by a ``chargerId``.

    The operator's naming convention encodes a unit number as the last
    three digits of the identifier (e.g. ``<prefix>006``); the portion
    before those digits is the physical station. Returns ``None`` for a
    missing ID and the input unchanged when no three-digit suffix is
    found.
    """
    if not charger_id or not isinstance(charger_id, str):
        return None
    m = _STATION_ID_RE.match(charger_id)
    return m.group(1) if m else charger_id


def build_station_clusters(
    sessions: pd.DataFrame,
    *,
    manifest: pd.DataFrame | None = None,
    lp_min_chargers: int = 10,
    lp_max_chargers: int = 30,
) -> pd.DataFrame:
    """Aggregate charger-level session counts into station-level rows.

    Each station is keyed by :func:`derive_station_id` applied to
    ``charger_id``. The optional ``manifest`` should map ``charger_id``
    to ``model`` (and any metadata); model values are collapsed into a
    pipe-separated ``models`` string per station.

    The returned DataFrame has columns ``station_id``, ``n_chargers``,
    ``charger_ids`` (pipe-separated), ``models`` (pipe-separated unique
    models), ``n_sessions``, ``median_sessions_per_charger``,
    ``is_lp_candidate`` (True when ``lp_min_chargers ≤ n_chargers ≤
    lp_max_chargers``), sorted by ``n_chargers`` descending.
    """
    if sessions.empty:
        return pd.DataFrame(
            columns=[
                "station_id",
                "n_chargers",
                "charger_ids",
                "models",
                "n_sessions",
                "median_sessions_per_charger",
                "is_lp_candidate",
            ]
        )

    charger_sessions = (
        sessions.groupby("charger_id").size().rename("n_sessions").reset_index()
    )
    charger_sessions["station_id"] = charger_sessions["charger_id"].map(
        derive_station_id
    )
    if manifest is not None and not manifest.empty and "model" in manifest.columns:
        charger_sessions = charger_sessions.merge(
            manifest[["charger_id", "model"]], on="charger_id", how="left"
        )

    def _concat_unique_strings(values: pd.Series) -> str:
        uniq = sorted({str(v) for v in values.dropna().tolist() if v})
        return "|".join(uniq)

    grouped = charger_sessions.groupby("station_id", dropna=True).agg(
        n_chargers=("charger_id", "nunique"),
        charger_ids=("charger_id", _concat_unique_strings),
        n_sessions=("n_sessions", "sum"),
        median_sessions_per_charger=("n_sessions", "median"),
    )
    if "model" in charger_sessions.columns:
        grouped["models"] = charger_sessions.groupby("station_id")["model"].apply(
            _concat_unique_strings
        )
    else:
        grouped["models"] = ""

    grouped = grouped.reset_index()
    grouped["is_lp_candidate"] = (
        (grouped["n_chargers"] >= lp_min_chargers)
        & (grouped["n_chargers"] <= lp_max_chargers)
    )
    return grouped.sort_values(
        ["n_chargers", "n_sessions"], ascending=False
    ).reset_index(drop=True)


def _apply_transaction_events(
    mv_sessions: pd.DataFrame, tx: pd.DataFrame
) -> pd.DataFrame:
    """Merge authoritative StartTx/StopTx fields into MV-derived sessions.

    Strategy: outer merge on ``(charger_id, transaction_id)``. When
    ``start_ts`` / ``stop_ts`` are present, they replace ``arrival_ts`` /
    ``plug_out_ts`` (OCPP events are the ground truth). Energy is
    recomputed from ``meter_stop_wh - meter_start_wh`` when both are
    present. Time features are recomputed from the corrected arrival.
    Rows that come from ``tx`` but not ``mv_sessions`` are kept with
    MeterValue-derived columns left NaN.
    """
    tx_norm = tx.copy()
    tx_norm["transaction_id"] = pd.to_numeric(
        tx_norm["transaction_id"], errors="coerce"
    ).astype("Int64")
    if mv_sessions.empty:
        merged = tx_norm.copy()
        for col in (
            "n_samples",
            "mean_current_a",
            "mean_voltage_v",
            "start_soc_pct",
            "end_soc_pct",
            "capacity_bound_flag",
            "capacity_bound_duration_min",
            "initial_mean_a",
            "initial_std_a",
            "initial_slope_a_per_min",
            "initial_n_samples",
            "session_id",
            "session_key",
            "message_id",
        ):
            merged[col] = pd.NA
    else:
        mv = mv_sessions.copy()
        mv["transaction_id"] = pd.to_numeric(
            mv["transaction_id"], errors="coerce"
        ).astype("Int64")
        merged = mv.merge(
            tx_norm,
            on=["charger_id", "transaction_id"],
            how="outer",
            suffixes=("", "_tx"),
        )
        # Connector id may appear on both sides; prefer MV-side, fall back to tx.
        if "connector_id_tx" in merged.columns:
            merged["connector_id"] = merged["connector_id"].combine_first(
                merged.pop("connector_id_tx")
            )

    # Authoritative timing
    if "arrival_ts" in merged.columns:
        merged["arrival_ts"] = merged["start_ts"].combine_first(merged["arrival_ts"])
    else:
        merged["arrival_ts"] = merged["start_ts"]
    if "plug_out_ts" in merged.columns:
        merged["plug_out_ts"] = merged["stop_ts"].combine_first(merged["plug_out_ts"])
    else:
        merged["plug_out_ts"] = merged["stop_ts"]
    duration_td = merged["plug_out_ts"] - merged["arrival_ts"]
    merged["duration_min"] = duration_td.dt.total_seconds() / 60.0

    # Authoritative energy: meterStop - meterStart (when both present)
    tx_energy = merged["meter_stop_wh"] - merged["meter_start_wh"]
    if "energy_delivered_wh" in merged.columns:
        merged["energy_delivered_wh"] = tx_energy.combine_first(
            merged["energy_delivered_wh"]
        )
    else:
        merged["energy_delivered_wh"] = tx_energy

    # Recompute time features from corrected arrival
    local_arrival = merged["arrival_ts"].dt.tz_convert("Asia/Seoul")
    merged["hour"] = local_arrival.dt.hour.astype("Int64")
    merged["dayofweek"] = local_arrival.dt.dayofweek.astype("Int64")
    merged["month"] = local_arrival.dt.month.astype("Int64")
    hour_rad = 2 * np.pi * merged["hour"].astype("Float64") / 24
    dow_rad = 2 * np.pi * merged["dayofweek"].astype("Float64") / 7
    month_rad = 2 * np.pi * (merged["month"].astype("Float64") - 1) / 12
    merged["hour_sin"] = np.sin(hour_rad)
    merged["hour_cos"] = np.cos(hour_rad)
    merged["dow_sin"] = np.sin(dow_rad)
    merged["dow_cos"] = np.cos(dow_rad)
    merged["month_sin"] = np.sin(month_rad)
    merged["month_cos"] = np.cos(month_rad)

    # Has-MV flag for downstream filtering
    merged["has_meter_values"] = merged["n_samples"].fillna(0).astype(int) > 0

    # Drop the intermediate tx-only columns we've already folded in
    for col in ("start_ts", "stop_ts", "meter_start_wh", "meter_stop_wh"):
        if col in merged.columns:
            merged = merged.drop(columns=col)

    return merged
