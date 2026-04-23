from __future__ import annotations

import pandas as pd
import pytest

from tinkaton.transform import (
    SessionAggregateConfig,
    aggregate_sessions,
    compute_capacity_bound_flag,
    extract_initial_profile,
    normalize_measurement_columns,
)


def _ts_range(start: str, periods: int, freq_s: int = 30) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq=f"{freq_s}s", tz="UTC")


def _wide_session(
    *,
    charger_id: str = "C1",
    transaction_id: int = 100,
    session_id: str = "sess-1",
    periods: int = 20,
    current_a: list[float] | None = None,
    energy_start_wh: float = 1000.0,
    energy_step_wh: float = 60.0,
    voltage_v: float = 220.0,
    soc_start: float = 30.0,
    current_offered_a: float | None = None,
) -> pd.DataFrame:
    ts = _ts_range("2026-04-20 00:00:00", periods)
    currents = current_a if current_a is not None else [30.0] * periods
    rows = []
    for i, (t, c) in enumerate(zip(ts, currents, strict=True)):
        rows.append(
            {
                "timestamp": t,
                "server_timestamp": t,
                "charger_id": charger_id,
                "transaction_id": transaction_id,
                "session_id": session_id,
                "connector_id": 1,
                "message_id": f"m{i}",
                "current_a": c,
                "voltage_v": voltage_v,
                "energy_wh": energy_start_wh + i * energy_step_wh,
                "soc_pct": soc_start + i * 0.5,
                "current_offered_a": current_offered_a,
            }
        )
    return pd.DataFrame(rows)


def test_normalize_measurement_columns_maps_aliases():
    df = pd.DataFrame(
        {
            "current_outlet_a": [10.0, 20.0],
            "soc_ev_pct": [50.0, 51.0],
            "voltage_l1_v": [220.0, 221.0],
        }
    )
    out = normalize_measurement_columns(df)
    assert "current_a" in out.columns
    assert "soc_pct" in out.columns
    assert "voltage_v" in out.columns
    assert out["current_a"].tolist() == [10.0, 20.0]


def test_normalize_prefers_canonical_when_present():
    df = pd.DataFrame(
        {
            "current_a": [1.0, 2.0],
            "current_outlet_a": [99.0, 99.0],
        }
    )
    out = normalize_measurement_columns(df)
    assert out["current_a"].tolist() == [1.0, 2.0]


def test_normalize_coalesces_missing_from_alias():
    df = pd.DataFrame(
        {
            "current_a": [1.0, None],
            "current_outlet_a": [None, 22.0],
        }
    )
    out = normalize_measurement_columns(df)
    assert out["current_a"].tolist() == [1.0, 22.0]


def test_compute_capacity_bound_flag_true_when_bound_long_enough():
    df = _wide_session(periods=60, current_a=[31.0] * 60)
    flag, minutes = compute_capacity_bound_flag(df, min_minutes=20.0, i_pwm_assumed_a=31.2)
    assert flag is True
    assert minutes == pytest.approx(29.5, abs=0.1)  # 60 samples @ 30s = 29.5 min span


def test_compute_capacity_bound_flag_false_when_bound_too_short():
    bound = [31.0] * 20  # 20 samples @ 30s = 9.5 min
    slack = [10.0] * 20
    df = _wide_session(periods=40, current_a=bound + slack)
    flag, minutes = compute_capacity_bound_flag(df, min_minutes=20.0, i_pwm_assumed_a=31.2)
    assert flag is False
    assert minutes < 20.0


def test_compute_capacity_bound_flag_uses_current_offered_when_present():
    df = _wide_session(periods=60, current_a=[10.0] * 60, current_offered_a=10.0)
    flag, _ = compute_capacity_bound_flag(df, min_minutes=20.0)
    assert flag is True


def test_extract_initial_profile_mean_and_slope():
    df = _wide_session(periods=30, current_a=[10.0, 11.0, 12.0] + [15.0] * 27)
    profile = extract_initial_profile(df, window_min=1.5)
    assert profile["initial_n_samples"] == 4  # samples at 0, 30, 60, 90s
    assert profile["initial_mean_a"] == pytest.approx(12.0, abs=0.1)


def test_extract_initial_profile_empty_on_missing_column():
    df = pd.DataFrame({"timestamp": _ts_range("2026-01-01", 5)})
    profile = extract_initial_profile(df)
    assert profile["initial_n_samples"] == 0


def test_aggregate_sessions_basic():
    df = _wide_session(periods=60, current_a=[31.0] * 60)
    out = aggregate_sessions(df, config=SessionAggregateConfig(i_pwm_assumed_a=31.2))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["charger_id"] == "C1"
    assert row["transaction_id"] == 100
    assert row["n_samples"] == 60
    assert bool(row["capacity_bound_flag"]) is True
    assert row["duration_min"] == pytest.approx(29.5, abs=0.1)
    assert row["energy_delivered_wh"] == pytest.approx(59 * 60.0)


def test_aggregate_sessions_handles_non_monotonic_energy():
    df = _wide_session(periods=10)
    df.loc[5, "energy_wh"] = 100.0  # sudden drop
    out = aggregate_sessions(df)
    # after the drop energy recovers; delta from positive-only diffs
    # should still be finite and non-negative.
    assert len(out) == 1
    assert out.iloc[0]["energy_delivered_wh"] >= 0


def test_aggregate_sessions_drops_rows_without_keys():
    df = _wide_session(periods=5)
    df["transaction_id"] = None
    df["session_id"] = None
    out = aggregate_sessions(df)
    assert out.empty


def test_aggregate_sessions_multiple_sessions_independent():
    df1 = _wide_session(transaction_id=1, session_id="a", periods=10)
    df2 = _wide_session(
        transaction_id=2,
        session_id="b",
        periods=10,
        current_a=[5.0] * 10,
    )
    df = pd.concat([df1, df2], ignore_index=True)
    out = aggregate_sessions(df)
    assert len(out) == 2
    by_txn = out.set_index("transaction_id")
    assert by_txn.loc[1, "mean_current_a"] == pytest.approx(30.0)
    assert by_txn.loc[2, "mean_current_a"] == pytest.approx(5.0)


def test_aggregate_sessions_time_features_filled():
    df = _wide_session(periods=10)
    out = aggregate_sessions(df)
    row = out.iloc[0]
    assert -1.0 <= row["hour_sin"] <= 1.0
    assert -1.0 <= row["hour_cos"] <= 1.0
    assert 0 <= row["hour"] <= 23
    assert 0 <= row["dayofweek"] <= 6
    assert 1 <= row["month"] <= 12


def _tx_events(
    *,
    charger_id: str = "C1",
    transaction_id: int = 100,
    start: str = "2026-04-20 00:00:00",
    stop: str = "2026-04-20 00:20:00",
    meter_start_wh: float = 1000.0,
    meter_stop_wh: float = 3000.0,
    stop_reason: str = "Local",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "charger_id": charger_id,
                "transaction_id": transaction_id,
                "connector_id": 1,
                "start_ts": pd.Timestamp(start, tz="UTC"),
                "start_id_tag": "tag-a",
                "meter_start_wh": meter_start_wh,
                "stop_ts": pd.Timestamp(stop, tz="UTC"),
                "stop_id_tag": "tag-a",
                "meter_stop_wh": meter_stop_wh,
                "stop_reason": stop_reason,
            }
        ]
    )


def test_aggregate_sessions_prefers_transaction_timestamps():
    mv = _wide_session(periods=10, current_a=[30.0] * 10)
    tx = _tx_events(
        start="2026-04-20 00:05:00",
        stop="2026-04-20 01:05:00",
        meter_start_wh=500.0,
        meter_stop_wh=8000.0,
    )
    out = aggregate_sessions(mv, transaction_events=tx)
    row = out.iloc[0]
    assert row["arrival_ts"] == pd.Timestamp("2026-04-20 00:05:00", tz="UTC")
    assert row["plug_out_ts"] == pd.Timestamp("2026-04-20 01:05:00", tz="UTC")
    assert row["duration_min"] == pytest.approx(60.0, abs=0.1)
    assert row["energy_delivered_wh"] == pytest.approx(7500.0)
    assert row["stop_reason"] == "Local"
    assert bool(row["has_meter_values"]) is True


def test_aggregate_sessions_label_only_session_has_meter_values_false():
    mv = _wide_session(periods=10, current_a=[30.0] * 10, transaction_id=1)
    tx = pd.concat(
        [
            _tx_events(transaction_id=1),
            _tx_events(transaction_id=999, stop_reason="EVDisconnected"),
        ],
        ignore_index=True,
    )
    out = aggregate_sessions(mv, transaction_events=tx)
    assert len(out) == 2
    by_tx = out.set_index("transaction_id")
    assert bool(by_tx.loc[1, "has_meter_values"]) is True
    assert bool(by_tx.loc[999, "has_meter_values"]) is False
    assert pd.isna(by_tx.loc[999, "mean_current_a"])
    assert by_tx.loc[999, "stop_reason"] == "EVDisconnected"


def test_aggregate_sessions_tx_only_when_mv_empty():
    tx = _tx_events(transaction_id=7, stop_reason="Local")
    out = aggregate_sessions(pd.DataFrame(), transaction_events=tx)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["transaction_id"] == 7
    assert bool(row["has_meter_values"]) is False
    assert row["energy_delivered_wh"] == pytest.approx(2000.0)


def _icap_session(charger_id: str, mean_current: float, **overrides) -> dict:
    base = {
        "charger_id": charger_id,
        "has_meter_values": True,
        "mean_current_a": mean_current,
        "duration_min": 120.0,
        "stop_reason": "Other",
    }
    base.update(overrides)
    return base


def test_estimate_icap_per_charger_basic():
    from tinkaton.transform import estimate_icap_per_charger

    sessions = pd.DataFrame(
        [_icap_session("C1", c) for c in [10, 12, 15, 18, 20, 22, 24, 26, 28, 30, 31]]
        + [_icap_session("C2", c) for c in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
    )
    out = estimate_icap_per_charger(sessions, quantile=0.99)
    c1 = out.set_index("charger_id").loc["C1"]
    c2 = out.set_index("charger_id").loc["C2"]
    # 99th percentile for 11 samples is the top value (31 for C1, 15 for C2)
    assert c1["i_cap_observed_a"] == pytest.approx(31.0, abs=0.5)
    assert c2["i_cap_observed_a"] == pytest.approx(15.0, abs=0.5)
    assert c1["n_sessions_used"] == 11
    assert c2["n_sessions_used"] == 11


def test_estimate_icap_per_charger_filters_out_shorts_and_orphans():
    from tinkaton.transform import estimate_icap_per_charger

    sessions = pd.DataFrame(
        [_icap_session("C1", 30.0, duration_min=120.0, stop_reason="Other")]
        + [_icap_session("C1", 5.0, duration_min=1.0, stop_reason="Other")]
        + [_icap_session("C1", 5.0, duration_min=120.0, stop_reason="EVDisconnected")]
    )
    out = estimate_icap_per_charger(sessions)
    assert out.iloc[0]["n_sessions_used"] == 1
    assert out.iloc[0]["i_cap_observed_a"] == pytest.approx(30.0)


def test_estimate_icap_per_charger_uses_fallback_quantile_for_sparse_chargers():
    from tinkaton.transform import estimate_icap_per_charger

    sessions = pd.DataFrame([_icap_session("C1", c) for c in [10, 20]])
    out = estimate_icap_per_charger(sessions, quantile=0.99, fallback_quantile=0.95)
    row = out.iloc[0]
    assert row["estimation_quantile"] == 0.95


def test_estimate_icap_per_charger_empty_input():
    from tinkaton.transform import estimate_icap_per_charger

    out = estimate_icap_per_charger(pd.DataFrame())
    assert out.empty
    assert "i_cap_observed_a" in out.columns


def test_derive_station_id_strips_trailing_unit_digits():
    from tinkaton.transform import derive_station_id

    # Synthetic IDs in the real operator's naming shape: ``<region><site><unit>``
    # where the last three digits are the unit number within a station.
    assert derive_station_id("999ABCDEFG006") == "999ABCDEFG"
    assert derive_station_id("099XYZSITE01001") == "099XYZSITE01"
    assert derive_station_id("888STATION023") == "888STATION"


def test_derive_station_id_handles_none_and_unconventional():
    from tinkaton.transform import derive_station_id

    assert derive_station_id(None) is None
    assert derive_station_id("") is None
    # Shorter IDs without a 3-digit suffix fall through unchanged.
    assert derive_station_id("ABC12") == "ABC12"


def test_build_station_clusters_groups_and_flags_lp_range():
    from tinkaton.transform import build_station_clusters

    sessions = pd.DataFrame(
        {
            "charger_id": (
                ["STATION001"] * 5
                + ["STATION002"] * 3
                + ["OTHER001"] * 8
                + ["OTHER002"] * 7
                + ["BIG001"] * 20
            ),
        }
    )
    out = build_station_clusters(sessions, lp_min_chargers=2, lp_max_chargers=2)
    by_station = out.set_index("station_id")
    assert "STATION" in by_station.index
    assert by_station.loc["STATION", "n_chargers"] == 2
    assert by_station.loc["STATION", "n_sessions"] == 8
    # Only STATION has exactly 2 chargers in range
    assert bool(by_station.loc["STATION", "is_lp_candidate"]) is True
    assert bool(by_station.loc["BIG", "is_lp_candidate"]) is False


def test_build_station_clusters_attaches_models_from_manifest():
    from tinkaton.transform import build_station_clusters

    sessions = pd.DataFrame(
        {"charger_id": ["STATION001", "STATION002", "OTHER001"]}
    )
    manifest = pd.DataFrame(
        {
            "charger_id": ["STATION001", "STATION002", "OTHER001"],
            "model": ["ELA007C01", "ELA007C02", "E01AS007K10KR0101"],
        }
    )
    out = build_station_clusters(sessions, manifest=manifest)
    by_station = out.set_index("station_id")
    assert by_station.loc["STATION", "models"] == "ELA007C01|ELA007C02"
    assert by_station.loc["OTHER", "models"] == "E01AS007K10KR0101"


def test_build_station_clusters_empty_input():
    from tinkaton.transform import build_station_clusters

    out = build_station_clusters(pd.DataFrame())
    assert out.empty
    assert "station_id" in out.columns
    assert "is_lp_candidate" in out.columns
