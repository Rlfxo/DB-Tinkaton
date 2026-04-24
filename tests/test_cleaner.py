from __future__ import annotations

import pandas as pd

from tinkaton.cleaner import SessionCleanConfig, clean_sessions


def _row(**overrides) -> dict:
    base = {
        "transaction_id": 1,
        "duration_min": 60.0,
        "energy_delivered_wh": 4000.0,
        "stop_reason": "Other",
    }
    base.update(overrides)
    return base


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_clean_sessions_keeps_valid_row():
    df = _frame([_row()])
    result = clean_sessions(df)
    assert len(result.clean) == 1
    assert len(result.rejected) == 0
    assert result.summary["kept"] == 1


def test_clean_sessions_rejects_non_positive_duration():
    df = _frame(
        [
            _row(transaction_id=1, duration_min=-5.0),
            _row(transaction_id=2, duration_min=0.0),
            _row(transaction_id=3, duration_min=None),
        ]
    )
    result = clean_sessions(df)
    assert len(result.clean) == 0
    assert len(result.rejected) == 3
    assert set(result.rejected["rejection_reason"]) == {"non_positive_duration"}


def test_clean_sessions_rejects_duration_exceeding_max():
    df = _frame([_row(duration_min=48 * 60 + 1)])
    result = clean_sessions(df)
    assert result.rejected.iloc[0]["rejection_reason"] == "duration_above_max"


def test_clean_sessions_rejects_negative_energy():
    df = _frame([_row(energy_delivered_wh=-10.0)])
    result = clean_sessions(df)
    assert result.rejected.iloc[0]["rejection_reason"] == "non_positive_energy"


def test_clean_sessions_rejects_impossible_energy_rate():
    # 60 min session delivering 50 kWh → 50 kWh/h, far above the 7.5 limit.
    df = _frame([_row(duration_min=60, energy_delivered_wh=50_000)])
    result = clean_sessions(df)
    assert result.rejected.iloc[0]["rejection_reason"] == "energy_rate_above_physical_max"


def test_clean_sessions_rejects_orphan_when_configured():
    df = _frame([_row(stop_reason=None)])
    result = clean_sessions(df)
    assert result.rejected.iloc[0]["rejection_reason"] == "orphan_no_stop_reason"


def test_clean_sessions_can_keep_orphan_when_disabled():
    df = _frame([_row(stop_reason=None)])
    cfg = SessionCleanConfig(drop_orphan_stop_reason=False)
    result = clean_sessions(df, config=cfg)
    assert len(result.clean) == 1
    assert len(result.rejected) == 0


def test_clean_sessions_allowed_stop_reasons_subset():
    df = _frame(
        [
            _row(transaction_id=1, stop_reason="Other"),
            _row(transaction_id=2, stop_reason="EmergencyStop"),
        ]
    )
    cfg = SessionCleanConfig(allowed_stop_reasons=("Other", "Local"))
    result = clean_sessions(df, config=cfg)
    assert len(result.clean) == 1
    assert result.clean.iloc[0]["transaction_id"] == 1
    assert result.rejected.iloc[0]["rejection_reason"] == "stop_reason_not_allowed"


def test_clean_sessions_summary_counts_match():
    df = _frame(
        [
            _row(transaction_id=1),
            _row(transaction_id=2, duration_min=-1),
            _row(transaction_id=3, energy_delivered_wh=-1),
        ]
    )
    result = clean_sessions(df)
    assert result.summary["input"] == 3
    assert result.summary["kept"] == 1
    assert result.summary["rejected"] == 2
    assert result.summary["reject:non_positive_duration"] == 1
    assert result.summary["reject:non_positive_energy"] == 1


def test_clean_sessions_empty_input():
    df = pd.DataFrame()
    result = clean_sessions(df)
    assert result.clean.empty
    assert result.rejected.empty
    assert result.summary == {}


def test_clean_sessions_respects_charger_blacklist():
    df = pd.DataFrame(
        [
            _row(transaction_id=1) | {"charger_id": "GOOD_CHARGER"},
            _row(transaction_id=2) | {"charger_id": "BAD_CHARGER"},
            _row(transaction_id=3) | {"charger_id": "BAD_CHARGER"},
        ]
    )
    from tinkaton.cleaner import SessionCleanConfig

    cfg = SessionCleanConfig(blacklist_charger_ids=("BAD_CHARGER",))
    result = clean_sessions(df, config=cfg)
    assert len(result.clean) == 1
    assert result.clean.iloc[0]["charger_id"] == "GOOD_CHARGER"
    assert len(result.rejected) == 2
    assert set(result.rejected["rejection_reason"]) == {"charger_blacklisted"}


def test_clean_sessions_default_blacklist_drops_known_broken_charger():
    df = pd.DataFrame(
        [
            _row(transaction_id=1) | {"charger_id": "003DJKCRUN003"},
            _row(transaction_id=2) | {"charger_id": "NORMAL_CHARGER"},
        ]
    )
    result = clean_sessions(df)
    assert len(result.clean) == 1
    assert result.clean.iloc[0]["charger_id"] == "NORMAL_CHARGER"
    assert result.rejected.iloc[0]["rejection_reason"] == "charger_blacklisted"


def test_clean_sessions_blacklist_can_be_disabled():
    df = pd.DataFrame(
        [_row(transaction_id=1) | {"charger_id": "003DJKCRUN003"}]
    )
    from tinkaton.cleaner import SessionCleanConfig

    cfg = SessionCleanConfig(blacklist_charger_ids=())
    result = clean_sessions(df, config=cfg)
    assert len(result.clean) == 1
    assert len(result.rejected) == 0
