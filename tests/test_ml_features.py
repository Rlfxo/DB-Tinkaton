from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tinkaton.ml_features import (
    _EXCLUDED_COLUMNS,
    FeatureBuildConfig,
    assign_split,
    build_feature_matrix,
    load_split_cutoffs,
)


def _sample_session(**overrides) -> dict:
    base = {
        "charger_id": "STATION001",
        "arrival_ts": pd.Timestamp("2026-02-01 10:00:00", tz="UTC"),
        "plug_out_ts": pd.Timestamp("2026-02-01 12:00:00", tz="UTC"),
        "duration_min": 120.0,
        "hour_sin": 0.5,
        "hour_cos": 0.5,
        "dow_sin": 0.1,
        "dow_cos": 0.9,
        "month_sin": -0.5,
        "month_cos": 0.9,
        "connector_id": 1.0,
        "i_cap_observed_a": 30.0,
        "initial_mean_a": 25.0,
        "initial_std_a": 2.0,
        "initial_slope_a_per_min": 0.5,
        "initial_n_samples": 20,
        "has_meter_values": True,
        "mean_current_a": 28.0,  # must NOT end up in X
        "energy_delivered_wh": 4000.0,  # must NOT end up in X
        "stop_reason": "Other",  # must NOT end up in X
        "start_id_tag": "12345",  # PII — must NOT end up in X
    }
    base.update(overrides)
    return base


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _write_split(tmp_path: Path, train_end: str, val_end: str) -> Path:
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "cutoffs": {
                    "train_end_exclusive_of_val": train_end,
                    "val_end_exclusive_of_test": val_end,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_split_cutoffs(tmp_path: Path):
    path = _write_split(
        tmp_path,
        "2026-02-15T00:00:00+00:00",
        "2026-03-15T00:00:00+00:00",
    )
    cutoffs = load_split_cutoffs(path)
    assert cutoffs["train_end"] == pd.Timestamp("2026-02-15 00:00:00", tz="UTC")
    assert cutoffs["val_end"] == pd.Timestamp("2026-03-15 00:00:00", tz="UTC")


def test_assign_split_labels_by_cutoff():
    arrivals = pd.Series(
        [
            pd.Timestamp("2026-01-10", tz="UTC"),
            pd.Timestamp("2026-02-20", tz="UTC"),
            pd.Timestamp("2026-04-01", tz="UTC"),
        ]
    )
    cutoffs = {
        "train_end": pd.Timestamp("2026-02-15", tz="UTC"),
        "val_end": pd.Timestamp("2026-03-15", tz="UTC"),
    }
    labels = assign_split(arrivals, cutoffs)
    assert labels.tolist() == ["train", "val", "test"]


def test_build_feature_matrix_excludes_leakage_columns(tmp_path: Path):
    df = _frame([_sample_session(charger_id=f"STATION{i:03d}") for i in range(5)])
    split_path = _write_split(
        tmp_path,
        "2026-02-15T00:00:00+00:00",
        "2026-03-15T00:00:00+00:00",
    )
    fm = build_feature_matrix(df, split_definition_path=split_path)
    for col in _EXCLUDED_COLUMNS:
        assert col not in fm.X.columns, f"leakage column {col!r} present in X"
    assert "duration_min" not in fm.X.columns
    assert fm.y.name == "duration_min"
    assert len(fm.y) == len(fm.X)


def test_build_feature_matrix_has_profile_features_in_plus_10min_horizon(tmp_path: Path):
    df = _frame([_sample_session()])
    split_path = _write_split(
        tmp_path,
        "2026-02-15T00:00:00+00:00",
        "2026-03-15T00:00:00+00:00",
    )
    fm = build_feature_matrix(df, split_definition_path=split_path)
    for name in ("initial_mean_a", "initial_std_a", "initial_slope_a_per_min"):
        assert name in fm.X.columns


def test_build_feature_matrix_omits_profile_features_for_plug_in_horizon(tmp_path: Path):
    df = _frame([_sample_session()])
    split_path = _write_split(
        tmp_path,
        "2026-02-15T00:00:00+00:00",
        "2026-03-15T00:00:00+00:00",
    )
    fm = build_feature_matrix(
        df,
        split_definition_path=split_path,
        config=FeatureBuildConfig(horizon="plug_in"),
    )
    for name in ("initial_mean_a", "initial_std_a"):
        assert name not in fm.X.columns
    assert "hour_sin" in fm.X.columns


def test_build_feature_matrix_assigns_station_id(tmp_path: Path):
    df = _frame([_sample_session(charger_id="999ABCDEFG006")])
    split_path = _write_split(
        tmp_path,
        "2026-02-15T00:00:00+00:00",
        "2026-03-15T00:00:00+00:00",
    )
    fm = build_feature_matrix(df, split_definition_path=split_path)
    assert "station_id" in fm.X.columns
    # Suffix-stripped station id
    assert fm.X["station_id"].iloc[0] == "999ABCDEFG"


def test_build_feature_matrix_categorical_dtype(tmp_path: Path):
    df = _frame([_sample_session(charger_id=f"STATION{i:03d}") for i in range(3)])
    split_path = _write_split(
        tmp_path,
        "2026-02-15T00:00:00+00:00",
        "2026-03-15T00:00:00+00:00",
    )
    fm = build_feature_matrix(df, split_definition_path=split_path)
    assert fm.X["charger_id"].dtype.name == "category"
    assert fm.X["station_id"].dtype.name == "category"


def test_build_feature_matrix_cap_duration(tmp_path: Path):
    df = _frame(
        [
            _sample_session(duration_min=60.0),
            _sample_session(duration_min=5000.0),  # would exceed cap
        ]
    )
    split_path = _write_split(
        tmp_path,
        "2026-02-15T00:00:00+00:00",
        "2026-03-15T00:00:00+00:00",
    )
    fm = build_feature_matrix(
        df,
        split_definition_path=split_path,
        config=FeatureBuildConfig(cap_duration_min=2000.0),
    )
    assert fm.y.max() == 2000.0


def test_build_feature_matrix_empty_raises(tmp_path: Path):
    split_path = _write_split(
        tmp_path,
        "2026-02-15T00:00:00+00:00",
        "2026-03-15T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="empty"):
        build_feature_matrix(pd.DataFrame(), split_definition_path=split_path)


def test_build_feature_matrix_splits_correctly(tmp_path: Path):
    rows = [
        _sample_session(charger_id="C1", arrival_ts=pd.Timestamp("2026-01-01", tz="UTC")),
        _sample_session(charger_id="C1", arrival_ts=pd.Timestamp("2026-02-20", tz="UTC")),
        _sample_session(charger_id="C1", arrival_ts=pd.Timestamp("2026-04-01", tz="UTC")),
    ]
    split_path = _write_split(
        tmp_path,
        "2026-02-15T00:00:00+00:00",
        "2026-03-15T00:00:00+00:00",
    )
    fm = build_feature_matrix(_frame(rows), split_definition_path=split_path)
    assert fm.split.tolist() == ["train", "val", "test"]
