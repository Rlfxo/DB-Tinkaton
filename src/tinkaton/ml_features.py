"""Leakage-safe feature matrix builder for departure-time prediction.

Consumes ``session_dataset_clean_v2.parquet`` and emits an X / y pair
plus a split assignment so that XGBoost / LSTM training, evaluation, and
hold-out residual export all draw from the same authoritative source.

Design principles:

- Only include features knowable at the *prediction horizon*. Two horizons
  are supported:
  1. ``"plug_in"``: immediately upon plug-in (arrival time + static
     charger metadata only).
  2. ``"plus_10min"``: plug-in plus the first-10-minute MeterValue
     profile. Rows without MeterValues (``has_meter_values = False``) get
     NaN for profile features — XGBoost handles NaN natively.
- Drop targets, session-mean metrics, end-of-session fields, identifiers,
  and PII (``start_id_tag``).
- Split assignment is driven by ``split_definition.json`` cutoffs so
  walk-forward validation is reproducible.

The target is ``duration_min`` (minutes between ``arrival_ts`` and
``plug_out_ts``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from .transform import derive_station_id

__all__ = [
    "FeatureHorizon",
    "FeatureBuildConfig",
    "FeatureMatrix",
    "build_feature_matrix",
    "load_split_cutoffs",
    "assign_split",
]


FeatureHorizon = Literal["plug_in", "plus_10min"]


# Columns immediately available at plug-in time.
_PLUG_IN_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "connector_id",
    "i_cap_observed_a",
    "charger_id",
    "station_id",
]

# Columns available only after the first MeterValue window.
_PROFILE_FEATURES = [
    "initial_mean_a",
    "initial_std_a",
    "initial_slope_a_per_min",
    "initial_n_samples",
]

# Categorical columns that XGBoost should treat as native categories.
_CATEGORICAL_COLUMNS = ("charger_id", "station_id")

# Columns that must never appear in X (leakage, identifiers, PII).
_EXCLUDED_COLUMNS = frozenset(
    {
        # Target + target-adjacent
        "duration_min",
        "plug_out_ts",
        "arrival_ts",
        # End-of-session or whole-session aggregates
        "energy_delivered_wh",
        "mean_current_a",
        "mean_voltage_v",
        "capacity_bound_flag",
        "capacity_bound_duration_min",
        "end_soc_pct",
        "stop_reason",
        "stop_id_tag",
        "n_samples",
        "binding_ratio_self",
        "binding_ratio_global",
        # Identifiers / PII / metadata
        "session_key",
        "transaction_id",
        "session_id",
        "message_id",
        "start_id_tag",
        "source_file",
        "estimation_quantile",
        "start_soc_pct",  # Phase B has no SoC; included for completeness
    }
)


@dataclass(frozen=True)
class FeatureBuildConfig:
    horizon: FeatureHorizon = "plus_10min"
    target_col: str = "duration_min"
    cap_duration_min: float | None = None  # optional outlier clip for target


@dataclass(frozen=True)
class FeatureMatrix:
    X: pd.DataFrame
    y: pd.Series
    split: pd.Series  # "train" | "val" | "test"
    arrival_ts: pd.Series
    charger_id: pd.Series
    feature_names: list[str]
    horizon: FeatureHorizon


def load_split_cutoffs(path: str | Path) -> dict[str, pd.Timestamp]:
    """Read ``split_definition.json`` and return the two cutoff timestamps."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cutoffs = data["cutoffs"]
    return {
        "train_end": pd.Timestamp(cutoffs["train_end_exclusive_of_val"]),
        "val_end": pd.Timestamp(cutoffs["val_end_exclusive_of_test"]),
    }


def assign_split(
    arrival_ts: pd.Series, cutoffs: dict[str, pd.Timestamp]
) -> pd.Series:
    """Map each arrival_ts to ``"train"`` / ``"val"`` / ``"test"``."""
    train_end = cutoffs["train_end"]
    val_end = cutoffs["val_end"]
    labels = pd.Series("test", index=arrival_ts.index, dtype="object")
    labels[arrival_ts <= train_end] = "train"
    labels[(arrival_ts > train_end) & (arrival_ts <= val_end)] = "val"
    return labels


def _apply_categorical_dtype(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def build_feature_matrix(
    sessions: pd.DataFrame,
    *,
    split_definition_path: str | Path,
    config: FeatureBuildConfig | None = None,
) -> FeatureMatrix:
    """Build an (X, y, split) tuple from the clean v2 session parquet.

    ``sessions`` must carry the columns emitted by
    :func:`tinkaton.dataset.write_session_dataset` plus the per-charger
    I_cap columns added by ``normalize_per_charger_icap.py``.
    """
    cfg = config or FeatureBuildConfig()
    if sessions.empty:
        raise ValueError("sessions DataFrame is empty")

    df = sessions.copy()

    # Derive station_id as a categorical feature.
    if "charger_id" in df.columns:
        df["station_id"] = df["charger_id"].map(derive_station_id)

    # Select features for the chosen horizon.
    features = list(_PLUG_IN_FEATURES)
    if cfg.horizon == "plus_10min":
        features += _PROFILE_FEATURES

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing expected feature columns: {missing}. "
            f"Ensure the input is session_dataset_clean_v2.parquet."
        )

    # Leakage sanity: nothing in features should be excluded.
    overlap = _EXCLUDED_COLUMNS.intersection(features)
    if overlap:
        raise AssertionError(
            f"Leakage: excluded column(s) {overlap} appeared in the feature list."
        )

    target = df[cfg.target_col].astype(float)
    if cfg.cap_duration_min is not None:
        target = target.clip(upper=cfg.cap_duration_min)

    X = _apply_categorical_dtype(df[features])  # noqa: N806 — X/y convention
    cutoffs = load_split_cutoffs(split_definition_path)
    split = assign_split(df["arrival_ts"], cutoffs)
    arrival_ts = df["arrival_ts"].copy()
    charger_id = df["charger_id"].copy()

    return FeatureMatrix(
        X=X,
        y=target,
        split=split,
        arrival_ts=arrival_ts,
        charger_id=charger_id,
        feature_names=features,
        horizon=cfg.horizon,
    )
