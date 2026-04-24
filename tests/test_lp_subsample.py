from __future__ import annotations

import pandas as pd
import pytest

from tinkaton.lp_subsample import (
    list_station_chargers,
    sample_all_N_configurations,
    subsample_chargers,
)


def _manifest(charger_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"charger_id": charger_ids, "model": ["X"] * len(charger_ids)})


def test_list_station_chargers_returns_sorted():
    manifest = _manifest(
        [
            "STATION001",
            "STATION003",
            "STATION002",
            "OTHER001",
        ]
    )
    out = list_station_chargers("STATION", manifest)
    assert out == ["STATION001", "STATION002", "STATION003"]


def test_list_station_chargers_missing_column_raises():
    manifest = pd.DataFrame({"model": ["X"]})
    with pytest.raises(KeyError):
        list_station_chargers("STATION", manifest)


def test_subsample_chargers_deterministic_for_same_seed():
    manifest = _manifest([f"STATION{i:03d}" for i in range(1, 27)])
    first = subsample_chargers(
        station_prefix="STATION", n=10, seed=42, manifest=manifest
    )
    second = subsample_chargers(
        station_prefix="STATION", n=10, seed=42, manifest=manifest
    )
    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10


def test_subsample_chargers_different_seeds_give_different_samples():
    manifest = _manifest([f"STATION{i:03d}" for i in range(1, 27)])
    a = subsample_chargers(station_prefix="STATION", n=10, seed=1, manifest=manifest)
    b = subsample_chargers(station_prefix="STATION", n=10, seed=2, manifest=manifest)
    assert a != b  # Virtually certain given C(26,10)


def test_subsample_chargers_full_roster_returns_all_independent_of_seed():
    manifest = _manifest([f"STATION{i:03d}" for i in range(1, 5)])
    full = list_station_chargers("STATION", manifest)
    for seed in (1, 2, 99):
        out = subsample_chargers(
            station_prefix="STATION", n=len(full), seed=seed, manifest=manifest
        )
        assert out == full


def test_subsample_chargers_n_too_large_raises():
    manifest = _manifest([f"STATION{i:03d}" for i in range(1, 5)])
    with pytest.raises(ValueError, match="cannot sample"):
        subsample_chargers(
            station_prefix="STATION", n=10, seed=42, manifest=manifest
        )


def test_subsample_chargers_n_not_positive_raises():
    manifest = _manifest([f"STATION{i:03d}" for i in range(1, 5)])
    with pytest.raises(ValueError, match="positive"):
        subsample_chargers(
            station_prefix="STATION", n=0, seed=42, manifest=manifest
        )


def test_subsample_chargers_respects_exclude_list():
    manifest = _manifest([f"STATION{i:03d}" for i in range(1, 5)])
    out = subsample_chargers(
        station_prefix="STATION",
        n=3,
        seed=42,
        manifest=manifest,
        exclude_charger_ids=["STATION002"],
    )
    assert "STATION002" not in out
    assert len(out) == 3


def test_subsample_chargers_output_is_sorted():
    manifest = _manifest([f"STATION{i:03d}" for i in range(1, 27)])
    out = subsample_chargers(
        station_prefix="STATION", n=10, seed=42, manifest=manifest
    )
    assert out == sorted(out)


def test_sample_all_n_configurations_shape():  # noqa: N802 — name mirrors HANDOFF N
    manifest = _manifest([f"STATION{i:03d}" for i in range(1, 27)])
    df = sample_all_N_configurations(
        station_prefix="STATION",
        n_values=[10, 15, 20, 26],
        seeds=range(3),
        manifest=manifest,
    )
    assert len(df) == 4 * 3  # 4 × 3 configurations
    assert "n" in df.columns and "seed" in df.columns
    # Each row's charger-column sum equals n
    charger_cols = [c for c in df.columns if c.startswith("STATION")]
    for _, row in df.iterrows():
        assert int(row[charger_cols].sum()) == int(row["n"])


def test_sample_all_n_configurations_excludes_outlier():  # noqa: N802
    manifest = _manifest([f"STATION{i:03d}" for i in range(1, 10)])
    df = sample_all_N_configurations(
        station_prefix="STATION",
        n_values=[5],
        seeds=range(3),
        manifest=manifest,
        exclude_charger_ids=["STATION005"],
    )
    assert (df["STATION005"] == 0).all()
