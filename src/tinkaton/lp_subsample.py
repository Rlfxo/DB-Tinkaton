"""Deterministic per-station charger subsampling for LP N sweep.

HANDOFF v2.6 §8.3 runs the β × N × seed sweep on the signature station
``001SENGC02``. To keep the N sweep reproducible and comparable across
seeds, this module provides a single sampler that:

- Reads all chargers belonging to the given ``station_id`` (prefix
  match on ``charger_id``).
- Sorts them alphabetically so the enumeration is stable.
- Uses ``random.Random(seed).sample`` to draw ``n`` ids.

The minority-model charger (``001SENGC02009``, ELA007C01) is retained
in the draw pool — excluding it is a separate sensitivity knob
(§3.9). When ``n`` equals the station size, the returned list is the
sorted full roster regardless of seed.

Typical call:

    chargers = subsample_chargers(
        station_prefix="001SENGC02",
        n=20,
        seed=42,
        manifest=pd.read_csv("data/raw/phase_b/_charger_manifest.csv"),
    )
"""

from __future__ import annotations

import random
from collections.abc import Iterable

import pandas as pd

__all__ = [
    "list_station_chargers",
    "subsample_chargers",
    "sample_all_N_configurations",
]


def list_station_chargers(
    station_prefix: str, manifest: pd.DataFrame
) -> list[str]:
    """Return the sorted roster of ``charger_id`` matching ``station_prefix``."""
    if "charger_id" not in manifest.columns:
        raise KeyError("manifest must contain 'charger_id' column")
    mask = manifest["charger_id"].astype(str).str.startswith(station_prefix)
    return sorted(manifest.loc[mask, "charger_id"].dropna().astype(str).unique())


def subsample_chargers(
    *,
    station_prefix: str,
    n: int,
    seed: int,
    manifest: pd.DataFrame,
    exclude_charger_ids: Iterable[str] | None = None,
) -> list[str]:
    """Draw ``n`` charger ids deterministically from a station.

    The sampling is performed with ``random.Random(seed).sample`` over
    the sorted roster, so two calls with the same station, ``n``, and
    ``seed`` return identical lists.
    """
    roster = list_station_chargers(station_prefix, manifest)
    excluded = set(exclude_charger_ids or ())
    roster = [c for c in roster if c not in excluded]

    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n > len(roster):
        raise ValueError(
            f"station {station_prefix!r} has {len(roster)} chargers after exclusions; "
            f"cannot sample {n}"
        )
    if n == len(roster):
        return list(roster)

    rng = random.Random(seed)
    return sorted(rng.sample(roster, n))


def sample_all_N_configurations(  # noqa: N802 — capital N matches HANDOFF notation
    *,
    station_prefix: str,
    n_values: Iterable[int],
    seeds: Iterable[int],
    manifest: pd.DataFrame,
    exclude_charger_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a ``(n, seed)`` × charger membership matrix for the sweep.

    Each row corresponds to one (``n``, ``seed``) configuration; columns
    are the station's chargers; cells are ``1`` when the charger is in
    that sample and ``0`` otherwise. Useful for auditing which chargers
    appear across the full N × seed grid.
    """
    roster = list_station_chargers(station_prefix, manifest)
    records: list[dict] = []
    for n in n_values:
        for seed in seeds:
            sampled = set(
                subsample_chargers(
                    station_prefix=station_prefix,
                    n=n,
                    seed=seed,
                    manifest=manifest,
                    exclude_charger_ids=exclude_charger_ids,
                )
            )
            row = {"n": int(n), "seed": int(seed)}
            for c in roster:
                row[c] = int(c in sampled)
            records.append(row)
    return pd.DataFrame(records)
