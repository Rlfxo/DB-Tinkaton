"""Build Train/Val/Test split_definition.json and station_clusters.csv.

Consumes ``data/phase_b/session_dataset_clean_v2.parquet`` and the AC
charger manifest, and emits two artifacts listed as P1 follow-ups in
``HANDOFF_ModelPipeline_v2.md §6.3``:

- ``data/phase_b/split_definition.json`` — temporal quantile-based
  Train/Val/Test cutoffs (60/15/25 default). Walk-forward consumers
  can pin on these timestamps for reproducibility.
- ``data/phase_b/station_clusters.csv`` — station-level aggregation
  via prefix stripping, flagging stations in the 10–30 charger range
  as LP simulation candidates.

Both files carry the source path and the ISO generation timestamp so
they stay self-describing when re-examined months later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tinkaton.transform import build_station_clusters

ROOT = Path(__file__).resolve().parents[1]
SESSIONS_PARQUET = ROOT / "data" / "phase_b" / "session_dataset_clean_v2.parquet"
MANIFEST_CSV = ROOT / "data" / "raw" / "phase_b" / "_charger_manifest.csv"
SPLIT_JSON = ROOT / "data" / "phase_b" / "split_definition.json"
CLUSTERS_CSV = ROOT / "data" / "phase_b" / "station_clusters.csv"

TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.15
TEST_FRACTION = 0.25

LP_MIN = 10
LP_MAX = 30


@dataclass(frozen=True)
class SplitRange:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_sessions: int
    n_with_mv: int


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).isoformat()


def build_split_definition(sessions: pd.DataFrame) -> dict:
    """Temporal quantile split on ``arrival_ts``.

    Uses strictly monotonic cutoffs so a session belongs to exactly one
    bucket. The cutoffs are the arrival timestamps of the 60th and 75th
    percentile sessions, anchored to the actual data distribution
    rather than a prescribed calendar window.
    """
    ordered = sessions.sort_values("arrival_ts").reset_index(drop=True)
    n_total = len(ordered)
    train_end_idx = int(n_total * TRAIN_FRACTION)
    val_end_idx = int(n_total * (TRAIN_FRACTION + VAL_FRACTION))

    train_end_ts = ordered.iloc[train_end_idx - 1]["arrival_ts"]
    val_end_ts = ordered.iloc[val_end_idx - 1]["arrival_ts"]

    train = ordered.iloc[:train_end_idx]
    val = ordered.iloc[train_end_idx:val_end_idx]
    test = ordered.iloc[val_end_idx:]

    ranges = [
        SplitRange(
            name="train",
            start=train.iloc[0]["arrival_ts"],
            end=train.iloc[-1]["arrival_ts"],
            n_sessions=len(train),
            n_with_mv=int(train["has_meter_values"].astype(bool).sum()),
        ),
        SplitRange(
            name="val",
            start=val.iloc[0]["arrival_ts"],
            end=val.iloc[-1]["arrival_ts"],
            n_sessions=len(val),
            n_with_mv=int(val["has_meter_values"].astype(bool).sum()),
        ),
        SplitRange(
            name="test",
            start=test.iloc[0]["arrival_ts"],
            end=test.iloc[-1]["arrival_ts"],
            n_sessions=len(test),
            n_with_mv=int(test["has_meter_values"].astype(bool).sum()),
        ),
    ]

    return {
        "method": "temporal_quantile_on_arrival_ts",
        "target_fractions": {
            "train": TRAIN_FRACTION,
            "val": VAL_FRACTION,
            "test": TEST_FRACTION,
        },
        "cutoffs": {
            "train_end_exclusive_of_val": _iso(train_end_ts),
            "val_end_exclusive_of_test": _iso(val_end_ts),
        },
        "splits": [
            {
                "name": r.name,
                "start": _iso(r.start),
                "end": _iso(r.end),
                "n_sessions": r.n_sessions,
                "n_with_meter_values": r.n_with_mv,
                "fraction_of_total": round(r.n_sessions / n_total, 4),
            }
            for r in ranges
        ],
        "totals": {
            "n_sessions": n_total,
            "n_with_meter_values": int(ordered["has_meter_values"].astype(bool).sum()),
        },
        "provenance": {
            "source_parquet": str(SESSIONS_PARQUET.relative_to(ROOT)),
            "generated_at_kst": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        },
        "notes": [
            "Walk-forward consumers: use `cutoffs` timestamps as strict upper bounds.",
            "Rows are assigned to train if arrival_ts <= train_end_exclusive_of_val, "
            "to val if arrival_ts <= val_end_exclusive_of_test and > the train cutoff, "
            "and to test otherwise.",
        ],
    }


def main() -> None:
    if not SESSIONS_PARQUET.exists():
        raise SystemExit(f"Missing {SESSIONS_PARQUET}. Run the normalization pipeline first.")
    if not MANIFEST_CSV.exists():
        raise SystemExit(f"Missing {MANIFEST_CSV}. Run the Phase B export first.")

    sessions = pd.read_parquet(SESSIONS_PARQUET)
    manifest = pd.read_csv(MANIFEST_CSV)
    print(f"sessions: {len(sessions):,}  chargers: {sessions['charger_id'].nunique()}")
    print(f"manifest: {len(manifest):,} AC chargers")

    # --- split_definition.json ---
    split_def = build_split_definition(sessions)
    SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_JSON.write_text(json.dumps(split_def, indent=2), encoding="utf-8")
    print()
    print(f"wrote {SPLIT_JSON.relative_to(ROOT)}")
    for split in split_def["splits"]:
        print(
            f"  {split['name']:5s}  {split['start'][:10]} → {split['end'][:10]}  "
            f"n={split['n_sessions']:>6,}  ({split['fraction_of_total']*100:.1f}%)  "
            f"mv={split['n_with_meter_values']:,}"
        )

    # --- station_clusters.csv ---
    clusters = build_station_clusters(
        sessions, manifest=manifest, lp_min_chargers=LP_MIN, lp_max_chargers=LP_MAX
    )
    clusters["generated_at_kst"] = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
    clusters["source_parquet"] = str(SESSIONS_PARQUET.relative_to(ROOT))
    clusters.to_csv(CLUSTERS_CSV, index=False)
    print()
    print(f"wrote {CLUSTERS_CSV.relative_to(ROOT)}")
    print(f"  stations total: {len(clusters)}")
    lp_candidates = clusters[clusters["is_lp_candidate"]]
    print(f"  LP candidates ({LP_MIN}–{LP_MAX} chargers): {len(lp_candidates)}")
    for _, row in lp_candidates.iterrows():
        print(
            f"    {row['station_id']:20s}  chargers={int(row['n_chargers']):>3}  "
            f"sessions={int(row['n_sessions']):>6,}"
        )


if __name__ == "__main__":
    main()
