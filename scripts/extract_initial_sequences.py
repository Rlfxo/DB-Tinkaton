"""Extract first-10-minute current sequences per session for LSTM input.

Per HANDOFF v2.5 §7.2, the LSTM baseline consumes a fixed-length
sequence of session-start currents alongside the static features used
by XGBoost. This script walks ``data/raw/phase_b/*.json`` once,
resamples each session's first ``WINDOW_MIN`` minutes of
``current_a`` to ``N_STEPS`` equal bins, and persists the result as
``data/phase_b/initial_sequences.parquet`` keyed by
``(charger_id, transaction_id)``.

Missing bins are filled with 0.0 so the output has a stable shape.
Sessions whose MeterValues do not reach the window are still emitted
(zero-padded), so downstream LSTM dataloaders never need to special-case
MV-absent rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tinkaton.loader import load_ocpp_logs, meter_values_to_dataframe
from tinkaton.transform import normalize_measurement_columns

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "raw" / "phase_b"
SESSIONS_PARQUET = ROOT / "data" / "phase_b" / "session_dataset_clean_v2.parquet"
OUTPUT_PARQUET = ROOT / "data" / "phase_b" / "initial_sequences.parquet"

WINDOW_MIN = 10.0
N_STEPS = 20
STEP_SECONDS = WINDOW_MIN * 60 / N_STEPS  # 30 seconds


def _resample_to_fixed_bins(mv: pd.DataFrame, arrival_ts: pd.Timestamp) -> np.ndarray:
    """Average ``current_a`` within each of ``N_STEPS`` fixed bins."""
    if mv.empty or "current_a" not in mv.columns:
        return np.zeros(N_STEPS, dtype=np.float32)

    window_end = arrival_ts + pd.Timedelta(minutes=WINDOW_MIN)
    sub = mv[(mv["timestamp"] >= arrival_ts) & (mv["timestamp"] < window_end)]
    if sub.empty:
        return np.zeros(N_STEPS, dtype=np.float32)

    elapsed = (sub["timestamp"] - arrival_ts).dt.total_seconds().to_numpy()
    bins = np.floor(elapsed / STEP_SECONDS).astype(int)
    bins = np.clip(bins, 0, N_STEPS - 1)
    currents = sub["current_a"].to_numpy(dtype=np.float64)

    out = np.zeros(N_STEPS, dtype=np.float64)
    counts = np.zeros(N_STEPS, dtype=np.int32)
    for b, c in zip(bins, currents, strict=True):
        if np.isnan(c):
            continue
        out[b] += c
        counts[b] += 1
    mask = counts > 0
    out[mask] /= counts[mask]
    return out.astype(np.float32)


def main() -> None:
    if not SESSIONS_PARQUET.exists():
        raise SystemExit(f"Missing {SESSIONS_PARQUET}.")

    sessions = pd.read_parquet(SESSIONS_PARQUET)
    # Keys we need to produce a sequence for: every clean session.
    required_keys = (
        sessions[["charger_id", "transaction_id", "arrival_ts"]]
        .dropna(subset=["charger_id", "transaction_id"])
        .copy()
    )
    required_keys["transaction_id"] = required_keys["transaction_id"].astype("Int64")
    required_by_charger: dict[str, pd.DataFrame] = {
        cid: grp.sort_values("arrival_ts").reset_index(drop=True)
        for cid, grp in required_keys.groupby("charger_id")
    }

    print(
        f"sessions to produce: {len(required_keys):,}  "
        f"chargers: {len(required_by_charger)}"
    )

    rows: list[dict] = []
    json_paths = sorted(p for p in SOURCE_DIR.glob("*.json") if not p.name.startswith("_"))
    for idx, path in enumerate(json_paths, start=1):
        charger_id = path.stem
        targets = required_by_charger.get(charger_id)
        if targets is None or targets.empty:
            continue
        logs = load_ocpp_logs(path)
        mv_df = meter_values_to_dataframe(logs)
        if mv_df.empty:
            # Emit zero sequences for each target so every session has a row
            for _, row in targets.iterrows():
                rows.append(
                    {
                        "charger_id": charger_id,
                        "transaction_id": int(row["transaction_id"]),
                        "sequence": np.zeros(N_STEPS, dtype=np.float32),
                        "n_mv_samples_in_window": 0,
                    }
                )
            continue

        mv_df = normalize_measurement_columns(mv_df)
        if "current_a" not in mv_df.columns:
            for _, row in targets.iterrows():
                rows.append(
                    {
                        "charger_id": charger_id,
                        "transaction_id": int(row["transaction_id"]),
                        "sequence": np.zeros(N_STEPS, dtype=np.float32),
                        "n_mv_samples_in_window": 0,
                    }
                )
            continue

        mv_df = mv_df[["timestamp", "transaction_id", "current_a"]].copy()
        mv_df["transaction_id"] = pd.to_numeric(
            mv_df["transaction_id"], errors="coerce"
        ).astype("Int64")

        for _, row in targets.iterrows():
            tx = int(row["transaction_id"])
            arrival = pd.Timestamp(row["arrival_ts"])
            sub = mv_df[mv_df["transaction_id"] == tx]
            window_end = arrival + pd.Timedelta(minutes=WINDOW_MIN)
            in_window = sub[
                (sub["timestamp"] >= arrival) & (sub["timestamp"] < window_end)
            ]
            seq = _resample_to_fixed_bins(sub, arrival)
            rows.append(
                {
                    "charger_id": charger_id,
                    "transaction_id": tx,
                    "sequence": seq,
                    "n_mv_samples_in_window": int(len(in_window)),
                }
            )
        if idx % 50 == 0:
            print(f"  processed {idx} charger files, rows so far: {len(rows):,}")

    df = pd.DataFrame(rows)
    # Split the ndarray column into N_STEPS scalar columns — parquet friendly.
    sequence_matrix = np.stack(df["sequence"].to_list()).astype(np.float32)
    for i in range(N_STEPS):
        df[f"seq_step_{i:02d}"] = sequence_matrix[:, i]
    df = df.drop(columns=["sequence"])

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print()
    print(f"wrote {OUTPUT_PARQUET.relative_to(ROOT)}")
    print(f"rows: {len(df):,}")
    nonzero = (sequence_matrix.sum(axis=1) > 0).mean() * 100
    print(
        f"sessions with at least one non-zero bin: "
        f"{nonzero:.1f} % (others are MV-absent or outside the window)"
    )
    print(f"mean sequence amplitude: {sequence_matrix.mean():.2f} A")
    print(f"shape per row: ({N_STEPS},)")


if __name__ == "__main__":
    main()
