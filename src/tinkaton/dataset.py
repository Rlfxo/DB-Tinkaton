"""Dataset assembly — turn raw OCPP logs into session-level parquet files.

Orchestrates :func:`tinkaton.loader.load_ocpp_logs` →
:func:`tinkaton.loader.meter_values_to_dataframe` →
:func:`tinkaton.transform.aggregate_sessions` and emits the canonical
``session_dataset_raw.parquet`` and ``session_dataset.parquet`` defined
in HANDOFF_ModelPipeline v2 §5.2.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .loader import (
    load_ocpp_logs,
    meter_values_to_dataframe,
    transaction_events_to_dataframe,
)
from .transform import SessionAggregateConfig, aggregate_sessions

__all__ = [
    "BuildArtifacts",
    "build_session_dataset",
    "write_session_dataset",
]


@dataclass(frozen=True)
class BuildArtifacts:
    raw_path: Path
    ml_ready_path: Path
    n_sessions: int
    n_source_files: int


_ML_READY_COLUMNS = [
    "charger_id",
    "transaction_id",
    "session_id",
    "connector_id",
    "arrival_ts",
    "plug_out_ts",
    "duration_min",
    "n_samples",
    "has_meter_values",
    "energy_delivered_wh",
    "mean_current_a",
    "mean_voltage_v",
    "start_soc_pct",
    "end_soc_pct",
    "capacity_bound_flag",
    "capacity_bound_duration_min",
    "stop_reason",
    "start_id_tag",
    "stop_id_tag",
    "initial_mean_a",
    "initial_std_a",
    "initial_slope_a_per_min",
    "initial_n_samples",
    "hour",
    "dayofweek",
    "month",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]


def _collect_paths(sources: Iterable[str | Path]) -> list[Path]:
    """Expand sources into ``*.json`` file paths, excluding metadata files."""
    paths: list[Path] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            # Exclude metadata files like ``_checkpoint.json`` or ``_manifest.json``
            paths.extend(
                sorted(q for q in p.glob("*.json") if not q.name.startswith("_"))
            )
        elif p.is_file():
            paths.append(p)
    return paths


def _aggregate_per_file(
    path: Path,
    config: SessionAggregateConfig | None,
    charger_metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    """Load one OCPP JSON file and return its session rows."""
    logs = load_ocpp_logs(path)
    mv = meter_values_to_dataframe(logs)
    tx = transaction_events_to_dataframe(logs)
    if mv.empty and tx.empty:
        return pd.DataFrame()
    return aggregate_sessions(
        mv,
        config=config,
        charger_metadata=charger_metadata,
        transaction_events=tx if not tx.empty else None,
    )


def _build_sessions_streaming(
    paths: Iterable[Path],
    config: SessionAggregateConfig | None,
    charger_metadata: pd.DataFrame | None,
    progress_every: int = 0,
) -> tuple[pd.DataFrame, int]:
    frames: list[pd.DataFrame] = []
    n_source = 0
    for idx, path in enumerate(paths, start=1):
        sessions = _aggregate_per_file(path, config, charger_metadata)
        if not sessions.empty:
            sessions["source_file"] = path.name
            frames.append(sessions)
            n_source += 1
        if progress_every and idx % progress_every == 0:
            print(f"  processed {idx} files  (sessions so far: {sum(len(f) for f in frames):,})")
    if not frames:
        return pd.DataFrame(), n_source
    return pd.concat(frames, ignore_index=True, sort=False), n_source


def build_session_dataset(
    sources: Iterable[str | Path],
    charger_metadata: pd.DataFrame | None = None,
    config: SessionAggregateConfig | None = None,
    *,
    progress_every: int = 0,
) -> pd.DataFrame:
    """Load OCPP JSON files and return a session-level DataFrame.

    ``sources`` may point at individual files or directories; directories
    are scanned for ``*.json`` children (non-recursive). Metadata files
    whose names begin with ``_`` (checkpoint, manifest) are ignored.

    Files are aggregated one at a time; only session-level rows live in
    memory simultaneously, so the function scales to thousands of
    per-charger exports without loading every MeterValue at once.
    """
    sessions, _ = _build_sessions_streaming(
        _collect_paths(sources), config, charger_metadata, progress_every
    )
    return sessions


def write_session_dataset(
    sources: Iterable[str | Path],
    out_dir: str | Path,
    *,
    charger_metadata: pd.DataFrame | None = None,
    config: SessionAggregateConfig | None = None,
    raw_filename: str = "session_dataset_raw.parquet",
    ml_filename: str = "session_dataset.parquet",
    progress_every: int = 0,
) -> BuildArtifacts:
    """Build session dataset and write ``_raw`` and ML-ready parquet files.

    See :func:`build_session_dataset` for source semantics. Sessions are
    aggregated file-by-file, so memory stays proportional to session
    count, not raw document count.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _collect_paths(sources)
    sessions, n_files = _build_sessions_streaming(
        paths, config, charger_metadata, progress_every
    )
    if sessions.empty:
        empty = pd.DataFrame()
        empty.to_parquet(out_dir / raw_filename)
        empty.to_parquet(out_dir / ml_filename)
        return BuildArtifacts(
            raw_path=out_dir / raw_filename,
            ml_ready_path=out_dir / ml_filename,
            n_sessions=0,
            n_source_files=0,
        )
    for col in ("transaction_id",):
        if col in sessions.columns:
            sessions[col] = pd.to_numeric(sessions[col], errors="coerce").astype("Int64")
    for col in ("session_id", "session_key"):
        if col in sessions.columns:
            sessions[col] = sessions[col].astype("string")
    raw_path = out_dir / raw_filename
    ml_path = out_dir / ml_filename
    sessions.to_parquet(raw_path, index=False)
    ml_columns = [c for c in _ML_READY_COLUMNS if c in sessions.columns]
    extra_meta_cols = [c for c in ("floor", "charger_type") if c in sessions.columns]
    sessions.reindex(columns=ml_columns + extra_meta_cols).to_parquet(ml_path, index=False)
    return BuildArtifacts(
        raw_path=raw_path,
        ml_ready_path=ml_path,
        n_sessions=int(len(sessions)),
        n_source_files=n_files,
    )
