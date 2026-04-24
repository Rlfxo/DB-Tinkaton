"""Orchestrator for the LP β × N × seed × strategy sweep.

Reads ``configs/lp_simulation.yaml`` + ``data/phase_b/session_dataset_clean_v2.parquet``
+ ``results/xgb_residuals.parquet`` and emits ``results/lp_sweep/beta_sweep.csv``
plus a small summary JSON.

Design:
- One (β, N, seed, strategy) combination per iteration.
- Each combination aggregates across test-period days for that station.
- Progress printed every ``--progress-every`` iterations.

CLI flags:
- ``--days N``       limit to the first N test-period days (for pilots).
- ``--n-values``     override N sweep (e.g. "25" for a single N pilot).
- ``--betas``        override β sweep.
- ``--seeds``        override seed list.
- ``--strategies``   override strategies.
- ``--solver``       LP solver name (default from config).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml

from tinkaton.beta_simulator import (
    Strategy,
    build_day_plan,
    build_hybrid_departure_source,
    evaluate_loadbalance_day,
    evaluate_lp_day,
    evaluate_statusquo_day,
)
from tinkaton.lp_solver import LPConstants
from tinkaton.lp_subsample import subsample_chargers

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "lp_simulation.yaml"


@dataclass
class Row:
    strategy: str
    n_chargers: int
    beta: float
    seed: int
    mode: str
    n_days: int
    n_sessions_total: int
    peak_mean_kw: float
    peak_median_kw: float
    peak_p95_kw: float
    peak_max_kw: float
    energy_delivered_total_kwh: float
    infeasible_mean: float
    solve_time_total_s: float
    p_contract_kw: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--n-values", type=int, nargs="*", default=None)
    p.add_argument("--betas", type=float, nargs="*", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--strategies", nargs="*", default=None)
    p.add_argument("--p-contract-kw", type=float, default=None)
    p.add_argument("--solver", default=None)
    p.add_argument("--progress-every", type=int, default=1)
    return p.parse_args()


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _load_station_sessions(cfg: dict) -> pd.DataFrame:
    path = ROOT / cfg["inputs"]["sessions_parquet"]
    df = pd.read_parquet(path)
    station_id = cfg["station"]["id"]
    df = df[df["charger_id"].str.startswith(station_id)].copy()
    df = df.dropna(subset=["arrival_ts", "plug_out_ts", "energy_delivered_wh"])
    df["arrival_ts"] = pd.to_datetime(df["arrival_ts"], utc=True)
    df["plug_out_ts"] = pd.to_datetime(df["plug_out_ts"], utc=True)
    df = df.reset_index(drop=True)
    return df


def _load_predictions(cfg: dict) -> pd.DataFrame:
    path = ROOT / cfg["inputs"]["epsilon_primary"]
    df = pd.read_parquet(path)
    df["arrival_ts"] = pd.to_datetime(df["arrival_ts"], utc=True)
    return df[["charger_id", "arrival_ts", "y_pred"]]


def _iter_test_days(cfg: dict, limit: int | None) -> list[pd.Timestamp]:
    start = pd.Timestamp(cfg["test_period"]["start_utc"], tz="UTC").normalize()
    end = pd.Timestamp(cfg["test_period"]["end_utc"], tz="UTC").normalize()
    days = pd.date_range(start, end, freq="1D", tz="UTC")
    if limit is not None:
        days = days[:limit]
    return list(days)


def _select_roster(cfg: dict, manifest: pd.DataFrame, n: int, seed: int) -> tuple[str, ...]:
    chargers = subsample_chargers(
        station_prefix=cfg["station"]["id"],
        n=n,
        seed=seed,
        manifest=manifest,
        exclude_charger_ids=cfg["scope"]["excluded_charger_ids"],
    )
    return tuple(chargers)


def _aggregate_rows(results: list, combo_label: dict, p_contract_kw: float) -> Row:
    peaks = [r.peak_kw for r in results if not pd.isna(r.peak_kw)]
    solve_total = sum(r.solve_time_s for r in results)
    energy_total_kwh = sum(r.energy_delivered_wh for r in results) / 1000.0
    infeasibles = [r.infeasible_share for r in results]
    return Row(
        strategy=combo_label["strategy"],
        n_chargers=combo_label["n"],
        beta=combo_label["beta"],
        seed=combo_label["seed"],
        mode=combo_label["mode"],
        n_days=len(results),
        n_sessions_total=sum(r.n_sessions for r in results),
        peak_mean_kw=float(pd.Series(peaks).mean()) if peaks else float("nan"),
        peak_median_kw=float(pd.Series(peaks).median()) if peaks else float("nan"),
        peak_p95_kw=float(pd.Series(peaks).quantile(0.95)) if peaks else float("nan"),
        peak_max_kw=float(max(peaks)) if peaks else float("nan"),
        energy_delivered_total_kwh=energy_total_kwh,
        infeasible_mean=float(pd.Series(infeasibles).mean()) if infeasibles else 0.0,
        solve_time_total_s=solve_total,
        p_contract_kw=p_contract_kw,
    )


def _evaluate_config(
    sessions: pd.DataFrame,
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    days: list[pd.Timestamp],
    constants: LPConstants,
    cfg: dict,
    n: int,
    beta: float,
    seed: int,
    strategy: Strategy,
    solver: str,
    p_contract_kw: float,
) -> Row:
    charger_ids = _select_roster(cfg, manifest, n, seed)

    # Decide which plug_out timestamps to feed the LP for this run.
    if strategy in ("StatusQuo", "LoadBalance", "LP-User"):
        departure_source = None        # use true plug_out_ts
    elif strategy == "LP-ML":
        departure_source = build_hybrid_departure_source(
            sessions, predictions, beta=0.0, seed=seed
        )
    else:  # LP-Hybrid
        departure_source = build_hybrid_departure_source(
            sessions, predictions, beta=beta, seed=seed
        )

    day_results = []
    for day in days:
        plan = build_day_plan(sessions, day, charger_ids, constants, departure_source)
        if strategy == "StatusQuo":
            day_results.append(evaluate_statusquo_day(plan, constants))
        elif strategy == "LoadBalance":
            day_results.append(
                evaluate_loadbalance_day(plan, constants, p_contract_kw)
            )
        else:
            day_results.append(
                evaluate_lp_day(
                    plan,
                    constants,
                    p_contract_kw,
                    strategy=strategy,
                    solver=solver,
                )
            )

    return _aggregate_rows(
        day_results,
        combo_label={"strategy": strategy, "n": n, "beta": beta, "seed": seed, "mode": "offline"},
        p_contract_kw=p_contract_kw,
    )


def main() -> None:
    args = parse_args()
    cfg = _load_config()
    sessions = _load_station_sessions(cfg)
    predictions = _load_predictions(cfg)
    manifest = pd.read_csv(ROOT / cfg["inputs"]["manifest_csv"])

    n_values = args.n_values or cfg["sweep"]["n_values"]
    betas = args.betas or cfg["sweep"]["beta_values"]
    seeds = args.seeds or cfg["sweep"]["seeds"]
    strategies = args.strategies or cfg["sweep"]["strategies"]
    p_contract_kw = args.p_contract_kw or cfg["station"]["p_contract_kw"]
    solver = args.solver or cfg["solver"]["primary"]

    constants = LPConstants(
        eta_steady=cfg["lp_constants"]["eta_steady"],
        i_cap_a=cfg["lp_constants"]["i_cap_a"],
        voltage_v=cfg["lp_constants"]["voltage_v"],
        slot_minutes=cfg["lp_constants"]["slot_minutes"],
    )

    days = _iter_test_days(cfg, args.days)
    print(
        f"station={cfg['station']['id']}  P_contract={p_contract_kw} kW  "
        f"days={len(days)}  solver={solver}"
    )
    print(
        f"grid: N={n_values}  β={betas}  seeds={seeds}  strategies={strategies}"
    )

    # Short-circuit strategy combos that don't depend on β
    rows: list[Row] = []
    t_start = time.monotonic()
    total_configs = 0

    def _emit(row: Row) -> None:
        rows.append(row)
        n_done = len(rows)
        if args.progress_every and n_done % args.progress_every == 0:
            elapsed = time.monotonic() - t_start
            print(
                f"  [{n_done:>4}/{total_configs:>4}]  {row.strategy:<12} "
                f"N={row.n_chargers:<2}  β={row.beta:.1f}  seed={row.seed}  "
                f"peak_mean={row.peak_mean_kw:.2f} kW  "
                f"infeasible={row.infeasible_mean*100:.1f}%  "
                f"elapsed={elapsed:.1f}s"
            )

    # Enumerate the combos we actually need to run
    combos: list[dict] = []
    for strategy in strategies:
        for n in n_values:
            for seed in seeds:
                beta_iter = betas if strategy == "LP-Hybrid" else [0.0]
                for beta in beta_iter:
                    combos.append(
                        {"strategy": strategy, "n": n, "beta": beta, "seed": seed}
                    )
    total_configs = len(combos)
    print(f"total configurations: {total_configs}")

    for combo in combos:
        row = _evaluate_config(
            sessions, predictions, manifest, days, constants, cfg,
            n=combo["n"], beta=combo["beta"], seed=combo["seed"],
            strategy=combo["strategy"], solver=solver,
            p_contract_kw=p_contract_kw,
        )
        _emit(row)

    results_df = pd.DataFrame([asdict(r) for r in rows])
    out_dir = ROOT / cfg["outputs"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "beta_sweep.csv", index=False)

    summary = {
        "n_configs": len(rows),
        "n_days": len(days),
        "p_contract_kw": p_contract_kw,
        "total_solve_time_s": float(results_df["solve_time_total_s"].sum()),
        "peak_kw_by_strategy": (
            results_df.groupby("strategy")["peak_mean_kw"]
            .agg(["mean", "median", "min", "max"])
            .to_dict()
        ),
    }
    (out_dir / "beta_sweep_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print()
    print(f"wrote {(out_dir / 'beta_sweep.csv').relative_to(ROOT)}")
    print(f"wrote {(out_dir / 'beta_sweep_summary.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
