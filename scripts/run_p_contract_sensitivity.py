"""P_contract sensitivity sweep for the LP simulator.

Pilot finding (v2.8 §7.7): the β-framework's measurable effect on peak
reduction is concentrated in a narrow P_contract regime around 60–80 kW
at station 001SENGC02. This script runs a dedicated sweep across
P_contract values to produce the Ch.7 sub-figure that visualises the
regime boundary.

Each row of the output covers one (P_contract, β, seed, strategy)
configuration aggregated across the full test period. The orchestrator
reuses ``beta_simulator`` primitives but varies ``P_contract`` across
the configured sensitivity list; N is fixed at the design-scope 25.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from tinkaton.beta_simulator import (
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=None, help="Limit to first N test days.")
    p.add_argument(
        "--p-contracts",
        type=float,
        nargs="*",
        default=None,
        help="Override P_contract sweep values.",
    )
    p.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=[0.0, 0.5, 1.0],
        help="β values (default compact 0.0/0.5/1.0).",
    )
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--n", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    sessions = pd.read_parquet(ROOT / cfg["inputs"]["sessions_parquet"])
    sessions = sessions[sessions["charger_id"].str.startswith(cfg["station"]["id"])].copy()
    sessions = sessions.dropna(
        subset=["arrival_ts", "plug_out_ts", "energy_delivered_wh"]
    )
    sessions["arrival_ts"] = pd.to_datetime(sessions["arrival_ts"], utc=True)
    sessions["plug_out_ts"] = pd.to_datetime(sessions["plug_out_ts"], utc=True)
    sessions = sessions.reset_index(drop=True)

    predictions = pd.read_parquet(ROOT / cfg["inputs"]["epsilon_primary"])
    predictions["arrival_ts"] = pd.to_datetime(predictions["arrival_ts"], utc=True)
    predictions = predictions[["charger_id", "arrival_ts", "y_pred"]]

    manifest = pd.read_csv(ROOT / cfg["inputs"]["manifest_csv"])

    constants = LPConstants(
        eta_steady=cfg["lp_constants"]["eta_steady"],
        i_cap_a=cfg["lp_constants"]["i_cap_a"],
        voltage_v=cfg["lp_constants"]["voltage_v"],
        slot_minutes=cfg["lp_constants"]["slot_minutes"],
    )

    start = pd.Timestamp(cfg["test_period"]["start_utc"], tz="UTC").normalize()
    end = pd.Timestamp(cfg["test_period"]["end_utc"], tz="UTC").normalize()
    days = list(pd.date_range(start, end, freq="1D", tz="UTC"))
    if args.days:
        days = days[: args.days]

    p_contracts = args.p_contracts or cfg["sweep"]["p_contract_sensitivity_kw"]
    betas = args.betas
    seeds = args.seeds
    n = args.n

    # LP-Hybrid and the two baselines (StatusQuo, LoadBalance) cover the
    # regime — LP-User / LP-ML are β = 1 / β = 0 degenerate cases of Hybrid.
    strategies = ["StatusQuo", "LoadBalance", "LP-Hybrid"]

    print(
        f"station={cfg['station']['id']}  N={n}  days={len(days)}  "
        f"P_contracts={p_contracts}  betas={betas}  seeds={seeds}"
    )

    records: list[dict] = []
    total = 0
    for strategy in strategies:
        beta_iter = betas if strategy == "LP-Hybrid" else [0.0]
        total += len(p_contracts) * len(seeds) * len(beta_iter)
    print(f"total configurations: {total}")

    t_start = time.monotonic()
    done = 0

    for p_contract_kw in p_contracts:
        for strategy in strategies:
            beta_iter = betas if strategy == "LP-Hybrid" else [0.0]
            for seed in seeds:
                charger_ids = tuple(
                    subsample_chargers(
                        station_prefix=cfg["station"]["id"],
                        n=n,
                        seed=seed,
                        manifest=manifest,
                        exclude_charger_ids=cfg["scope"]["excluded_charger_ids"],
                    )
                )
                for beta in beta_iter:
                    if strategy == "LP-Hybrid":
                        departure_source = build_hybrid_departure_source(
                            sessions, predictions, beta=beta, seed=seed
                        )
                    else:
                        departure_source = None

                    day_results = []
                    for day in days:
                        plan = build_day_plan(
                            sessions, day, charger_ids, constants, departure_source
                        )
                        if strategy == "StatusQuo":
                            r = evaluate_statusquo_day(plan, constants)
                        elif strategy == "LoadBalance":
                            r = evaluate_loadbalance_day(plan, constants, p_contract_kw)
                        else:
                            r = evaluate_lp_day(
                                plan, constants, p_contract_kw, strategy="LP-Hybrid"
                            )
                        day_results.append(r)

                    peaks = [r.peak_kw for r in day_results if not pd.isna(r.peak_kw)]
                    records.append(
                        {
                            "p_contract_kw": p_contract_kw,
                            "strategy": strategy,
                            "beta": beta,
                            "seed": seed,
                            "n_chargers": n,
                            "n_days_feasible": len(peaks),
                            "peak_mean_kw": float(
                                pd.Series(peaks).mean()) if peaks else float("nan"),
                            "peak_median_kw": float(
                                pd.Series(peaks).median()) if peaks else float("nan"),
                            "peak_p95_kw": float(
                                pd.Series(peaks).quantile(0.95)) if peaks else float("nan"),
                            "infeasible_mean": float(
                                pd.Series([r.infeasible_share for r in day_results]).mean()
                            ),
                            "energy_kwh_total": sum(
                                r.energy_delivered_wh for r in day_results
                            ) / 1000.0,
                        }
                    )
                    done += 1
                    elapsed = time.monotonic() - t_start
                    print(
                        f"  [{done:>3}/{total:>3}]  "
                        f"Pc={p_contract_kw:>5.0f} kW  {strategy:<12}  "
                        f"β={beta:.1f}  seed={seed}  "
                        f"peak_mean={records[-1]['peak_mean_kw']:.2f}  "
                        f"infeas={records[-1]['infeasible_mean']*100:.0f}%  "
                        f"t={elapsed:.1f}s"
                    )

    out_dir = ROOT / cfg["outputs"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "p_contract_sensitivity.csv", index=False)

    summary = {
        "n_configs": len(records),
        "p_contracts": list(p_contracts),
        "betas": list(betas),
        "seeds": list(seeds),
        "n_chargers": n,
        "n_days": len(days),
    }
    (out_dir / "p_contract_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print()
    print(f"wrote {(out_dir / 'p_contract_sensitivity.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
