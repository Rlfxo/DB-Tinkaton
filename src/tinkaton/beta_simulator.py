"""β simulator — strategy comparison for LP peak-shaving results.

Five strategies evaluated per day on a fixed station roster:

- ``StatusQuo`` — historical replay: every active session draws at
  ``I_cap`` until ``E_target`` is delivered or the session ends. No
  optimization; gives the peak operators already experience.
- ``LoadBalance`` — reactive fair division of the contracted power at
  every slot across currently active sessions.
- ``LP-User`` — LP solved with true (β = 1) departure times.
- ``LP-ML`` — LP solved with ML-predicted (β = 0) departure times from
  ``xgb_residuals.parquet``.
- ``LP-Hybrid`` — β fraction of sessions reveal true departure; the rest
  use ML predictions. Selection is seed-deterministic.

HANDOFF v2.7 §8.3 grid: 6 β × 4 N × 10 seed × 1 mode × 5 strategy =
1,200 configurations (plus appendix sweeps). Each config aggregates
across the test period and emits one row to ``beta_sweep.csv``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .lp_solver import (
    DaySession,
    LPConstants,
    LPDayInput,
    solve_day,
)

__all__ = [
    "Strategy",
    "StrategyResult",
    "DayPlan",
    "evaluate_statusquo_day",
    "evaluate_loadbalance_day",
    "evaluate_lp_day",
    "build_day_plan",
]


Strategy = Literal["StatusQuo", "LoadBalance", "LP-ML", "LP-Hybrid", "LP-User"]


@dataclass(frozen=True)
class StrategyResult:
    strategy: Strategy
    day: pd.Timestamp
    peak_kw: float
    energy_delivered_wh: float
    n_sessions: int
    n_active_sessions: int    # sessions intersecting the day grid
    solve_time_s: float
    infeasible_share: float   # fraction of sessions whose energy target could not be met
    status: str


@dataclass(frozen=True)
class DayPlan:
    """Per-day slotted sessions for a fixed roster of chargers."""

    day: pd.Timestamp           # Midnight-aligned (UTC) day anchor
    charger_ids: tuple[str, ...]
    sessions: tuple[DaySession, ...]
    true_departure_slots: dict[int, int]   # session row → true departure slot
    n_slots: int


def _session_to_day_slot(
    session_row: pd.Series,
    day_start: pd.Timestamp,
    n_slots: int,
    slot_minutes: int,
) -> tuple[int, int]:
    """Clamp (arrival, departure) timestamps to slot indices within the day."""
    arrival = pd.Timestamp(session_row["arrival_ts"])
    departure = pd.Timestamp(session_row["plug_out_ts"])
    arrival_min = max(0.0, (arrival - day_start).total_seconds() / 60.0)
    departure_min = max(0.0, (departure - day_start).total_seconds() / 60.0)
    arrival_slot = int(np.floor(arrival_min / slot_minutes))
    departure_slot = int(np.ceil(departure_min / slot_minutes))
    arrival_slot = max(0, min(n_slots, arrival_slot))
    departure_slot = max(arrival_slot + 1, min(n_slots, departure_slot))
    return arrival_slot, departure_slot


def build_day_plan(
    sessions_df: pd.DataFrame,
    day: pd.Timestamp,
    charger_ids: tuple[str, ...],
    constants: LPConstants,
    departure_source: pd.Series | None = None,
) -> DayPlan:
    """Convert raw session rows for ``day`` into an LP-ready ``DayPlan``.

    ``sessions_df`` must contain ``charger_id``, ``arrival_ts``,
    ``plug_out_ts``, and ``energy_delivered_wh``. If ``departure_source``
    is supplied, it overrides the ``plug_out_ts`` column index-by-index —
    used to inject ML-predicted or mixed-horizon departures.
    """
    day_start = day.tz_convert("UTC").normalize()
    day_end = day_start + pd.Timedelta(days=1)
    n_slots = int(pd.Timedelta(days=1).total_seconds() / 60 / constants.slot_minutes)

    # Filter sessions whose activity overlaps this day.
    mask = (sessions_df["arrival_ts"] < day_end) & (sessions_df["plug_out_ts"] > day_start)
    day_sessions = sessions_df.loc[mask].copy()

    # Only keep sessions whose charger is part of the roster.
    charger_to_index = {cid: i for i, cid in enumerate(charger_ids)}
    day_sessions = day_sessions[day_sessions["charger_id"].isin(charger_to_index)]

    if departure_source is not None:
        override = departure_source.reindex(day_sessions.index)
        day_sessions["plug_out_ts"] = override.combine_first(day_sessions["plug_out_ts"])

    slotted: list[DaySession] = []
    true_dep_slots: dict[int, int] = {}
    for idx, row in day_sessions.iterrows():
        a, d = _session_to_day_slot(row, day_start, n_slots, constants.slot_minutes)
        if d <= a:
            continue
        target = float(row.get("energy_delivered_wh") or 0.0)
        # Cap target at physical maximum deliverable within the window
        max_deliverable = (
            (d - a) * constants.voltage_v * constants.eta_steady * constants.i_cap_a
            * constants.slot_hours
        )
        target = min(target, max_deliverable)
        idx_int = int(day_sessions.index.get_loc(idx))  # row index within day frame
        slotted.append(
            DaySession(
                charger_id=row["charger_id"],
                charger_index=charger_to_index[row["charger_id"]],
                arrival_slot=a,
                departure_slot=d,
                energy_target_wh=target,
            )
        )
        true_dep_slots[idx_int] = d

    return DayPlan(
        day=day,
        charger_ids=charger_ids,
        sessions=tuple(slotted),
        true_departure_slots=true_dep_slots,
        n_slots=n_slots,
    )


# ---------------------------------------------------------------------------
# Strategy evaluators
# ---------------------------------------------------------------------------


def evaluate_statusquo_day(plan: DayPlan, constants: LPConstants) -> StrategyResult:
    """Replay what the operator saw: every active session draws at cap.

    Peak kW at slot t = (number of sessions active at t) × per_charger_max_kw,
    clipped at the charger's individual energy target. Energy not delivered
    before departure counts toward ``infeasible_share``.
    """
    n_charg = len(plan.charger_ids)
    schedule = np.zeros((n_charg, plan.n_slots))
    energy_target = np.zeros(n_charg)

    for sess in plan.sessions:
        window = sess.departure_slot - sess.arrival_slot
        if window <= 0:
            continue
        full_rate_wh_per_slot = (
            constants.voltage_v * constants.eta_steady * constants.i_cap_a
            * constants.slot_hours
        )
        slots_needed = int(np.ceil(sess.energy_target_wh / full_rate_wh_per_slot))
        charged_slots = min(slots_needed, window)
        end_slot = sess.arrival_slot + charged_slots
        schedule[sess.charger_index, sess.arrival_slot:end_slot] = constants.i_cap_a
        energy_target[sess.charger_index] += sess.energy_target_wh

    station_kw = (
        constants.voltage_v * constants.eta_steady * schedule.sum(axis=0) / 1000.0
    )
    delivered = schedule.sum(axis=1) * (
        constants.voltage_v * constants.eta_steady * constants.slot_hours
    )
    infeasible = _infeasible_share(delivered, energy_target)

    return StrategyResult(
        strategy="StatusQuo",
        day=plan.day,
        peak_kw=float(station_kw.max()) if plan.n_slots else 0.0,
        energy_delivered_wh=float(delivered.sum()),
        n_sessions=len(plan.sessions),
        n_active_sessions=len(plan.sessions),
        solve_time_s=0.0,
        infeasible_share=infeasible,
        status="optimal",
    )


def evaluate_loadbalance_day(
    plan: DayPlan, constants: LPConstants, p_contract_kw: float
) -> StrategyResult:
    """Reactive fair division of P_contract at every slot."""
    n_charg = len(plan.charger_ids)
    if not plan.sessions:
        return StrategyResult(
            strategy="LoadBalance", day=plan.day, peak_kw=0.0,
            energy_delivered_wh=0.0, n_sessions=0, n_active_sessions=0,
            solve_time_s=0.0, infeasible_share=0.0, status="optimal",
        )

    active_by_slot: list[list[int]] = [[] for _ in range(plan.n_slots)]
    energy_target = np.zeros(n_charg)
    session_by_charger: dict[int, DaySession] = {}
    for sess in plan.sessions:
        for t in range(sess.arrival_slot, sess.departure_slot):
            active_by_slot[t].append(sess.charger_index)
        energy_target[sess.charger_index] += sess.energy_target_wh
        session_by_charger[sess.charger_index] = sess

    schedule = np.zeros((n_charg, plan.n_slots))
    station_kw_per_amp = constants.voltage_v * constants.eta_steady / 1000.0
    wh_per_amp_slot = (
        constants.voltage_v * constants.eta_steady * constants.slot_hours
    )
    remaining_wh = energy_target.copy()

    for t, active in enumerate(active_by_slot):
        if not active:
            continue
        # Fair kW share per charger, but respect individual I_max and
        # remaining-energy need so peak never exceeds contracted cap.
        fair_kw = p_contract_kw / len(active)
        fair_amps = min(fair_kw / station_kw_per_amp, constants.i_cap_a)
        for idx in active:
            remaining = remaining_wh[idx]
            if remaining <= 0:
                continue
            amps_needed_slot = remaining / wh_per_amp_slot
            amps = min(fair_amps, amps_needed_slot)
            schedule[idx, t] = amps
            remaining_wh[idx] -= amps * wh_per_amp_slot

    station_kw = station_kw_per_amp * schedule.sum(axis=0)
    delivered = schedule.sum(axis=1) * wh_per_amp_slot
    infeasible = _infeasible_share(delivered, energy_target)

    return StrategyResult(
        strategy="LoadBalance",
        day=plan.day,
        peak_kw=float(station_kw.max()) if plan.n_slots else 0.0,
        energy_delivered_wh=float(delivered.sum()),
        n_sessions=len(plan.sessions),
        n_active_sessions=len(plan.sessions),
        solve_time_s=0.0,
        infeasible_share=infeasible,
        status="optimal",
    )


def evaluate_lp_day(
    plan: DayPlan,
    constants: LPConstants,
    p_contract_kw: float,
    strategy: Strategy,
    solver: str = "HIGHS",
    max_seconds: float | None = None,
) -> StrategyResult:
    """Solve the LP for the plan's sessions and return aggregate metrics."""
    lp_input = LPDayInput(
        day=plan.day,
        n_chargers=len(plan.charger_ids),
        n_slots=plan.n_slots,
        sessions=plan.sessions,
        p_contract_kw=p_contract_kw,
        constants=constants,
    )
    result = solve_day(lp_input, solver=solver, max_seconds=max_seconds)

    if result.schedule_a is None:
        return StrategyResult(
            strategy=strategy, day=plan.day, peak_kw=float("nan"),
            energy_delivered_wh=0.0, n_sessions=len(plan.sessions),
            n_active_sessions=len(plan.sessions), solve_time_s=result.solve_time_s,
            infeasible_share=1.0, status=result.status,
        )

    energy_target = np.zeros(len(plan.charger_ids))
    for sess in plan.sessions:
        energy_target[sess.charger_index] += sess.energy_target_wh
    infeasible = _infeasible_share(result.energy_delivered_wh, energy_target)

    return StrategyResult(
        strategy=strategy,
        day=plan.day,
        peak_kw=result.peak_kw,
        energy_delivered_wh=float(result.energy_delivered_wh.sum()),
        n_sessions=len(plan.sessions),
        n_active_sessions=len(plan.sessions),
        solve_time_s=result.solve_time_s,
        infeasible_share=infeasible,
        status=result.status,
    )


def _infeasible_share(delivered: np.ndarray, target: np.ndarray) -> float:
    active = target > 0
    if not active.any():
        return 0.0
    tol_wh = 50.0  # ≤ 50 Wh shortfall counts as satisfied
    shortfalls = np.maximum(target[active] - delivered[active], 0.0)
    return float((shortfalls > tol_wh).mean())


def build_hybrid_departure_source(
    sessions_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    beta: float,
    seed: int,
) -> pd.Series:
    """Mix true and predicted plug-out timestamps deterministically.

    ``beta`` fraction of the predictions retain the true timestamp
    (user-declared); the rest adopt the XGBoost-predicted duration.
    Returns a Series aligned to ``sessions_df.index`` with the resolved
    ``plug_out_ts`` for each session.
    """
    merged = sessions_df[["charger_id", "transaction_id", "arrival_ts"]].merge(
        predictions_df[["charger_id", "arrival_ts", "y_pred"]],
        on=["charger_id", "arrival_ts"],
        how="left",
    )
    merged.index = sessions_df.index
    predicted_plug_out = merged["arrival_ts"] + pd.to_timedelta(merged["y_pred"], unit="m")

    rng = random.Random(seed)
    n = len(sessions_df)
    flags = np.array([rng.random() < beta for _ in range(n)])
    true_plug_out = sessions_df["plug_out_ts"]

    out = predicted_plug_out.copy()
    out[flags] = true_plug_out[flags]
    # Fall back to true when prediction is missing
    out = out.combine_first(true_plug_out)
    out.name = "plug_out_ts"
    return out
