"""CVXPY LP for station-level PWM peak-shaving scheduling.

Implements the formulation in HANDOFF_ModelPipeline_v2.md §3.3–§3.4:

    min  P_peak
    s.t. P_peak  ≥  V · η · Σ_i I[i, t]                for all t
         V · η · Σ_i I[i, t] · (slot_min / 60)  ≥  0   (energy delivered — below)
         Σ_t V · η · I[i, t] · (slot_min / 60) ≥ E_target[i]
         0 ≤ I[i, t] ≤ I_max        for t ∈ [arrival_i, departure_i)
         I[i, t] = 0                otherwise
         V · η · Σ_i I[i, t]  ≤  P_contract            for all t

A single ``solve_day`` call plans one calendar day (1,440 1-minute slots
by default) for one charger roster. The β simulator invokes this once per
(day, strategy) combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cvxpy as cp
import numpy as np
import pandas as pd

__all__ = [
    "LPConstants",
    "DaySession",
    "LPDayInput",
    "LPDayResult",
    "solve_day",
]


@dataclass(frozen=True)
class LPConstants:
    """Physical constants shared across all LP runs."""

    eta_steady: float = 0.98
    i_cap_a: float = 30.0
    voltage_v: float = 220.0
    slot_minutes: int = 1
    per_charger_max_kw: float = field(init=False)
    slot_hours: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "per_charger_max_kw",
            self.voltage_v * self.eta_steady * self.i_cap_a / 1000.0,
        )
        object.__setattr__(self, "slot_hours", self.slot_minutes / 60.0)


@dataclass(frozen=True)
class DaySession:
    """One charging session, expressed in slot indices for a given day grid."""

    charger_id: str
    charger_index: int          # row in the LP decision-variable matrix
    arrival_slot: int           # inclusive
    departure_slot: int         # exclusive — scheduler horizon upper bound
    energy_target_wh: float


@dataclass(frozen=True)
class LPDayInput:
    day: pd.Timestamp
    n_chargers: int
    n_slots: int
    sessions: tuple[DaySession, ...]
    p_contract_kw: float
    constants: LPConstants


@dataclass
class LPDayResult:
    day: pd.Timestamp
    status: str
    solve_time_s: float
    peak_kw: float
    schedule_a: np.ndarray | None  # shape (n_chargers, n_slots); amps per slot
    energy_delivered_wh: np.ndarray | None  # shape (n_chargers,)
    n_sessions: int
    infeasible: bool


def _active_mask(session: DaySession, n_slots: int) -> np.ndarray:
    mask = np.zeros(n_slots, dtype=bool)
    a = max(0, session.arrival_slot)
    d = min(n_slots, session.departure_slot)
    if d > a:
        mask[a:d] = True
    return mask


def solve_day(
    lp_input: LPDayInput,
    *,
    solver: str = "HIGHS",
    fallback_solver: str = "CLARABEL",
    max_seconds: float | None = None,
    verbose: bool = False,
) -> LPDayResult:
    """Solve one day's LP and return the schedule + station peak."""
    cst = lp_input.constants
    n_chargers = lp_input.n_chargers
    n_slots = lp_input.n_slots

    if not lp_input.sessions:
        # Empty day: peak = 0, nothing to schedule.
        return LPDayResult(
            day=lp_input.day,
            status="empty",
            solve_time_s=0.0,
            peak_kw=0.0,
            schedule_a=np.zeros((n_chargers, n_slots)),
            energy_delivered_wh=np.zeros(n_chargers),
            n_sessions=0,
            infeasible=False,
        )

    station_kw_per_amp = cst.voltage_v * cst.eta_steady / 1000.0  # kW per amp
    energy_per_amp_slot = cst.voltage_v * cst.eta_steady * cst.slot_hours  # Wh per amp-slot

    current = cp.Variable((n_chargers, n_slots), nonneg=True, name="I")
    peak = cp.Variable(nonneg=True, name="P_peak")

    constraints: list[cp.Constraint] = []

    # Build the active-window mask and apply it as upper-bound zeros
    active_mask = np.zeros((n_chargers, n_slots), dtype=bool)
    energy_targets = np.zeros(n_chargers, dtype=float)
    for sess in lp_input.sessions:
        mask = _active_mask(sess, n_slots)
        active_mask[sess.charger_index] |= mask
        energy_targets[sess.charger_index] += sess.energy_target_wh

    # I[i, t] ≤ I_cap where active, 0 where not.
    cap_matrix = np.where(active_mask, cst.i_cap_a, 0.0)
    constraints.append(current <= cap_matrix)

    # Energy constraint per charger: only rows with a session contribute.
    for i, target in enumerate(energy_targets):
        if target <= 0:
            continue
        energy_delivered = cp.sum(current[i, :]) * energy_per_amp_slot
        constraints.append(energy_delivered >= target)

    # Station peak constraint
    station_kw_per_slot = station_kw_per_amp * cp.sum(current, axis=0)
    constraints.append(station_kw_per_slot <= peak)

    # Contracted power cap
    if lp_input.p_contract_kw > 0:
        constraints.append(station_kw_per_slot <= lp_input.p_contract_kw)

    problem = cp.Problem(cp.Minimize(peak), constraints)

    solver_kwargs: dict = {"verbose": verbose}
    if max_seconds is not None:
        solver_kwargs["time_limit"] = max_seconds

    try:
        problem.solve(solver=solver, **solver_kwargs)
    except (cp.SolverError, ValueError):
        problem.solve(solver=fallback_solver, verbose=verbose)

    status: Literal["optimal", "infeasible", "unbounded", "unknown"] = problem.status
    solve_time = float(problem.solver_stats.solve_time or 0.0)

    if status not in ("optimal", "optimal_inaccurate"):
        return LPDayResult(
            day=lp_input.day,
            status=status,
            solve_time_s=solve_time,
            peak_kw=float("nan"),
            schedule_a=None,
            energy_delivered_wh=None,
            n_sessions=len(lp_input.sessions),
            infeasible=status == "infeasible",
        )

    schedule = np.asarray(current.value, dtype=float)
    schedule = np.clip(schedule, 0.0, cst.i_cap_a)
    energy = schedule.sum(axis=1) * energy_per_amp_slot

    return LPDayResult(
        day=lp_input.day,
        status=status,
        solve_time_s=solve_time,
        peak_kw=float(peak.value),
        schedule_a=schedule,
        energy_delivered_wh=energy,
        n_sessions=len(lp_input.sessions),
        infeasible=False,
    )


def compute_peak_from_schedule(schedule_a: np.ndarray, constants: LPConstants) -> float:
    """Station-level peak in kW from a per-charger-per-slot amp schedule."""
    station_kw_per_amp = constants.voltage_v * constants.eta_steady / 1000.0
    return float(station_kw_per_amp * schedule_a.sum(axis=0).max())
