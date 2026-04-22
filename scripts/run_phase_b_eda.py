"""Pilot EDA — build the session dataset from available logs and emit plots.

Data inputs (pilot scope):
- ``data/raw/260417-ST-22-98.json`` (Phase A, L7 full session)
- ``data/raw/260420-ST-93-100.json`` (Phase A, L7 taper zoom)
- ``data/processed/*.json`` (Phase B fragments from random-cars-logs.json)

The Phase A sessions are tagged so we can separate them when reporting
Phase B distributions. Plots land in ``outputs/phase_b_eda/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tinkaton.dataset import write_session_dataset
from tinkaton.transform import SessionAggregateConfig

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT_DATA = ROOT / "data" / "phase_b"
OUT_REPORT_DIR = ROOT / "reports"
OUT_FIG_DIR = ROOT / "outputs" / "phase_b_eda"

PHASE_A_FILES = {
    "260417-ST-22-98.json": "PhaseA-L7-Full",
    "260420-ST-93-100.json": "PhaseA-L7-Taper",
}


def main() -> None:
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    raw_sources = [RAW / name for name in PHASE_A_FILES] + [PROCESSED]
    artifacts = write_session_dataset(
        raw_sources,
        out_dir=OUT_DATA,
        config=SessionAggregateConfig(),
    )
    print(f"source_files: {artifacts.n_source_files}")
    print(f"sessions:     {artifacts.n_sessions}")
    print(f"raw_parquet:  {artifacts.raw_path}")
    print(f"ml_parquet:   {artifacts.ml_ready_path}")

    sessions = pd.read_parquet(artifacts.raw_path)
    if sessions.empty:
        print("no sessions — skipping plots")
        return

    phase_a_dates = {"2026-04-17", "2026-04-19", "2026-04-20"}
    sessions["cohort"] = sessions["arrival_ts"].dt.strftime("%Y-%m-%d").where(
        sessions["arrival_ts"].dt.strftime("%Y-%m-%d").isin(phase_a_dates),
        other="PhaseB-fragments",
    )
    sessions["cohort"] = sessions["cohort"].replace(
        {d: "PhaseA" for d in phase_a_dates}
    )

    _plot_duration_hist(sessions)
    _plot_samples_vs_duration(sessions)
    _plot_sessions_per_charger(sessions)
    _plot_arrival_hour_heatmap(sessions)
    _plot_capacity_bound_share(sessions)

    _write_markdown_report(sessions, artifacts)


def _plot_duration_hist(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        df["duration_min"].clip(upper=180),
        bins=40,
        color="steelblue",
        edgecolor="white",
    )
    ax.set_xlabel("Session duration (min, clipped at 180)")
    ax.set_ylabel("Session count")
    ax.set_title("Session duration distribution — pilot")
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "duration_hist.png", dpi=120)
    plt.close(fig)


def _plot_samples_vs_duration(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for cohort, sub in df.groupby("cohort"):
        ax.scatter(
            sub["duration_min"].clip(upper=600),
            sub["n_samples"],
            label=cohort,
            alpha=0.6,
            s=18,
        )
    ax.set_xlabel("Duration (min)")
    ax.set_ylabel("MeterValue sample count")
    ax.set_title("Samples vs duration by cohort")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "samples_vs_duration.png", dpi=120)
    plt.close(fig)


def _plot_sessions_per_charger(df: pd.DataFrame) -> None:
    counts = df.groupby("charger_id").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(counts)), counts.values, color="darkorange")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=40, ha="right")
    ax.set_ylabel("Session count")
    ax.set_title("Sessions per charger")
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "sessions_per_charger.png", dpi=120)
    plt.close(fig)


def _plot_arrival_hour_heatmap(df: pd.DataFrame) -> None:
    pivot = (
        df.groupby(["dayofweek", "hour"]).size().unstack(fill_value=0).reindex(range(7))
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xlabel("Hour of day (KST)")
    ax.set_ylabel("Day of week")
    ax.set_title("Arrival time heatmap")
    fig.colorbar(im, ax=ax, label="sessions")
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "arrival_heatmap.png", dpi=120)
    plt.close(fig)


def _plot_capacity_bound_share(df: pd.DataFrame) -> None:
    share = df.groupby("cohort")["capacity_bound_flag"].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(share.index, share.values, color="teal")
    ax.set_ylabel("capacity_bound_flag = True (%)")
    ax.set_title("Capacity-bound share by cohort")
    ax.set_ylim(0, 100)
    for i, v in enumerate(share.values):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center")
    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "capacity_bound_share.png", dpi=120)
    plt.close(fig)


def _write_markdown_report(df: pd.DataFrame, artifacts) -> None:
    now = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M KST")
    n_total = len(df)
    n_phase_a = int((df["cohort"] == "PhaseA").sum())
    n_phase_b = int((df["cohort"] == "PhaseB-fragments").sum())
    dur = df["duration_min"].describe()
    samp = df["n_samples"].describe()
    energy = df["energy_delivered_wh"].dropna()
    cap = df.groupby("cohort")["capacity_bound_flag"].mean() * 100

    lines: list[str] = []
    lines.append("# Phase B EDA — Pilot Report")
    lines.append("")
    lines.append(f"_Generated {now}_")
    lines.append("")
    lines.append("## 1. Scope Disclosure")
    lines.append("")
    lines.append("This pilot runs the session aggregation pipeline on all OCPP logs")
    lines.append("currently available in the repo. **The 9,000-session Phase B dataset")
    lines.append("referenced in HANDOFF v2 §6 does not yet exist locally** — it must be")
    lines.append("exported from the internal MongoDB collection. This report exercises")
    lines.append("the pipeline end-to-end so that, once the export lands, the same")
    lines.append("scripts run unchanged and will replace these numbers.")
    lines.append("")
    lines.append("### Data inventory used here")
    lines.append("")
    lines.append("| Source | Purpose | Nature |")
    lines.append("|---|---|---|")
    lines.append(
        "| `data/raw/260417-ST-22-98.json` | Phase A L7 full "
        "| complete 9h session, SoC present |"
    )
    lines.append(
        "| `data/raw/260420-ST-93-100.json` | Phase A L7 taper "
        "| 53 min SoC 93→99 zoom |"
    )
    lines.append(
        "| `data/processed/*.json` | Phase B fragments "
        "| 137 txn-split files from `random-cars-logs.json` "
        "(632 msgs, 2 chargers, Aug 2025–Mar 2026) |"
    )
    lines.append("")
    lines.append("## 2. Session-level summary")
    lines.append("")
    lines.append(f"- Total aggregated sessions: **{n_total}**")
    lines.append(f"- Phase A cohort: {n_phase_a}")
    lines.append(f"- Phase B fragments cohort: {n_phase_b}")
    lines.append("")
    lines.append("### Duration (minutes)")
    lines.append("")
    lines.append("| stat | value |")
    lines.append("|---|---|")
    for k in ("count", "mean", "std", "min", "25%", "50%", "75%", "max"):
        lines.append(f"| {k} | {dur[k]:.2f} |")
    lines.append("")
    lines.append("### Samples per session")
    lines.append("")
    lines.append("| stat | value |")
    lines.append("|---|---|")
    for k in ("count", "mean", "std", "min", "25%", "50%", "75%", "max"):
        lines.append(f"| {k} | {samp[k]:.2f} |")
    lines.append("")
    if not energy.empty:
        lines.append("### Energy delivered (Wh, non-null)")
        lines.append("")
        lines.append("| stat | value |")
        lines.append("|---|---|")
        desc = energy.describe()
        for k in ("count", "mean", "std", "min", "25%", "50%", "75%", "max"):
            lines.append(f"| {k} | {desc[k]:.1f} |")
        lines.append("")
    lines.append("### Capacity-bound share")
    lines.append("")
    for cohort, pct in cap.items():
        lines.append(f"- **{cohort}**: {pct:.1f}%")
    lines.append("")
    lines.append("### Sessions per charger")
    lines.append("")
    for charger, n in df.groupby("charger_id").size().sort_values(ascending=False).items():
        lines.append(f"- `{charger}`: {n}")
    lines.append("")
    lines.append("## 3. Data quality observations")
    lines.append("")
    lines.append("- **Phase B fragments are not real sessions.** Median 2 messages / 0 minutes")
    lines.append("  duration. These came from a 632-message sample file covering 2 chargers")
    lines.append("  over 7 months — not the 9,000-session station log. Expect distribution")
    lines.append("  plots here to be dominated by noise until the MongoDB export is available.")
    lines.append("- **No `current_offered_a` (PWM cap) is logged** in either Phase A or the")
    lines.append("  Phase B sample. `capacity_bound_flag` falls back to the configured")
    lines.append("  `i_pwm_assumed_a` (31.2 A = L7 default). For the Phase A L7 sessions this")
    lines.append("  is exactly the experimental setting; for Phase B it's a field-default")
    lines.append("  approximation that must be revisited per charger once available.")
    lines.append("- **Column-name variation confirmed**: `current_outlet_a`, `soc_ev_pct`,")
    lines.append("  3-phase voltage columns (`voltage_l1_v`..`l3_v`) appear in Phase B but")
    lines.append("  not in Phase A. The `normalize_measurement_columns` step coalesces these")
    lines.append("  to the canonical set (`current_a`, `soc_pct`, `voltage_v`).")
    lines.append("- **Energy monotonicity**: Phase A readings are strictly monotonic")
    lines.append("  accumulators. Several Phase B fragments show non-monotonic energy")
    lines.append("  (mixed streams under the same txn_id). `_energy_delivered_wh` falls back")
    lines.append("  to positive-diff sums for those cases.")
    lines.append("")
    lines.append("## 4. Plots")
    lines.append("")
    for name in (
        "duration_hist.png",
        "samples_vs_duration.png",
        "sessions_per_charger.png",
        "arrival_heatmap.png",
        "capacity_bound_share.png",
    ):
        lines.append(f"![{name}](../outputs/phase_b_eda/{name})")
        lines.append("")
    lines.append("## 5. Outputs")
    lines.append("")
    lines.append(f"- Raw session parquet: `{artifacts.raw_path.relative_to(ROOT)}`")
    lines.append(f"- ML-ready parquet: `{artifacts.ml_ready_path.relative_to(ROOT)}`")
    lines.append("- Plots: `outputs/phase_b_eda/*.png`")
    lines.append("")
    lines.append("## 6. Next action")
    lines.append("")
    lines.append("Export the production Phase B collection from MongoDB (filtered by")
    lines.append("station and time window per HANDOFF v2 §6) and re-run this script.")
    lines.append("No code change required — the pipeline is tolerant to column-name")
    lines.append("variation and fragments, and parquet schemas are stable.")
    lines.append("")

    (OUT_REPORT_DIR / "phase_b_eda.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report:       {OUT_REPORT_DIR / 'phase_b_eda.md'}")


if __name__ == "__main__":
    main()
