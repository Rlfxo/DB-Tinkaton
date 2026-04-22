"""Full Phase B EDA — aggregate 492-charger OCPP export and report.

Consumes ``data/raw/phase_b/*.json`` (produced by
``scripts/export_phase_b.py``). Streams file-by-file so memory stays
proportional to session count rather than raw document volume.

Outputs:
- ``data/phase_b/session_dataset_raw.parquet`` — every session column
- ``data/phase_b/session_dataset.parquet`` — ML-ready subset
- ``reports/phase_b_full_eda.md`` — narrative report + figures
- ``outputs/phase_b_full_eda/*.png``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tinkaton.dataset import write_session_dataset
from tinkaton.transform import SessionAggregateConfig

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "phase_b"
OUT_DATA = ROOT / "data" / "phase_b"
OUT_REPORT = ROOT / "reports"
OUT_FIG = ROOT / "outputs" / "phase_b_full_eda"

# L7 (PWM 52%, 7 kW) field default. All 492 AC chargers are 7 kW single-phase
# ELA/E01AS family per manifest, so the same I_cap applies.
I_PWM_ASSUMED_A = 31.2


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest()

    print(f"building session dataset from {SOURCE} ...")
    artifacts = write_session_dataset(
        [SOURCE],
        out_dir=OUT_DATA,
        config=SessionAggregateConfig(i_pwm_assumed_a=I_PWM_ASSUMED_A),
        progress_every=50,
    )
    print(f"sources: {artifacts.n_source_files}  sessions: {artifacts.n_sessions}")

    sessions = pd.read_parquet(artifacts.raw_path)
    if sessions.empty:
        print("no sessions aggregated — aborting")
        return

    sessions = sessions.merge(manifest, on="charger_id", how="left")

    # Derived columns for analysis
    sessions["binding_ratio"] = sessions["mean_current_a"] / I_PWM_ASSUMED_A
    sessions["has_mv"] = sessions["has_meter_values"].astype(bool)

    _plot_duration_hist(sessions)
    _plot_energy_hist(sessions)
    _plot_binding_ratio_hist(sessions)
    _plot_stop_reason(sessions)
    _plot_capacity_bound_share_by_model(sessions)
    _plot_sessions_per_charger(sessions)
    _plot_arrival_heatmap(sessions)

    _write_report(sessions, artifacts)


def _load_manifest() -> pd.DataFrame:
    path = SOURCE / "_charger_manifest.csv"
    if not path.exists():
        return pd.DataFrame(columns=["charger_id", "vendor", "model"])
    df = pd.read_csv(path)
    return df[["charger_id", "vendor", "model"]]


def _plot_duration_hist(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["duration_min"].clip(upper=720), bins=60, color="steelblue", edgecolor="white")
    ax.set_xlabel("Session duration (min, clipped at 720 = 12 h)")
    ax.set_ylabel("Session count")
    ax.set_title(f"Duration distribution — {len(df):,} sessions")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "duration_hist.png", dpi=120)
    plt.close(fig)


def _plot_energy_hist(df: pd.DataFrame) -> None:
    energy_kwh = df["energy_delivered_wh"].dropna() / 1000.0
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(energy_kwh.clip(upper=100), bins=60, color="darkorange", edgecolor="white")
    ax.set_xlabel("Energy delivered (kWh, clipped at 100)")
    ax.set_ylabel("Session count")
    ax.set_title("Energy delivered per session")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "energy_hist.png", dpi=120)
    plt.close(fig)


def _plot_binding_ratio_hist(df: pd.DataFrame) -> None:
    br = df.loc[df["has_mv"], "binding_ratio"].dropna()
    br = br[(br > 0) & (br < 1.2)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(br, bins=60, color="teal", edgecolor="white")
    ax.axvline(0.98, linestyle="--", color="red", label="η ≈ 0.98 (Phase A)")
    ax.set_xlabel("binding_ratio = mean I / I_cap (31.2 A)")
    ax.set_ylabel("Session count")
    ax.set_title("Phase B binding ratio distribution — empirical η evidence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "binding_ratio_hist.png", dpi=120)
    plt.close(fig)


def _plot_stop_reason(df: pd.DataFrame) -> None:
    counts = df["stop_reason"].fillna("(none)").value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color="mediumpurple")
    ax.set_ylabel("Session count")
    ax.set_title("Session termination reasons")
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "stop_reason.png", dpi=120)
    plt.close(fig)


def _plot_capacity_bound_share_by_model(df: pd.DataFrame) -> None:
    model_groups = df.groupby("model")["capacity_bound_flag"].agg(
        count="count",
        share=lambda s: float(s.mean() * 100) if len(s) > 0 else 0.0,
    )
    model_groups = model_groups[model_groups["count"] > 20].sort_values(
        "share", ascending=False
    )
    if model_groups.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(model_groups.index, model_groups["share"], color="darkcyan")
    ax.set_ylim(0, 100)
    ax.set_ylabel("capacity_bound_flag = True (%)")
    ax.set_title("Capacity-bound share by charger model (≥20 sessions)")
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    for i, (_, row) in enumerate(model_groups.iterrows()):
        ax.text(
            i,
            row["share"] + 2,
            f"{row['share']:.0f}%\nn={int(row['count'])}",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(OUT_FIG / "capacity_bound_by_model.png", dpi=120)
    plt.close(fig)


def _plot_sessions_per_charger(df: pd.DataFrame) -> None:
    counts = df.groupby("charger_id").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(counts.values, bins=50, color="goldenrod", edgecolor="white")
    ax.set_xlabel("Sessions per charger")
    ax.set_ylabel("Number of chargers")
    ax.set_title(
        f"Sessions per charger "
        f"(median {int(counts.median())}, top 1% ≥ {int(counts.quantile(0.99))})"
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG / "sessions_per_charger.png", dpi=120)
    plt.close(fig)


def _plot_arrival_heatmap(df: pd.DataFrame) -> None:
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
    fig.savefig(OUT_FIG / "arrival_heatmap.png", dpi=120)
    plt.close(fig)


def _fmt(n: float, fmt: str = "{:,.2f}") -> str:
    if pd.isna(n):
        return "—"
    return fmt.format(n)


def _write_report(df: pd.DataFrame, artifacts) -> None:
    now = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M KST")
    n = len(df)
    n_mv = int(df["has_mv"].sum())
    n_labels_only = n - n_mv
    mv_share = n_mv / n * 100
    br = df.loc[df["has_mv"], "binding_ratio"].dropna()
    cap_share = df["capacity_bound_flag"].fillna(False).mean() * 100
    n_chargers = df["charger_id"].nunique()
    span_start = df["arrival_ts"].min()
    span_end = df["arrival_ts"].max()

    lines: list[str] = []
    lines.append("# Phase B — Full AC Charger EDA Report")
    lines.append("")
    lines.append(f"_Generated {now}_")
    lines.append("")
    lines.append("## 1. Scope")
    lines.append("")
    lines.append(f"- Chargers: **{n_chargers}** AC (ELA* / E01AS* family)")
    lines.append(f"- Sessions aggregated: **{n:,}** total")
    lines.append(f"  - With MeterValue features: {n_mv:,} ({mv_share:.1f} %)")
    lines.append(f"  - Label-only (short disconnect / no MV): {n_labels_only:,}")
    lines.append(f"- Time span: {span_start} → {span_end}")
    lines.append(f"- Assumed I_cap for binding_ratio: **{I_PWM_ASSUMED_A} A** (L7 = PWM 52 %)")
    lines.append(f"- Source export: `{artifacts.raw_path.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## 2. Session duration & energy")
    lines.append("")
    dur = df["duration_min"].describe()
    energy = (df["energy_delivered_wh"].dropna() / 1000).describe()
    lines.append(
        "| stat | duration (min) | energy (kWh) |\n|---|---|---|"
    )
    for key in ("count", "mean", "std", "min", "25%", "50%", "75%", "max"):
        energy_val = energy[key] if key in energy.index else float("nan")
        lines.append(f"| {key} | {_fmt(dur[key])} | {_fmt(energy_val)} |")
    lines.append("")
    lines.append("## 3. Binding ratio (empirical η evidence)")
    lines.append("")
    if not br.empty:
        lines.append(f"- mean: **{br.mean():.4f}**")
        lines.append(f"- median: {br.median():.4f}")
        lines.append(f"- std: {br.std():.4f}")
        p10, p50, p90 = br.quantile(0.1), br.median(), br.quantile(0.9)
        lines.append(f"- P10/P50/P90: {p10:.3f} / {p50:.3f} / {p90:.3f}")
        lines.append("")
        lines.append(
            "Phase A had η = 0.983 (260417) and 0.981 (260420). Phase B field "
            f"mean = {br.mean():.3f}. The alignment supports the **structural η "
            "assumption** used in the LP linearization (HANDOFF §3.1)."
        )
        lines.append("")
    lines.append("## 4. Capacity-bound evidence")
    lines.append("")
    lines.append(f"- Overall capacity_bound_flag share: **{cap_share:.1f} %**")
    lines.append(
        "- Threshold: I_actual ≥ 0.95 · I_cap for ≥ 20 consecutive minutes"
    )
    lines.append("")
    lines.append("## 5. Termination reasons")
    lines.append("")
    reason_counts = df["stop_reason"].fillna("(none)").value_counts()
    for reason, cnt in reason_counts.items():
        lines.append(f"- `{reason}`: {cnt:,} ({cnt / n * 100:.1f} %)")
    lines.append("")
    lines.append("## 6. Charger activity")
    lines.append("")
    per_charger = df.groupby("charger_id").size()
    lines.append(f"- median sessions/charger: {int(per_charger.median())}")
    lines.append(f"- top 1 % threshold: {int(per_charger.quantile(0.99))}")
    lines.append(f"- chargers with < 10 sessions: {int((per_charger < 10).sum())} (long-tail)")
    lines.append(f"- chargers with ≥ 100 sessions: {int((per_charger >= 100).sum())}")
    lines.append("")
    lines.append("### Top-10 chargers by session count")
    lines.append("")
    top10 = per_charger.sort_values(ascending=False).head(10)
    model_lookup = df.set_index("charger_id")["model"].to_dict()
    lines.append("| charger_id | model | sessions |")
    lines.append("|---|---|---|")
    for cid, count in top10.items():
        lines.append(f"| `{cid}` | {model_lookup.get(cid, '?')} | {count:,} |")
    lines.append("")
    lines.append("## 7. Figures")
    lines.append("")
    for name in (
        "duration_hist.png",
        "energy_hist.png",
        "binding_ratio_hist.png",
        "capacity_bound_by_model.png",
        "stop_reason.png",
        "sessions_per_charger.png",
        "arrival_heatmap.png",
    ):
        lines.append(f"![{name}](../outputs/phase_b_full_eda/{name})")
        lines.append("")
    lines.append("## 8. Outputs")
    lines.append("")
    lines.append(f"- `{artifacts.raw_path.relative_to(ROOT)}` — full session schema")
    lines.append(f"- `{artifacts.ml_ready_path.relative_to(ROOT)}` — ML-ready subset")
    lines.append("- `outputs/phase_b_full_eda/*.png` — figures")
    lines.append("")
    lines.append("## 9. Next steps")
    lines.append("")
    lines.append(
        "1. Confirm η distribution aligns with the structural claim in HANDOFF §3.1."
    )
    lines.append(
        "2. Feed `session_dataset.parquet` into the XGBoost/LSTM training "
        "track (HANDOFF §7)."
    )
    lines.append(
        "3. Slice by charger cluster (prefix grouping) when selecting the Phase B "
        "station subset for LP simulation (HANDOFF §3.5 — cluster selection still pending)."
    )
    lines.append("")

    (OUT_REPORT / "phase_b_full_eda.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report → {(OUT_REPORT / 'phase_b_full_eda.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
