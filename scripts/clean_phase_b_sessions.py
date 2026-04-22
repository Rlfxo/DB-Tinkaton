"""Apply quality filters to the aggregated Phase B session dataset.

Reads ``data/phase_b/session_dataset_raw.parquet`` (produced by
``scripts/run_phase_b_full_eda.py``), applies the
:class:`tinkaton.cleaner.SessionCleanConfig` defaults, and writes:

- ``data/phase_b/session_dataset_clean.parquet`` — rows that pass
- ``data/phase_b/session_dataset_rejected.csv`` — rows that fail, with
  ``rejection_reason``
- ``reports/phase_b_clean_summary.md`` — side-by-side raw vs clean
  comparison (counts, binding ratio, capacity-bound share)
- ``outputs/phase_b_full_eda/binding_ratio_raw_vs_clean.png`` — visual
  evidence for the thesis Ch.3 figure
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tinkaton.cleaner import SessionCleanConfig, clean_sessions

ROOT = Path(__file__).resolve().parents[1]
RAW_PARQUET = ROOT / "data" / "phase_b" / "session_dataset_raw.parquet"
CLEAN_PARQUET = ROOT / "data" / "phase_b" / "session_dataset_clean.parquet"
REJECTED_CSV = ROOT / "data" / "phase_b" / "session_dataset_rejected.csv"
REPORT_PATH = ROOT / "reports" / "phase_b_clean_summary.md"
FIG_DIR = ROOT / "outputs" / "phase_b_full_eda"

I_PWM_ASSUMED_A = 31.2


def main() -> None:
    if not RAW_PARQUET.exists():
        raise SystemExit(
            f"Missing {RAW_PARQUET}. Run scripts/run_phase_b_full_eda.py first."
        )

    raw = pd.read_parquet(RAW_PARQUET)
    print(f"raw sessions: {len(raw):,}")

    cfg = SessionCleanConfig()
    result = clean_sessions(raw, config=cfg)

    print(f"kept:     {len(result.clean):,}")
    print(f"rejected: {len(result.rejected):,}")
    print("rejection breakdown:")
    for key, value in result.summary.items():
        if key.startswith("reject:"):
            reason = key.split(":", 1)[1]
            print(f"  - {reason}: {value:,}")

    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    result.clean.to_parquet(CLEAN_PARQUET, index=False)
    print(f"wrote {CLEAN_PARQUET.relative_to(ROOT)}")

    rejected_cols = [
        "charger_id",
        "transaction_id",
        "arrival_ts",
        "plug_out_ts",
        "duration_min",
        "energy_delivered_wh",
        "stop_reason",
        "rejection_reason",
    ]
    rejected_out = result.rejected[[c for c in rejected_cols if c in result.rejected.columns]]
    rejected_out.to_csv(REJECTED_CSV, index=False)
    print(f"wrote {REJECTED_CSV.relative_to(ROOT)}")

    _plot_binding_comparison(raw, result.clean)
    _write_report(raw, result)


def _plot_binding_comparison(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw_mask = raw["has_meter_values"].astype(bool)
    clean_mask = clean["has_meter_values"].astype(bool)
    raw_br = (raw.loc[raw_mask, "mean_current_a"] / I_PWM_ASSUMED_A).dropna()
    clean_br = (clean.loc[clean_mask, "mean_current_a"] / I_PWM_ASSUMED_A).dropna()
    raw_br = raw_br[(raw_br > 0) & (raw_br < 1.2)]
    clean_br = clean_br[(clean_br > 0) & (clean_br < 1.2)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = 80
    ax.hist(
        raw_br,
        bins=bins,
        color="lightgray",
        edgecolor="white",
        label=f"raw  (n={len(raw_br):,})",
    )
    ax.hist(
        clean_br,
        bins=bins,
        color="teal",
        alpha=0.75,
        edgecolor="white",
        label=f"clean (n={len(clean_br):,})",
    )
    ax.axvline(0.98, linestyle="--", color="red", label="Phase A η ≈ 0.98")
    ax.set_xlabel("binding_ratio = mean I / I_cap (31.2 A)")
    ax.set_ylabel("Session count")
    ax.set_title("Phase B binding ratio — raw vs clean")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "binding_ratio_raw_vs_clean.png", dpi=120)
    plt.close(fig)
    print(f"wrote {(FIG_DIR / 'binding_ratio_raw_vs_clean.png').relative_to(ROOT)}")


def _binding_stats(df: pd.DataFrame) -> dict[str, float]:
    series = df.loc[df["has_meter_values"].astype(bool), "mean_current_a"] / I_PWM_ASSUMED_A
    series = series.dropna()
    series = series[(series > 0) & (series < 1.2)]
    if series.empty:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "n": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "p10": float(series.quantile(0.1)),
        "p90": float(series.quantile(0.9)),
    }


def _write_report(raw: pd.DataFrame, result) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean = result.clean
    rejected = result.rejected

    raw_stats = _binding_stats(raw)
    clean_stats = _binding_stats(clean)
    raw_cap = raw["capacity_bound_flag"].fillna(False).astype(bool).mean() * 100
    clean_cap = clean["capacity_bound_flag"].fillna(False).astype(bool).mean() * 100

    lines: list[str] = []
    lines.append("# Phase B Clean vs Raw Comparison")
    lines.append("")
    lines.append(f"_Generated {pd.Timestamp.now(tz='Asia/Seoul'):%Y-%m-%d %H:%M %Z}_")
    lines.append("")
    lines.append("## Filter configuration")
    lines.append("")
    lines.append("```")
    lines.extend(SessionCleanConfig().describe())
    lines.append("```")
    lines.append("")
    lines.append("## Row-level outcome")
    lines.append("")
    lines.append("| split | sessions |")
    lines.append("|---|---|")
    lines.append(f"| raw input | {len(raw):,} |")
    lines.append(f"| kept (clean) | {len(clean):,} |")
    lines.append(f"| rejected | {len(rejected):,} ({len(rejected)/len(raw)*100:.1f} %) |")
    lines.append("")
    lines.append("### Rejection breakdown")
    lines.append("")
    for reason, count in rejected["rejection_reason"].value_counts().items():
        lines.append(f"- `{reason}`: {count:,}")
    lines.append("")
    lines.append("## Binding ratio shift (thesis Ch.3 evidence)")
    lines.append("")
    lines.append("| stat | raw | clean |")
    lines.append("|---|---|---|")
    for key in ("n", "mean", "median", "std", "p10", "p90"):
        raw_val = raw_stats[key]
        clean_val = clean_stats[key]
        lines.append(f"| {key} | {_fmt(raw_val)} | {_fmt(clean_val)} |")
    lines.append("")
    lines.append(
        "The clean distribution is the honest field estimate of η. Compare "
        "against Phase A field defaults (η = 0.983 / 0.981 from sessions "
        "`260417` and `260420`)."
    )
    lines.append("")
    lines.append("## Capacity-bound share")
    lines.append("")
    lines.append(f"- raw: {raw_cap:.1f} %")
    lines.append(f"- clean: {clean_cap:.1f} %")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- `{CLEAN_PARQUET.relative_to(ROOT)}`")
    lines.append(f"- `{REJECTED_CSV.relative_to(ROOT)}`")
    lines.append("- `outputs/phase_b_full_eda/binding_ratio_raw_vs_clean.png`")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_PATH.relative_to(ROOT)}")


def _fmt(v: float) -> str:
    if v != v:  # NaN
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.4f}"


if __name__ == "__main__":
    main()
