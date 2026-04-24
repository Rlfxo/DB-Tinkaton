"""Compare XGBoost and LSTM baselines on the held-out test set.

Consumes both residual parquets plus the session dataset for duration
buckets. Emits the two figures NOTES_to_DB_session_v1.md §P0-4 requests
for Ch.4 Figure 10 material:

- ``outputs/model_compare/mae_by_hour.png`` — MAE by arrival hour
- ``outputs/model_compare/mae_by_duration_bucket.png`` — MAE by duration
  bucket (Ch.3 buckets: <5m, 5–15m, 15–60m, 1–3h, 3–6h, 6–12h, >12h)

A tiny comparison JSON is also written for the paper session to pull
the headline numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XGB_RESID = ROOT / "results" / "xgb_residuals.parquet"
LSTM_RESID = ROOT / "results" / "lstm_residuals.parquet"
OUT_DIR = ROOT / "outputs" / "model_compare"
SUMMARY_JSON = ROOT / "results" / "model_compare_summary.json"

DURATION_BUCKETS = [0, 5, 15, 60, 180, 360, 720, 10_000]
DURATION_LABELS = ["<5m", "5–15m", "15–60m", "1–3h", "3–6h", "6–12h", ">12h"]


def _load(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["model"] = model_name
    df["abs_err"] = np.abs(df["y_pred"] - df["y_true"])
    df["arrival_ts"] = pd.to_datetime(df["arrival_ts"], utc=True)
    df["hour_kst"] = df["arrival_ts"].dt.tz_convert("Asia/Seoul").dt.hour
    return df


def _mae_by_group(df: pd.DataFrame, group_col: str) -> pd.Series:
    return df.groupby(group_col)["abs_err"].mean()


def _plot_mae_by_hour(xgb: pd.DataFrame, lstm: pd.DataFrame, out_path: Path) -> None:
    xgb_mae = _mae_by_group(xgb, "hour_kst").reindex(range(24))
    lstm_mae = _mae_by_group(lstm, "hour_kst").reindex(range(24))
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.4
    idx = np.arange(24)
    ax.bar(idx - width / 2, xgb_mae.values, width, label="XGBoost", color="steelblue")
    ax.bar(idx + width / 2, lstm_mae.values, width, label="LSTM", color="darkorange")
    ax.set_xticks(idx)
    ax.set_xlabel("Arrival hour (KST)")
    ax.set_ylabel("Test MAE (min)")
    ax.set_title("Test MAE by hour-of-day — XGBoost vs LSTM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_mae_by_duration(xgb: pd.DataFrame, lstm: pd.DataFrame, out_path: Path) -> None:
    for df in (xgb, lstm):
        df["duration_bucket"] = pd.cut(
            df["y_true"], bins=DURATION_BUCKETS, labels=DURATION_LABELS, right=False
        )
    xgb_mae = xgb.groupby("duration_bucket", observed=True)["abs_err"].mean()
    lstm_mae = lstm.groupby("duration_bucket", observed=True)["abs_err"].mean()
    counts = xgb.groupby("duration_bucket", observed=True).size()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    idx = np.arange(len(DURATION_LABELS))
    width = 0.4
    xgb_vals = xgb_mae.reindex(DURATION_LABELS).values
    lstm_vals = lstm_mae.reindex(DURATION_LABELS).values
    ax.bar(idx - width / 2, xgb_vals, width, label="XGBoost", color="steelblue")
    ax.bar(idx + width / 2, lstm_vals, width, label="LSTM", color="darkorange")
    ax.set_xticks(idx)
    ax.set_xticklabels(DURATION_LABELS)
    ax.set_xlabel("Actual session duration bucket")
    ax.set_ylabel("Test MAE (min)")
    ax.set_title("Test MAE by duration bucket — XGBoost vs LSTM")
    # Annotate sample count per bucket
    for i, label in enumerate(DURATION_LABELS):
        n = int(counts.get(label, 0))
        ax.text(i, 5, f"n={n}", ha="center", fontsize=8, color="gray")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xgb = _load(XGB_RESID, "XGBoost")
    lstm = _load(LSTM_RESID, "LSTM")

    # Safety: both should cover the same test set
    n_diff = len(set(map(tuple, xgb[["charger_id", "arrival_ts"]].to_numpy()))
                 - set(map(tuple, lstm[["charger_id", "arrival_ts"]].to_numpy())))
    if n_diff:
        print(f"  warning: {n_diff} XGBoost rows not in LSTM residuals")

    _plot_mae_by_hour(xgb, lstm, OUT_DIR / "mae_by_hour.png")
    _plot_mae_by_duration(xgb, lstm, OUT_DIR / "mae_by_duration_bucket.png")

    summary = {
        "xgboost": {
            "n": int(len(xgb)),
            "mae_min": float(xgb["abs_err"].mean()),
            "median_ae_min": float(xgb["abs_err"].median()),
            "bias_min": float((xgb["y_pred"] - xgb["y_true"]).mean()),
            "pct_within_15min": float((xgb["abs_err"] <= 15).mean() * 100),
            "pct_within_30min": float((xgb["abs_err"] <= 30).mean() * 100),
            "pct_within_60min": float((xgb["abs_err"] <= 60).mean() * 100),
        },
        "lstm": {
            "n": int(len(lstm)),
            "mae_min": float(lstm["abs_err"].mean()),
            "median_ae_min": float(lstm["abs_err"].median()),
            "bias_min": float((lstm["y_pred"] - lstm["y_true"]).mean()),
            "pct_within_15min": float((lstm["abs_err"] <= 15).mean() * 100),
            "pct_within_30min": float((lstm["abs_err"] <= 30).mean() * 100),
            "pct_within_60min": float((lstm["abs_err"] <= 60).mean() * 100),
        },
        "xgb_minus_lstm_mae_pct": 100.0 * (
            xgb["abs_err"].mean() - lstm["abs_err"].mean()
        ) / lstm["abs_err"].mean(),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"XGBoost MAE: {summary['xgboost']['mae_min']:.2f} min  |  "
        f"LSTM MAE: {summary['lstm']['mae_min']:.2f} min"
    )
    rel = summary["xgb_minus_lstm_mae_pct"]
    direction = "worse" if rel > 0 else "better"
    print(f"XGBoost is {abs(rel):.1f}% {direction} than LSTM on test MAE")
    print(f"wrote {OUT_DIR.relative_to(ROOT)}/*.png and {SUMMARY_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
