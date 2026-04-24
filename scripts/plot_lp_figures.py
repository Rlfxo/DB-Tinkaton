"""Post-process LP sweep CSVs into the signature figures.

Consumes:
- ``results/lp_sweep/beta_sweep.csv`` (Figure 14 signature — β × N curve)
- ``results/lp_sweep/p_contract_sensitivity.csv`` (Ch.7 sub-figure)

Produces:
- ``outputs/lp_sweep/beta_curve.png`` — peak reduction vs β for each N
- ``outputs/lp_sweep/peak_by_strategy.png`` — strategy comparison bar
- ``outputs/lp_sweep/p_contract_sensitivity.png`` — regime boundary
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWEEP_CSV = ROOT / "results" / "lp_sweep" / "beta_sweep.csv"
P_CONTRACT_CSV = ROOT / "results" / "lp_sweep" / "p_contract_sensitivity.csv"
FIG_DIR = ROOT / "outputs" / "lp_sweep"


def _load_beta_sweep() -> pd.DataFrame:
    if not SWEEP_CSV.exists():
        raise SystemExit(f"Missing {SWEEP_CSV}. Run scripts/run_lp_sweep.py first.")
    return pd.read_csv(SWEEP_CSV)


def _compute_peak_reduction(df: pd.DataFrame) -> pd.DataFrame:
    """Attach % peak reduction vs the StatusQuo reference per (N, seed)."""
    sq = (
        df[df["strategy"] == "StatusQuo"]
        .groupby(["n_chargers", "seed"])["peak_mean_kw"]
        .first()
    )
    # Use median-across-seeds StatusQuo as the robust reference per N
    sq_ref = sq.groupby(level="n_chargers").median().rename("sq_reference_kw")
    merged = df.merge(sq_ref, left_on="n_chargers", right_index=True)
    merged["peak_reduction_pct"] = (
        (merged["sq_reference_kw"] - merged["peak_mean_kw"])
        / merged["sq_reference_kw"]
        * 100.0
    )
    return merged


def plot_beta_curve(df: pd.DataFrame, out_path: Path) -> None:
    """β vs peak-reduction % with IQR band, one line per N."""
    aug = _compute_peak_reduction(df)
    hybrid = aug[aug["strategy"] == "LP-Hybrid"]
    if hybrid.empty:
        print("skipping beta_curve — no LP-Hybrid rows")
        return
    agg = hybrid.groupby(["n_chargers", "beta"])["peak_reduction_pct"].agg(
        median="median",
        lo=lambda s: np.percentile(s, 25),
        hi=lambda s: np.percentile(s, 75),
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = plt.cm.viridis(np.linspace(0.1, 0.85, agg.index.get_level_values(0).nunique()))
    for color, (n, sub) in zip(palette, agg.groupby(level="n_chargers"), strict=True):
        sub = sub.droplevel("n_chargers")
        ax.plot(sub.index, sub["median"], marker="o", label=f"N = {n}", color=color, linewidth=2)
        ax.fill_between(sub.index, sub["lo"], sub["hi"], color=color, alpha=0.15)
    ax.set_xlabel("β (user adoption fraction)")
    ax.set_ylabel("Peak reduction vs StatusQuo (%)")
    ax.set_title("LP peak reduction by β × N (signature curve)")
    ax.legend(title="Station subsample")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(ROOT)}")


def plot_strategy_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of median peak across strategies at the default N."""
    max_n = df["n_chargers"].max()
    sub = df[df["n_chargers"] == max_n]
    # For LP-Hybrid use β = 0.0 as the representative hybrid (≈ LP-ML)
    sub = sub[~((sub["strategy"] == "LP-Hybrid") & (sub["beta"] != 0.0))]
    order = ["StatusQuo", "LoadBalance", "LP-ML", "LP-Hybrid", "LP-User"]
    sub = sub[sub["strategy"].isin(order)]
    stats = sub.groupby("strategy")["peak_mean_kw"].agg(["median", "min", "max"])
    stats = stats.reindex(order).dropna()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(stats.index, stats["median"], color="steelblue", edgecolor="white")
    for i, (label, row) in enumerate(stats.iterrows()):
        ax.text(
            i,
            row["median"] + 0.5,
            f"{row['median']:.2f}",
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Median peak (kW, test period mean-per-day)")
    ax.set_title(f"Strategy comparison at N = {max_n} (P_contract = 175 kW)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(ROOT)}")


def plot_p_contract_sensitivity(out_path: Path) -> None:
    if not P_CONTRACT_CSV.exists():
        print(f"skipping p_contract_sensitivity — {P_CONTRACT_CSV} not found")
        return
    df = pd.read_csv(P_CONTRACT_CSV)

    fig, (ax_peak, ax_inf) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Upper plot: peak per strategy over P_contract
    for strategy, color in [
        ("StatusQuo", "gray"),
        ("LoadBalance", "steelblue"),
        ("LP-Hybrid", "darkorange"),
    ]:
        sub = df[df["strategy"] == strategy]
        if sub.empty:
            continue
        if strategy == "LP-Hybrid":
            # average across β since the finding is β has no effect here
            agg = sub.groupby("p_contract_kw")["peak_mean_kw"].median()
        else:
            agg = sub.groupby("p_contract_kw")["peak_mean_kw"].median()
        ax_peak.plot(agg.index, agg.values, marker="o", label=strategy, color=color, linewidth=2)

    # Overlay β sensitivity spread for LP-Hybrid
    hy = df[df["strategy"] == "LP-Hybrid"]
    if not hy.empty:
        spread = hy.groupby("p_contract_kw")["peak_mean_kw"].agg(
            lo=lambda s: np.percentile(s, 25),
            hi=lambda s: np.percentile(s, 75),
        )
        ax_peak.fill_between(
            spread.index, spread["lo"], spread["hi"],
            alpha=0.18, color="darkorange", label="LP-Hybrid β IQR",
        )

    ax_peak.set_ylabel("Median peak (kW)")
    ax_peak.set_title("Peak by strategy across P_contract (N = 25 design scope)")
    ax_peak.legend()
    ax_peak.grid(True, alpha=0.3)

    # Lower plot: infeasibility
    for strategy, color in [
        ("LP-Hybrid", "darkorange"),
    ]:
        sub = df[df["strategy"] == strategy]
        if sub.empty:
            continue
        agg = sub.groupby("p_contract_kw")["infeasible_mean"].mean() * 100
        ax_inf.plot(agg.index, agg.values, marker="s", color=color, linewidth=2,
                    label=f"{strategy} infeasibility")
    ax_inf.axhline(10, color="red", linestyle="--", alpha=0.5, label="10 % threshold")
    ax_inf.set_ylabel("Infeasible day share (%)")
    ax_inf.set_xlabel("P_contract (kW)")
    ax_inf.legend()
    ax_inf.grid(True, alpha=0.3)

    fig.suptitle(
        "P_contract sensitivity: β-framework effect concentrates in the tight regime"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_beta_sweep()
    plot_beta_curve(df, FIG_DIR / "beta_curve.png")
    plot_strategy_comparison(df, FIG_DIR / "peak_by_strategy.png")
    plot_p_contract_sensitivity(FIG_DIR / "p_contract_sensitivity.png")


if __name__ == "__main__":
    main()
