# ruff: noqa: N803, N806
"""XGBoost hyperparameter search for departure-time prediction.

- Uses ``session_dataset_clean_v2.parquet`` + ``split_definition.json``.
- Optuna TPE with 30 trials (configurable) on the train split, 5-fold
  walk-forward CV for each trial. Scoring = MAE (minutes).
- Refits the best configuration on train+val and reports val / test
  metrics alongside residuals for the LP simulator.

Outputs (under ``models/``, ``results/``, ``outputs/``):
- ``models/xgb_hpo.pkl`` — best XGBRegressor fit on train+val
- ``results/hpo_trials.csv`` — full Optuna trial history
- ``results/xgb_metrics.json`` — val + test metrics
- ``results/xgb_residuals.parquet`` — per-test-row residuals for LP
- ``outputs/xgb/pred_vs_actual.png``
- ``outputs/xgb/feat_importance.png``
- ``outputs/xgb/hpo_importance.png``
- ``outputs/xgb/residual_distribution.png``
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from tinkaton.ml_features import FeatureBuildConfig, build_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
SESSIONS_PARQUET = ROOT / "data" / "phase_b" / "session_dataset_clean_v2.parquet"
SPLIT_JSON = ROOT / "data" / "phase_b" / "split_definition.json"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "outputs" / "xgb"

RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument(
        "--horizon",
        choices=("plug_in", "plus_10min"),
        default="plus_10min",
    )
    p.add_argument(
        "--cap-duration-min",
        type=float,
        default=2000.0,
        help="Clip target duration at this many minutes; 2000 ≈ 33h covers normal sessions.",
    )
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "n": int(len(y_true)),
        "mae_min": float(abs_err.mean()),
        "rmse_min": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_ae_min": float(np.median(abs_err)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "pct_within_15min": float((abs_err <= 15).mean() * 100),
        "pct_within_30min": float((abs_err <= 30).mean() * 100),
        "pct_within_60min": float((abs_err <= 60).mean() * 100),
        "bias_min": float(err.mean()),
    }


def _fit_model(params: dict, X_train, y_train, X_val, y_val) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        objective="reg:absoluteerror",
        tree_method="hist",
        enable_categorical=True,
        random_state=RANDOM_SEED,
        eval_metric="mae",
        early_stopping_rounds=30,
        n_jobs=-1,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _objective_factory(X_train, y_train, cv_splits: int):
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    fold_indices = list(tscv.split(X_train))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        }
        fold_maes: list[float] = []
        for tr_idx, va_idx in fold_indices:
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            model = _fit_model(params, X_tr, y_tr, X_va, y_va)
            pred = model.predict(X_va)
            fold_maes.append(mean_absolute_error(y_va, pred))
        return float(np.mean(fold_maes))

    return objective


def _plot_pred_vs_actual(y_true, y_pred, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=5, alpha=0.3)
    lim = max(y_true.max(), y_pred.max(), 1.0)
    ax.plot([0, lim], [0, lim], color="red", linestyle="--", label="y = ŷ")
    ax.set_xlabel("Actual duration (min)")
    ax.set_ylabel("Predicted duration (min)")
    ax.set_title("Test set: predicted vs actual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_feat_importance(
    model: xgb.XGBRegressor, feature_names: list[str], out_path: Path
) -> None:
    importance = model.feature_importances_
    order = np.argsort(importance)[-20:]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(order))))
    ax.barh(
        [feature_names[i] for i in order],
        importance[order],
        color="steelblue",
    )
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Top feature importances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_residuals(residuals: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    clipped = np.clip(residuals, -500, 500)
    ax.hist(clipped, bins=80, color="teal", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual = ŷ − y (min, clipped at ±500)")
    ax.set_ylabel("Test session count")
    ax.set_title(
        f"Test residuals   "
        f"mean={residuals.mean():.1f}   median={np.median(residuals):.1f}   "
        f"std={residuals.std():.1f}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_hpo_importance(study: optuna.Study, out_path: Path) -> None:
    try:
        importance = optuna.importance.get_param_importances(study)
    except (RuntimeError, ValueError):
        return  # too few trials or constant params
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(importance.keys())[::-1], list(importance.values())[::-1], color="goldenrod")
    ax.set_xlabel("Hyperparameter importance (fanova)")
    ax.set_title("Optuna HPO importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {SESSIONS_PARQUET.relative_to(ROOT)} ...")
    sessions = pd.read_parquet(SESSIONS_PARQUET)
    fm = build_feature_matrix(
        sessions,
        split_definition_path=SPLIT_JSON,
        config=FeatureBuildConfig(
            horizon=args.horizon, cap_duration_min=args.cap_duration_min
        ),
    )
    print(f"horizon={fm.horizon}  n_features={len(fm.feature_names)}")

    is_train = fm.split == "train"
    is_val = fm.split == "val"
    is_test = fm.split == "test"
    X_train, y_train = fm.X[is_train].reset_index(drop=True), fm.y[is_train].reset_index(drop=True)
    X_val, y_val = fm.X[is_val].reset_index(drop=True), fm.y[is_val].reset_index(drop=True)
    X_test, y_test = fm.X[is_test].reset_index(drop=True), fm.y[is_test].reset_index(drop=True)
    print(f"split sizes — train={len(y_train):,}  val={len(y_val):,}  test={len(y_test):,}")

    print(f"running Optuna HPO: {args.n_trials} trials x {args.cv_splits}-fold walk-forward ...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(
            _objective_factory(X_train, y_train, args.cv_splits),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

    print(f"best trial:  MAE={study.best_value:.2f}  params={study.best_params}")

    # Refit best config on train+val with early stop via internal val slice
    combined_X = pd.concat([X_train, X_val], ignore_index=True)
    combined_y = pd.concat([y_train, y_val], ignore_index=True)
    # Use the last 10 % of the combined series as an internal eval set for early stop
    n_combined = len(combined_X)
    eval_split = int(n_combined * 0.9)
    X_fit, y_fit = combined_X.iloc[:eval_split], combined_y.iloc[:eval_split]
    X_es, y_es = combined_X.iloc[eval_split:], combined_y.iloc[eval_split:]

    best_model = _fit_model(study.best_params, X_fit, y_fit, X_es, y_es)

    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)
    val_metrics = _metrics(y_val.to_numpy(), y_val_pred)
    test_metrics = _metrics(y_test.to_numpy(), y_test_pred)
    print("VAL  metrics:", val_metrics)
    print("TEST metrics:", test_metrics)

    # Save model
    with (MODELS_DIR / "xgb_hpo.pkl").open("wb") as f:
        pickle.dump(best_model, f)

    # HPO trial history
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df.to_csv(RESULTS_DIR / "hpo_trials.csv", index=False)

    # Metrics json
    metrics = {
        "horizon": fm.horizon,
        "n_features": len(fm.feature_names),
        "feature_names": fm.feature_names,
        "n_trials": args.n_trials,
        "best_cv_mae_min": float(study.best_value),
        "best_params": study.best_params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (RESULTS_DIR / "xgb_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Test residuals for LP simulator
    residuals_df = pd.DataFrame(
        {
            "arrival_ts": fm.arrival_ts[is_test].reset_index(drop=True),
            "charger_id": fm.charger_id[is_test].reset_index(drop=True),
            "y_true": y_test,
            "y_pred": y_test_pred,
            "residual": y_test_pred - y_test.to_numpy(),
        }
    )
    residuals_df.to_parquet(RESULTS_DIR / "xgb_residuals.parquet", index=False)

    # Figures
    _plot_pred_vs_actual(y_test.to_numpy(), y_test_pred, FIG_DIR / "pred_vs_actual.png")
    _plot_feat_importance(best_model, fm.feature_names, FIG_DIR / "feat_importance.png")
    _plot_residuals(residuals_df["residual"].to_numpy(), FIG_DIR / "residual_distribution.png")
    _plot_hpo_importance(study, FIG_DIR / "hpo_importance.png")

    print()
    print("outputs written:")
    print(f"  {(MODELS_DIR / 'xgb_hpo.pkl').relative_to(ROOT)}")
    print(f"  {(RESULTS_DIR / 'hpo_trials.csv').relative_to(ROOT)}")
    print(f"  {(RESULTS_DIR / 'xgb_metrics.json').relative_to(ROOT)}")
    print(f"  {(RESULTS_DIR / 'xgb_residuals.parquet').relative_to(ROOT)}")
    print(f"  {FIG_DIR.relative_to(ROOT)}/*.png")


if __name__ == "__main__":
    main()
