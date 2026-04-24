# ruff: noqa: N803, N806
"""LSTM baseline for departure-time prediction.

Mirrors the XGBoost baseline (``scripts/train_xgb_hpo.py``) so the two
models share split, target, and evaluation protocol. Per HANDOFF v2.5
§7.2 and NOTES_to_DB_session_v1.md:

- Sequence input: first-10-minute current resampled to 20 bins
  (``data/phase_b/initial_sequences.parquet``).
- Static input: same 10 plug-in features as XGBoost plus
  ``has_meter_values`` flag, with categorical ``charger_id`` and
  ``station_id`` passed through learned embeddings.
- Target: ``duration_min`` clipped at 2,000 min.
- Loss: L1 (MAE) — directly comparable against the XGBoost
  ``reg:absoluteerror`` objective.

Outputs (mirror XGBoost artifacts):
- ``models/lstm.pt`` — best-weights state_dict + metadata.
- ``results/lstm_metrics.json`` — same schema as ``xgb_metrics.json``.
- ``results/lstm_residuals.parquet`` — per-test-session residuals (LP
  simulator ε source #2).
- ``outputs/lstm/pred_vs_actual.png``, ``training_curve.png``,
  ``residual_distribution.png``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tinkaton.ml_features import load_split_cutoffs
from tinkaton.transform import derive_station_id

ROOT = Path(__file__).resolve().parents[1]
SESSIONS_PARQUET = ROOT / "data" / "phase_b" / "session_dataset_clean_v2.parquet"
SEQ_PARQUET = ROOT / "data" / "phase_b" / "initial_sequences.parquet"
SPLIT_JSON = ROOT / "data" / "phase_b" / "split_definition.json"
CONFIG_PATH = ROOT / "configs" / "lstm_config.yaml"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "outputs" / "lstm"

OOV_INDEX = 0  # Reserved for unseen categories at val/test time


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


@dataclass
class Vocabs:
    charger: dict[str, int]
    station: dict[str, int]

    def encode_charger(self, values: pd.Series) -> np.ndarray:
        return values.map(self.charger).fillna(OOV_INDEX).astype(int).to_numpy()

    def encode_station(self, values: pd.Series) -> np.ndarray:
        return values.map(self.station).fillna(OOV_INDEX).astype(int).to_numpy()

    def vocab_size_charger(self) -> int:
        return max(self.charger.values()) + 1 if self.charger else 1

    def vocab_size_station(self) -> int:
        return max(self.station.values()) + 1 if self.station else 1


def build_vocabs(train_df: pd.DataFrame) -> Vocabs:
    """Label-encode categories using only the train split (no leakage)."""
    charger_levels = sorted(train_df["charger_id"].dropna().unique())
    station_levels = sorted(train_df["station_id"].dropna().unique())
    # Reserve 0 for OOV / unknown
    charger_map = {cid: i + 1 for i, cid in enumerate(charger_levels)}
    station_map = {sid: i + 1 for i, sid in enumerate(station_levels)}
    return Vocabs(charger=charger_map, station=station_map)


class SessionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_cols: list[str],
        static_numeric_cols: list[str],
        vocabs: Vocabs,
        y_col: str,
    ):
        self.seq = df[seq_cols].to_numpy(dtype=np.float32)
        self.static = df[static_numeric_cols].to_numpy(dtype=np.float32)
        self.charger_idx = vocabs.encode_charger(df["charger_id"]).astype(np.int64)
        self.station_idx = vocabs.encode_station(df["station_id"]).astype(np.int64)
        self.y = df[y_col].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.seq[idx]).unsqueeze(-1),  # (T, 1)
            torch.from_numpy(self.static[idx]),
            torch.tensor(self.charger_idx[idx], dtype=torch.long),
            torch.tensor(self.station_idx[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        *,
        seq_input_size: int,
        lstm_hidden: int,
        lstm_num_layers: int,
        lstm_dropout: float,
        bidirectional: bool,
        charger_vocab: int,
        station_vocab: int,
        charger_embedding_dim: int,
        station_embedding_dim: int,
        n_static: int,
        dense_hidden: list[int],
        dense_dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.charger_embed = nn.Embedding(charger_vocab, charger_embedding_dim)
        self.station_embed = nn.Embedding(station_vocab, station_embedding_dim)

        direction = 2 if bidirectional else 1
        fusion_in = (
            lstm_hidden * direction
            + charger_embedding_dim
            + station_embedding_dim
            + n_static
        )

        layers: list[nn.Module] = []
        prev = fusion_in
        for units in dense_hidden:
            layers += [nn.Linear(prev, units), nn.ReLU(), nn.Dropout(dense_dropout)]
            prev = units
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, seq, static, charger_idx, station_idx):
        _, (h_n, _) = self.lstm(seq)
        last = h_n[-1]  # (B, hidden)
        ch = self.charger_embed(charger_idx)
        st = self.station_embed(station_idx)
        fused = torch.cat([last, ch, st, static], dim=-1)
        return self.head(fused).squeeze(-1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _run_epoch(model, loader, loss_fn, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    total_n = 0
    for seq, static, ch_idx, st_idx, y in loader:
        seq = seq.to(device)
        static = static.to(device)
        ch_idx = ch_idx.to(device)
        st_idx = st_idx.to(device)
        y = y.to(device)
        if train:
            optimizer.zero_grad()
        pred = model(seq, static, ch_idx, st_idx)
        loss = loss_fn(pred, y)
        if train:
            loss.backward()
            optimizer.step()
        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(total_n, 1)


def _predict(model, loader, device) -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for seq, static, ch_idx, st_idx, _y in loader:
            pred = model(
                seq.to(device),
                static.to(device),
                ch_idx.to(device),
                st_idx.to(device),
            )
            outs.append(pred.cpu().numpy())
    return np.concatenate(outs)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(((err) ** 2).mean()))
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n": int(len(y_true)),
        "mae_min": mae,
        "rmse_min": rmse,
        "median_ae_min": float(np.median(abs_err)),
        "r2": float(r2),
        "pct_within_15min": float((abs_err <= 15).mean() * 100),
        "pct_within_30min": float((abs_err <= 30).mean() * 100),
        "pct_within_60min": float((abs_err <= 60).mean() * 100),
        "bias_min": float(err.mean()),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_training_curve(history: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = [h["epoch"] for h in history]
    train_mae = [h["train_mae"] for h in history]
    val_mae = [h["val_mae"] for h in history]
    ax.plot(epochs, train_mae, label="train", color="steelblue")
    ax.plot(epochs, val_mae, label="val", color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (min)")
    ax.set_title("LSTM training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_pred_vs_actual(y_true, y_pred, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=5, alpha=0.3)
    lim = max(float(y_true.max()), float(y_pred.max()), 1.0)
    ax.plot([0, lim], [0, lim], color="red", linestyle="--", label="y = ŷ")
    ax.set_xlabel("Actual duration (min)")
    ax.set_ylabel("Predicted duration (min)")
    ax.set_title("LSTM test set: predicted vs actual")
    ax.legend()
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
        f"LSTM test residuals   "
        f"mean={residuals.mean():.1f}   median={np.median(residuals):.1f}   "
        f"std={residuals.std():.1f}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    _set_seed(cfg["training"]["seed"])
    device = _select_device()
    print(f"device: {device}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {SESSIONS_PARQUET.relative_to(ROOT)} + sequences ...")
    sessions = pd.read_parquet(SESSIONS_PARQUET)
    sessions["station_id"] = sessions["charger_id"].map(derive_station_id)

    sequences = pd.read_parquet(SEQ_PARQUET)
    seq_cols = [c for c in sequences.columns if c.startswith("seq_step_")]
    assert len(seq_cols) == cfg["architecture"]["sequence_length"]
    sequences["transaction_id"] = sequences["transaction_id"].astype("Int64")
    sessions["transaction_id"] = sessions["transaction_id"].astype("Int64")
    df = sessions.merge(
        sequences[["charger_id", "transaction_id", *seq_cols]],
        on=["charger_id", "transaction_id"],
        how="left",
    )
    missing_seq = df[seq_cols].isna().any(axis=1).sum()
    if missing_seq > 0:
        print(f"  filling {missing_seq} sessions with zero-sequence (no merge key)")
        df[seq_cols] = df[seq_cols].fillna(0.0)

    # Prepare numeric static features
    cfg_input = cfg["input"]
    static_cols = list(cfg_input["features_static_numeric"])
    df["has_meter_values"] = df["has_meter_values"].astype(float)
    for col in static_cols:
        df[col] = df[col].astype(float).fillna(0.0)
    # Target clip
    cap = cfg["training"]["cap_duration_min"]
    df["duration_min"] = df["duration_min"].clip(upper=cap).astype(float)

    cutoffs = load_split_cutoffs(SPLIT_JSON)
    df["split"] = "test"
    df.loc[df["arrival_ts"] <= cutoffs["train_end"], "split"] = "train"
    df.loc[
        (df["arrival_ts"] > cutoffs["train_end"])
        & (df["arrival_ts"] <= cutoffs["val_end"]),
        "split",
    ] = "val"
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(
        f"split sizes — train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}"
    )

    vocabs = build_vocabs(train_df)
    print(
        f"vocab — charger: {vocabs.vocab_size_charger()}  "
        f"station: {vocabs.vocab_size_station()}"
    )

    # Standardize static numerics using train stats
    stat_mean = train_df[static_cols].mean()
    stat_std = train_df[static_cols].std().replace(0.0, 1.0)
    for col in static_cols:
        for frame in (train_df, val_df, test_df):
            frame[col] = (frame[col] - stat_mean[col]) / stat_std[col]
    # Standardize sequence using train stats (shared scale — current_a in amps)
    seq_mean = float(train_df[seq_cols].to_numpy().mean())
    seq_std = float(train_df[seq_cols].to_numpy().std())
    if seq_std == 0:
        seq_std = 1.0
    for frame in (train_df, val_df, test_df):
        frame[seq_cols] = (frame[seq_cols] - seq_mean) / seq_std

    datasets = {
        name: SessionDataset(
            frame,
            seq_cols=seq_cols,
            static_numeric_cols=static_cols,
            vocabs=vocabs,
            y_col="duration_min",
        )
        for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]
    }
    batch_size = cfg["training"]["batch_size"]
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }

    arch = cfg["architecture"]
    model = LSTMRegressor(
        seq_input_size=arch["sequence_features"],
        lstm_hidden=arch["lstm_hidden"],
        lstm_num_layers=arch["lstm_num_layers"],
        lstm_dropout=arch["lstm_dropout"],
        bidirectional=arch["bidirectional"],
        charger_vocab=vocabs.vocab_size_charger(),
        station_vocab=vocabs.vocab_size_station(),
        charger_embedding_dim=arch["charger_embedding_dim"],
        station_embedding_dim=arch["station_embedding_dim"],
        n_static=len(static_cols),
        dense_hidden=arch["dense_hidden"],
        dense_dropout=arch["dense_dropout"],
    ).to(device)
    loss_fn = nn.L1Loss() if cfg["training"]["loss"] == "l1" else nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best_val_mae = float("inf")
    best_state = None
    patience = cfg["training"]["early_stopping_patience"]
    stale = 0
    history: list[dict] = []

    print("training ...")
    for epoch in range(1, cfg["training"]["max_epochs"] + 1):
        tr_mae = _run_epoch(model, loaders["train"], loss_fn, optimizer, device, train=True)
        va_mae = _run_epoch(model, loaders["val"], loss_fn, optimizer, device, train=False)
        history.append({"epoch": epoch, "train_mae": tr_mae, "val_mae": va_mae})
        marker = " *" if va_mae < best_val_mae else ""
        print(f"  epoch {epoch:3d}  train MAE={tr_mae:7.3f}  val MAE={va_mae:7.3f}{marker}")
        if va_mae < best_val_mae:
            best_val_mae = va_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch}  (best val MAE={best_val_mae:.3f})")
                break

    assert best_state is not None
    model.load_state_dict(best_state)

    val_pred = _predict(model, loaders["val"], device)
    test_pred = _predict(model, loaders["test"], device)
    val_metrics = _metrics(val_df["duration_min"].to_numpy(), val_pred)
    test_metrics = _metrics(test_df["duration_min"].to_numpy(), test_pred)
    print("VAL  metrics:", val_metrics)
    print("TEST metrics:", test_metrics)

    # Persist model
    torch.save(
        {
            "state_dict": best_state,
            "vocab_charger_size": vocabs.vocab_size_charger(),
            "vocab_station_size": vocabs.vocab_size_station(),
            "arch_config": arch,
            "static_cols": static_cols,
            "seq_cols": seq_cols,
            "stat_mean": stat_mean.to_dict(),
            "stat_std": stat_std.to_dict(),
            "seq_mean": seq_mean,
            "seq_std": seq_std,
        },
        MODELS_DIR / "lstm.pt",
    )

    metrics = {
        "horizon": "plus_10min_sequence",
        "n_features_static": len(static_cols),
        "static_features": static_cols,
        "sequence_length": len(seq_cols),
        "model": "LSTM-with-embeddings",
        "best_val_mae_min": best_val_mae,
        "n_epochs_trained": history[-1]["epoch"],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "config": cfg,
    }
    (RESULTS_DIR / "lstm_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    residuals_df = pd.DataFrame(
        {
            "arrival_ts": test_df["arrival_ts"],
            "charger_id": test_df["charger_id"],
            "transaction_id": test_df["transaction_id"],
            "y_true": test_df["duration_min"].to_numpy(),
            "y_pred": test_pred,
            "residual": test_pred - test_df["duration_min"].to_numpy(),
        }
    )
    residuals_df.to_parquet(RESULTS_DIR / "lstm_residuals.parquet", index=False)

    _plot_training_curve(history, FIG_DIR / "training_curve.png")
    _plot_pred_vs_actual(
        test_df["duration_min"].to_numpy(), test_pred, FIG_DIR / "pred_vs_actual.png"
    )
    _plot_residuals(residuals_df["residual"].to_numpy(), FIG_DIR / "residual_distribution.png")

    print()
    print("outputs written:")
    print(f"  {(MODELS_DIR / 'lstm.pt').relative_to(ROOT)}")
    print(f"  {(RESULTS_DIR / 'lstm_metrics.json').relative_to(ROOT)}")
    print(f"  {(RESULTS_DIR / 'lstm_residuals.parquet').relative_to(ROOT)}")
    print(f"  {FIG_DIR.relative_to(ROOT)}/*.png")


if __name__ == "__main__":
    main()
