"""Generalised re-implementation of the archived two-stage recursive trainer.

Mirrors run_metaheuristic_search.teacher_pretrain / rollout_finetune and
run_softcomputing_search.rollout_predict_common exactly, but takes the feature
list and the model factory as arguments so that ablations and other sequence
families receive identical treatment (same scaling, losses, optimiser, early
stopping, rollout fine-tuning and common evaluation window).

Information set (matches the archived code): the prediction for checkup i uses
the window of rows i-L .. i-1; each row holds that row's exogenous descriptors
plus its (measured or predicted) Q and Re. Rows < 20 carry measured targets
(forecast origin = row 20); from row 20 onward the targets in the buffer are the
model's own predictions. Exogenous descriptors of row i itself are not used.
"""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent.parent / "experiments"
INPUT = HERE / "inputs"
import os
DEVICE = torch.device(os.environ.get("JES_DEVICE", "cpu"))
LEGACY_FEEDBACK = os.environ.get("LEGACY_FEEDBACK") == "1"  # reproduce archived (buggy) feedback only
TARGETS = ["Q"]
FULL9 = ["k_exp", "temperature", "c_rate_chg", "c_rate_dischg", "soc_window", "age_type", "Q0"]
COMMON_START = int(__import__("os").environ.get("JES_ORIGIN", "20"))  # forecast origin (number of measured checkups)
SCHEDULES = {
    "search": dict(teacher_epochs=80, teacher_patience=18, rollout_epochs=15, rollout_patience=5),
    "rescreen": dict(teacher_epochs=160, teacher_patience=30, rollout_epochs=30, rollout_patience=10),
    "final": dict(teacher_epochs=220, teacher_patience=40, rollout_epochs=40, rollout_patience=15),
    "smoke": dict(teacher_epochs=2, teacher_patience=1, rollout_epochs=1, rollout_patience=1),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mape(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.mean(np.abs((a - b) / np.clip(np.abs(a), 1e-8, None))) * 100)


def metrics_2d(yt, yp):
    out, ms = {}, []
    for j, n in enumerate(TARGETS):
        m = mape(yt[:, j], yp[:, j]); ms.append(m)
        out[f"{n}_MAPE"] = m
        out[f"{n}_RMSE"] = float(np.sqrt(np.mean((yt[:, j] - yp[:, j]) ** 2)))
        out[f"{n}_R2"] = float(r2_score(yt[:, j], yp[:, j]))
    out["macro_MAPE"] = float(np.mean(ms))
    return out


# ----------------------------------------------------------------- data
def load_frame(extra: bool = True) -> pd.DataFrame:
    df = pd.read_csv(INPUT / "phase1_cv_all_rows.csv")
    if extra:
        first = df.groupby("cell_id")["timestamp_s"].transform("min")
        df["t_days"] = (df["timestamp_s"] - first) / 86400.0
    return df


def split_fixed(df: pd.DataFrame):
    split = pd.read_csv(INPUT / "fixed_cell_split.csv").set_index("cell_id")["split"]
    s = df["cell_id"].map(split)
    assert s.notna().all()
    return df[s.eq("train")].copy(), df[s.eq("validation") | s.eq("val")].copy(), df[s.eq("test")].copy()


def ordered(g0: pd.DataFrame) -> pd.DataFrame:
    # archived code sorts by k_exp (ties keep file order, which is time order); keep identical semantics
    return g0.sort_values("timestamp_s", kind="mergesort").reset_index(drop=True)


def build_sequences(df, features, L):
    cols = features + TARGETS
    xs, ys = [], []
    for _, g0 in df.groupby("cell_id", sort=False):
        g = ordered(g0)
        arr = g[cols].to_numpy(np.float32); targ = g[TARGETS].to_numpy(np.float32)
        for i in range(L, len(g)):
            xs.append(arr[i - L:i]); ys.append(targ[i])
    if not xs:
        return np.empty((0, L, len(cols)), np.float32), np.empty((0, 2), np.float32)
    return np.stack(xs), np.stack(ys)


def rollout_segments(df, features, L, H, sx, sy, stride=2):
    x0s, exogs, ys = [], [], []
    nf = len(features)
    for _, g0 in df.groupby("cell_id", sort=False):
        g = ordered(g0)
        if len(g) < L + H:
            continue
        full = g[features + TARGETS].to_numpy(np.float32)
        exog = g[features].to_numpy(np.float32)
        targ = g[TARGETS].to_numpy(np.float32)
        full_s = ((full - sx.mean_) / sx.scale_).astype(np.float32)
        exog_s = ((exog - sx.mean_[:nf]) / sx.scale_[:nf]).astype(np.float32)
        targ_s = sy.transform(targ).astype(np.float32)
        for i in range(L, len(g) - H + 1, stride):
            x0s.append(full_s[i - L:i]); exogs.append(exog_s[i:i + H]); ys.append(targ_s[i:i + H])
    return np.asarray(x0s), np.asarray(exogs), np.asarray(ys)


# ----------------------------------------------------------------- models
class LSTMMulti(nn.Module):
    def __init__(self, in_dim, hidden=192, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(TARGETS)))

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :])


class GRUMulti(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=1, dropout=0.2):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(TARGETS)))

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h[:, -1, :])


class TCNBlock(nn.Module):
    def __init__(self, ch, dilation, dropout=0.1):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(ch, ch, 3, dilation=dilation, padding=self.pad)
        self.norm = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv(x)[:, :, :-self.pad]
        return torch.relu(x + self.drop(torch.relu(self.norm(y))))


class TCNMulti(nn.Module):
    """Causal residual TCN, same family as the archived TCN baseline."""
    def __init__(self, in_dim, hidden=96, layers=4, dropout=0.1):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, hidden, 1)
        self.blocks = nn.Sequential(*[TCNBlock(hidden, 2 ** i, dropout) for i in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, len(TARGETS)))

    def forward(self, x):
        return self.head(self.blocks(self.inp(x.transpose(1, 2)))[:, :, -1])


class TransformerMulti(nn.Module):
    """Causal Transformer encoder with learned positions; last-token readout."""
    def __init__(self, in_dim, hidden=64, layers=2, heads=4, dropout=0.1, max_len=32):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, len(TARGETS)))
        self._masks = {}

    def forward(self, x):
        L = x.shape[1]
        if L not in self._masks:
            self._masks[L] = torch.triu(torch.full((L, L), float("-inf")), 1)
        h = self.inp(x) + self.pos[:, :L]
        h = self.enc(h, mask=self._masks[L].to(x.device))
        return self.head(self.norm(h[:, -1]))


def make_model(family: str, in_dim: int, cfg: dict) -> nn.Module:
    if family == "lstm":
        return LSTMMulti(in_dim, cfg["hidden"], cfg["layers"])
    if family == "gru":
        return GRUMulti(in_dim, cfg["hidden"], cfg["layers"])
    if family == "tcn":
        return TCNMulti(in_dim, cfg["hidden"], cfg["layers"])
    if family == "transformer":
        return TransformerMulti(in_dim, cfg["hidden"], cfg["layers"], cfg.get("heads", 4))
    raise ValueError(family)


def count_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


# ----------------------------------------------------------------- training
def teacher_pretrain(train, val, family, cfg, features, seed, epochs, patience):
    set_seed(seed)
    L = cfg["L"]
    xtr, ytr = build_sequences(train, features, L)
    xva, yva = build_sequences(val, features, L)
    nf = xtr.shape[-1]
    sx, sy = StandardScaler(), StandardScaler()
    xtr = sx.fit_transform(xtr.reshape(-1, nf)).reshape(xtr.shape).astype(np.float32)
    xva = sx.transform(xva.reshape(-1, nf)).reshape(xva.shape).astype(np.float32)
    ytr = sy.fit_transform(ytr).astype(np.float32)
    yva_s = sy.transform(yva).astype(np.float32)
    model = make_model(family, nf, cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.tensor(xtr), torch.tensor(ytr)), batch_size=512, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    xv = torch.tensor(xva, dtype=torch.float32, device=DEVICE)
    best, best_metric, bad = None, np.inf, 0
    for _ in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.smooth_l1_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            yvp = sy.inverse_transform(model(xv).cpu().numpy())
        m = metrics_2d(sy.inverse_transform(yva_s), yvp)["macro_MAPE"]
        if m < best_metric - 1e-7:
            best_metric, bad = m, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best)
    return model, sx, sy


def rollout_predict(model, sx, sy, frame, features, L, teacher=False):
    """Common-window evaluation from row 20. teacher=True feeds measured history (one-step)."""
    model.eval()
    cols = features + TARGETS
    truth, preds, meta = [], [], []
    secs = 0.0
    nf = len(features)
    for cid, g0 in frame.groupby("cell_id", sort=False):
        g = ordered(g0)
        if len(g) <= COMMON_START:
            continue
        buf = g[cols].to_numpy(np.float32).copy()
        for i in range(COMMON_START, len(g)):
            x = sx.transform(buf[i - L:i]).reshape(1, L, len(cols)).astype(np.float32)
            t0 = time.perf_counter()
            with torch.no_grad():
                yp = model(torch.tensor(x, device=DEVICE)).cpu().numpy()
            secs += time.perf_counter() - t0
            yp = np.clip(sy.inverse_transform(yp)[0], [1e-6], None)
            truth.append(g.loc[i, TARGETS].to_numpy(np.float32)); preds.append(yp)
            meta.append({"cell_id": cid, "row": i, "step": i - COMMON_START + 1, "k_exp": g.loc[i, "k_exp"],
                         "age_type": int(g.loc[i, "age_type"])})
            if not teacher:
                buf[i, nf:nf + len(TARGETS)] = yp
    return np.asarray(truth), np.asarray(preds), pd.DataFrame(meta), secs


def rollout_finetune(model, train, val, cfg, features, sx, sy, seed, epochs, patience):
    L, H = cfg["L"], cfg["H"]
    x0, ex, tg = rollout_segments(train, features, L, H, sx, sy, stride=2)
    if not len(x0):
        return np.inf
    loader = DataLoader(TensorDataset(torch.tensor(x0), torch.tensor(ex), torch.tensor(tg)), batch_size=128,
                        shuffle=True, generator=torch.Generator().manual_seed(seed + 1))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"] * 0.30, weight_decay=1e-5)
    # Model outputs live in output-scaler (sy) coordinates, while the target-history channels of the
    # window live in input-scaler (sx) coordinates. Map p -> input coordinates before feeding it back
    # (differentiable affine map). The archived code appended p directly (LEGACY_FEEDBACK).
    fb_mul = torch.as_tensor(sy.scale_ / sx.scale_[-len(TARGETS):], dtype=torch.float32, device=DEVICE)
    fb_off = torch.as_tensor((sy.mean_ - sx.mean_[-len(TARGETS):]) / sx.scale_[-len(TARGETS):], dtype=torch.float32, device=DEVICE)
    yt, yp, _, _ = rollout_predict(model, sx, sy, val, features, L)
    best_metric = metrics_2d(yt, yp)["macro_MAPE"]
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bad = 0
    for _ in range(1, epochs + 1):
        model.train()
        for win, fx, yy in loader:
            win, fx, yy = win.to(DEVICE), fx.to(DEVICE), yy.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            losses = []
            for s in range(H):
                p = model(win)
                losses.append(nn.functional.mse_loss(p, yy[:, s, :]))
                p_in = p if LEGACY_FEEDBACK else p * fb_mul + fb_off
                nxt = torch.cat([fx[:, s, :], p_in], dim=1).unsqueeze(1)
                win = torch.cat([win[:, 1:, :], nxt], dim=1)
            loss = torch.stack(losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        yt, yp, _, _ = rollout_predict(model, sx, sy, val, features, L)
        m = metrics_2d(yt, yp)["macro_MAPE"]
        if m < best_metric - 1e-5:
            best_metric, bad = m, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best)
    return best_metric


def train_eval(train, val, test, family, cfg, features=FULL9, seed=42, schedule="final", eval_teacher=False):
    sch = SCHEDULES[schedule]
    t0 = time.perf_counter()
    model, sx, sy = teacher_pretrain(train, val, family, cfg, features, seed, sch["teacher_epochs"], sch["teacher_patience"])
    vm = rollout_finetune(model, train, val, cfg, features, sx, sy, seed, sch["rollout_epochs"], sch["rollout_patience"])
    secs = time.perf_counter() - t0
    if not np.isfinite(vm):
        yt, yp, _, _ = rollout_predict(model, sx, sy, val, features, cfg["L"])
        vm = metrics_2d(yt, yp)["macro_MAPE"]
    row = {"family": family, **cfg, "features": "+".join(features), "seed": seed, "schedule": schedule,
           "val_macro_MAPE": vm, "parameters": count_params(model), "train_seconds": secs}
    pred = None
    if test is not None:
        yt, yp, meta, isecs = rollout_predict(model, sx, sy, test, features, cfg["L"])
        row.update({f"test_{k}": v for k, v in metrics_2d(yt, yp).items()})
        row["test_ms_per_trajectory"] = isecs / max(meta.cell_id.nunique(), 1) * 1000
        pred = meta.copy(); pred[["Q_true"]] = yt; pred[["Q_pred"]] = yp
        if eval_teacher:
            yt2, yp2, _, _ = rollout_predict(model, sx, sy, test, features, cfg["L"], teacher=True)
            row.update({f"tf_test_{k}": v for k, v in metrics_2d(yt2, yp2).items()})
    return row, pred, (model, sx, sy)


def cpu_latency_ms(model, L, in_dim, warm=20, blocks=5, per=20, example=None):
    """Single-thread batch-one forward latency on CPU (median over 5 blocks of 20 calls after 20 warm-ups).
    Uses a real scaled test window when `example` is given (same input for every model family)."""
    m = model.to("cpu").eval()
    n0 = torch.get_num_threads(); torch.set_num_threads(1)
    x = torch.zeros(1, L, in_dim) if example is None else torch.as_tensor(example, dtype=torch.float32).reshape(1, L, in_dim)
    with torch.no_grad():
        for _ in range(warm):
            m(x)
        ts = []
        for _ in range(blocks):
            t0 = time.perf_counter()
            for _ in range(per):
                m(x)
            ts.append((time.perf_counter() - t0) * 1000 / per)
    torch.set_num_threads(n0)
    return float(np.median(ts))


def serialized_bytes(model):
    import io
    b = io.BytesIO(); torch.save(model.state_dict(), b); return b.getbuffer().nbytes


def quantize_dynamic_int8(model):
    """Post-training dynamic int8 quantization of recurrent and linear layers (CPU)."""
    import torch.ao.quantization as tq
    return tq.quantize_dynamic(copy.deepcopy(model).to("cpu").eval(), {nn.LSTM, nn.GRU, nn.Linear}, dtype=torch.qint8)
