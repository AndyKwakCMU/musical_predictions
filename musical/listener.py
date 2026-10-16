##############################################################################

#           listener
#           Andy Kwak

##############################################################################

#       The "listener": one shared network that has heard every ticker's
#       history and learned what usually comes next. It is a causal
#       temporal CNN (dilated 1D convolutions that only look backward),
#       trained self-supervised to predict the next HORIZON_K bars of
#       motion, range, volume surprise and onset strength. No musical
#       labels are needed, so it learns from all the data; the musical
#       layers (beats, keys, chords, cadences) later sit on its hidden
#       states (`encode`). A small ticker embedding is the "composer".
#       Slot 0 of the embedding is reserved for tickers never seen.


#       @requires per-ticker feature frames (features.py + rhythm onset)
#       @ensures trained listener, per-bar hidden states, save/load

##############################################################################

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from musical.data import load_bars
from musical.features import add_context, bar_features
from musical.rhythm import onset_strength

TARGET_COLS = ["ret_atr", "range_atr", "vol_surprise", "onset"]
HORIZON_K = 4
CONTEXT_TICKERS = ["SPY", "QQQ"]     # the rest of the orchestra
RTH_OPEN_MIN, RTH_CLOSE_MIN = 570, 960   # 09:30-16:00 exchange time


@dataclass
class ListenerConfig:
    window: int = 256
    channels: int = 64
    levels: int = 7             # dilations 1, 2, 4, ..., 64
    emb_dim: int = 8
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    steps_per_epoch: int = 500
    epochs: int = 40
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


class CausalConv(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x):
        #@ensures \result[..., t] depends on x[..., <= t] only
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class Block(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.body = nn.Sequential(
            CausalConv(channels, channels, 3, dilation), nn.ReLU(),
            nn.Dropout(dropout), CausalConv(channels, channels, 3, dilation),
        )

    def forward(self, x):
        return torch.relu(x + self.body(x))


class Listener(nn.Module):
    def __init__(self, n_inputs, ticker_vocab, cfg=ListenerConfig(), input_cols=None):
        super().__init__()
        self.ticker_vocab = list(ticker_vocab)
        self._lookup = {t: i + 1 for i, t in enumerate(self.ticker_vocab)}
        self.cfg = cfg
        self.input_cols = list(input_cols) if input_cols is not None else None
        self.n_inputs = n_inputs
        self.embed = (nn.Embedding(len(self.ticker_vocab) + 1, cfg.emb_dim)
                      if cfg.emb_dim > 0 else None)
        self.inp = nn.Conv1d(n_inputs + cfg.emb_dim, cfg.channels, 1)
        self.blocks = nn.Sequential(*[Block(cfg.channels, 2 ** i, cfg.dropout)
                                      for i in range(cfg.levels)])
        self.head = nn.Conv1d(cfg.channels, len(TARGET_COLS) * HORIZON_K, 1)
        self.register_buffer("in_mean", torch.zeros(1, n_inputs, 1))
        self.register_buffer("in_std", torch.ones(1, n_inputs, 1))
        self.register_buffer("tgt_mean", torch.zeros(len(TARGET_COLS)))
        self.register_buffer("tgt_std", torch.ones(len(TARGET_COLS)))

    def ticker_index(self, tickers):
        #@ensures \result[i] is the embedding slot for tickers[i] (0 = unseen)
        return torch.tensor([self._lookup.get(t, 0) for t in tickers], dtype=torch.long)

    def encode(self, x, ticker_idx):
        #@requires x raw inputs (B, n_inputs, T), ticker_idx (B,)
        #@ensures \result (B, channels, T) hidden states; [..., t] uses x[..., <= t]
        x = (x - self.in_mean) / self.in_std
        if self.embed is not None:
            e = self.embed(ticker_idx.to(x.device))[:, :, None].expand(-1, -1, x.shape[2])
            x = torch.cat([x, e], dim=1)
        return self.blocks(self.inp(x))

    def forward(self, x, ticker_idx):
        #@ensures \result (B, len(TARGET_COLS) * HORIZON_K, T): standardized
        #         predictions of targets at t+1..t+K, row j*K + (k-1)
        return self.head(self.encode(x, ticker_idx))


def target_matrix(df):
    #@ensures \result[t, j*K + (k-1)] = TARGET_COLS[j] at bar t+k (NaN past the end)
    cols = []
    for col in TARGET_COLS:
        for k in range(1, HORIZON_K + 1):
            cols.append(df[col].shift(-k).to_numpy(float))
    return np.column_stack(cols).astype(np.float32)


def _window_ok(bad, window):
    #@ensures \result[e] True if no bad row in [e-window+1, e]
    c = np.r_[0, np.cumsum(bad)]
    ok = np.zeros(len(bad), dtype=bool)
    if len(bad) >= window:
        ok[window - 1:] = (c[window:] - c[:-window]) == 0
    return ok


def valid_ends(df, input_cols, start, end, window):
    #@ensures positions e whose window [e-window+1, e] and targets through
    #         e+K are NaN-free and lie inside [start, end)
    ts = df.index
    bad_in = df[input_cols].isna().any(axis=1).to_numpy()
    bad_tgt = df[TARGET_COLS].isna().any(axis=1).to_numpy()
    e = np.flatnonzero(_window_ok(bad_in, window))
    e = e[e + HORIZON_K < len(df)]
    e = e[_window_ok(bad_tgt, window + HORIZON_K)[e + HORIZON_K]]   # rows e-W+1 .. e+K
    e = e[(ts[e - window + 1] >= pd.Timestamp(start)) & (ts[e + HORIZON_K] < pd.Timestamp(end))]
    return e


def make_windows(series_by_ticker, input_cols, start, end, window):
    #@ensures list of (ticker, end_position) usable for training/eval
    return [(t, int(e)) for t, df in series_by_ticker.items()
            for e in valid_ends(df, input_cols, start, end, window)]


class _Data:
    # per-ticker numpy arrays so batches are cheap slices
    def __init__(self, series_by_ticker, input_cols):
        self.tickers = list(series_by_ticker)
        self.X = {t: df[input_cols].to_numpy(np.float32) for t, df in series_by_ticker.items()}
        self.Y = {t: target_matrix(df) for t, df in series_by_ticker.items()}

    def batch(self, items, window):
        xs = np.stack([self.X[t][e - window + 1:e + 1].T for t, e in items])
        ys = np.stack([self.Y[t][e - window + 1:e + 1].T for t, e in items])
        return torch.from_numpy(xs), torch.from_numpy(ys)


def _fit_standardization(model, series_by_ticker, input_cols, train_range):
    lo, hi = (pd.Timestamp(x) for x in train_range)
    rows = [df[(df.index >= lo) & (df.index < hi)] for df in series_by_ticker.values()]
    rows = pd.concat(rows)
    mean_in, std_in = rows[input_cols].mean(), rows[input_cols].std()
    mean_t, std_t = rows[TARGET_COLS].mean(), rows[TARGET_COLS].std()
    model.in_mean.copy_(torch.tensor(mean_in.to_numpy(np.float32)).view(1, -1, 1))
    model.in_std.copy_(torch.tensor((std_in.to_numpy(np.float32) + 1e-6)).view(1, -1, 1))
    model.tgt_mean.copy_(torch.tensor(mean_t.to_numpy(np.float32)))
    model.tgt_std.copy_(torch.tensor(std_t.to_numpy(np.float32) + 1e-6))


def _standardize_targets(model, y):
    mean = model.tgt_mean.repeat_interleave(HORIZON_K).view(1, -1, 1)
    std = model.tgt_std.repeat_interleave(HORIZON_K).view(1, -1, 1)
    return (y - mean) / std


def spaced(items, gap):
    #@ensures a subset of (ticker, end) items at least `gap` bars apart per
    #         ticker, so evaluation windows' scored halves don't overlap
    out, last = [], {}
    for t, e in items:
        if e - last.get(t, -10**9) >= gap:
            out.append((t, e))
            last[t] = e
    return out


def evaluate(model, data, items, window, device, size=512):
    #@ensures per-ticker sums of squared standardized error on the scored
    #         half of each window: {ticker: (sum_all, sum_onset, n_values)}
    model.eval()
    j = TARGET_COLS.index("onset")
    onset_rows = slice(j * HORIZON_K, (j + 1) * HORIZON_K)
    sums = {}
    with torch.no_grad():
        for i in range(0, len(items), size):
            chunk = items[i:i + size]
            x, y = data.batch(chunk, window)
            idx = model.ticker_index([t for t, _ in chunk]).to(device)
            pred = model(x.to(device), idx)[..., window // 2:]
            err = (pred - _standardize_targets(model, y.to(device))[..., window // 2:]) ** 2
            per_all = err.mean(dim=1).sum(dim=1).cpu().numpy()
            per_on = err[:, onset_rows].mean(dim=1).sum(dim=1).cpu().numpy()
            for (t, _), a, o in zip(chunk, per_all, per_on):
                s = sums.setdefault(t, [0.0, 0.0, 0])
                s[0] += float(a)
                s[1] += float(o)
                s[2] += err.shape[2]
    return {t: tuple(v) for t, v in sums.items()}


def _pooled(sums):
    n = sum(v[2] for v in sums.values())
    return sum(v[0] for v in sums.values()) / n, sum(v[1] for v in sums.values()) / n


def train_listener(series_by_ticker, input_cols, cfg, train_range, val_range):
    #@requires each frame has input_cols and TARGET_COLS, time-indexed
    #@ensures (best-val-loss listener, history with train_loss, val_loss,
    #         val_loss_onset); losses are MSE on standardized targets
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = Listener(len(input_cols), sorted(series_by_ticker), cfg, input_cols)
    _fit_standardization(model, series_by_ticker, input_cols, train_range)
    model.to(cfg.device)

    data = _Data(series_by_ticker, input_cols)
    train_items = make_windows(series_by_ticker, input_cols, *train_range, cfg.window)
    val_items = spaced(make_windows(series_by_ticker, input_cols, *val_range, cfg.window),
                       cfg.window // 2)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_loss_onset": []}
    best, best_state, flat = np.inf, None, 0
    half = cfg.window // 2
    for _ in range(cfg.epochs):
        model.train()
        total = 0.0
        for _ in range(cfg.steps_per_epoch):
            pick = rng.integers(0, len(train_items), cfg.batch_size)
            chunk = [train_items[i] for i in pick]
            x, y = data.batch(chunk, cfg.window)
            idx = model.ticker_index([t for t, _ in chunk]).to(cfg.device)
            y = _standardize_targets(model, y.to(cfg.device))
            loss = ((model(x.to(cfg.device), idx) - y)[..., half:] ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        history["train_loss"].append(total / cfg.steps_per_epoch)
        val, val_onset = _pooled(evaluate(model, data, val_items, cfg.window, cfg.device))
        history["val_loss"].append(val)
        history["val_loss_onset"].append(val_onset)
        print(f"epoch {len(history['val_loss'])}: train {history['train_loss'][-1]:.4f} "
              f"val {val:.4f} val_onset {val_onset:.4f}", flush=True)
        if val < best - 1e-4:
            best, best_state, flat = val, copy.deepcopy(model.state_dict()), 0
        else:
            flat += 1
            if flat >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.history = history
    return model, history


def save_listener(model, path):
    #@ensures path holds weights + everything needed to rebuild the listener
    torch.save({
        "state": model.state_dict(),
        "n_inputs": model.n_inputs,
        "ticker_vocab": model.ticker_vocab,
        "input_cols": model.input_cols,
        "cfg": asdict(model.cfg),
        "history": getattr(model, "history", {}),
    }, path)


def load_listener(path, device="cpu"):
    #@requires path written by save_listener
    #@ensures \result is the same listener in eval mode on device
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = ListenerConfig(**{**blob["cfg"], "device": device})
    model = Listener(blob["n_inputs"], blob["ticker_vocab"], cfg, blob["input_cols"])
    model.load_state_dict(blob["state"])
    model.history = blob["history"]
    return model.to(device).eval()


def load_series(raw_dir, end, interval="15m"):
    #@ensures {ticker: features + SPY/QQQ context + onset} built only from
    #         regular-hours bars strictly before `end` (stray extended-hours
    #         bars in the raw files are dropped)
    end = pd.Timestamp(end)
    feats = {}
    for path in sorted(Path(raw_dir).glob(f"*_{interval}.csv")):
        bars = load_bars(path)
        minutes = bars.index.hour * 60 + bars.index.minute
        rth = (minutes >= RTH_OPEN_MIN) & (minutes < RTH_CLOSE_MIN)
        bars = bars[rth & (bars.index < end)]
        feats[path.name.split("_")[0]] = bar_features(bars)
    context = {name: feats[name] for name in CONTEXT_TICKERS if name in feats}
    out = {}
    for ticker, f in feats.items():
        f = add_context(f, context)
        f["onset"] = onset_strength(f)
        out[ticker] = f
    return out
