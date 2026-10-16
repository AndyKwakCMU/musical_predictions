##############################################################################

#           cnn
#           Andy Kwak

##############################################################################

#       The "cadence recognizer": a small 1D CNN over (channels, width)
#       windows that predicts how a compression resolves (DOWN/NONE/UP).
#       One shared model learns cadences across all tickers; a small
#       ticker embedding lets it pick up each "composer's" accent.
#       Slot 0 of the embedding is reserved for tickers never seen in
#       training, so the model still works on new names.


#       @requires dataset.py windows and integer labels
#       @ensures trained model, probabilities, save/load

##############################################################################

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6          # early stop after this many flat val epochs
    dropout: float = 0.3
    emb_dim: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


class CadenceCNN(nn.Module):
    def __init__(self, n_channels, ticker_vocab, n_classes=3,
                 emb_dim=4, dropout=0.3):
        super().__init__()
        self.ticker_vocab = list(ticker_vocab)
        self._lookup = {t: i + 1 for i, t in enumerate(self.ticker_vocab)}
        self.hparams = dict(n_channels=n_channels, n_classes=n_classes,
                            emb_dim=emb_dim, dropout=dropout)

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=5, padding=2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
            )

        self.features = nn.Sequential(
            block(n_channels, 32), nn.MaxPool1d(2),
            block(32, 64), nn.MaxPool1d(2),
            block(64, 64),
        )
        # zero-width embeddings crash on CUDA, so emb_dim=0 means "none"
        self.embed = (nn.Embedding(len(self.ticker_vocab) + 1, emb_dim)
                      if emb_dim > 0 else None)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 * 2 + emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )
        self.register_buffer("norm_mean", torch.zeros(1, n_channels, 1))
        self.register_buffer("norm_std", torch.ones(1, n_channels, 1))

    def ticker_index(self, tickers):
        #@ensures \result[i] is the embedding slot for tickers[i] (0 = unseen)
        return torch.tensor([self._lookup.get(t, 0) for t in tickers],
                            dtype=torch.long)

    def forward(self, x, ticker_idx):
        #@requires x shape (N, C, W), ticker_idx shape (N,)
        #@ensures \result shape (N, n_classes) logits
        x = (x - self.norm_mean) / self.norm_std
        h = self.features(x)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        if self.embed is not None:
            pooled = torch.cat([pooled, self.embed(ticker_idx)], dim=1)
        return self.head(pooled)


def _batches(n, size, rng=None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, size):
        yield order[i:i + size]


def _eval_loss(model, X, idx, y, device, size=4096):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    with torch.no_grad():
        for b in _batches(len(X), size):
            xb = torch.from_numpy(X[b]).to(device)
            total += loss_fn(model(xb, idx[b].to(device)),
                             torch.from_numpy(y[b]).to(device)).item()
    return total / len(X)


def train(X_tr, y_tr, tickers_tr, X_val, y_val, tickers_val, cfg=TrainConfig()):
    #@requires X float32 (N, C, W), y int64 in {0,1,2}
    #@ensures (best-val-loss model, history dict)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = CadenceCNN(X_tr.shape[1], sorted(set(tickers_tr)),
                       emb_dim=cfg.emb_dim, dropout=cfg.dropout)
    model.norm_mean.copy_(torch.from_numpy(X_tr.mean(axis=(0, 2))).view(1, -1, 1))
    model.norm_std.copy_(torch.from_numpy(X_tr.std(axis=(0, 2)) + 1e-6).view(1, -1, 1))
    model.train_config = asdict(cfg)
    model.to(cfg.device)

    idx_tr = model.ticker_index(tickers_tr)
    idx_val = model.ticker_index(tickers_val)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": []}
    best, best_state, flat = np.inf, None, 0
    for _ in range(cfg.epochs):
        model.train()
        total = 0.0
        for b in _batches(len(X_tr), cfg.batch_size, rng):
            xb = torch.from_numpy(X_tr[b]).to(cfg.device)
            yb = torch.from_numpy(y_tr[b]).to(cfg.device)
            opt.zero_grad()
            loss = loss_fn(model(xb, idx_tr[b].to(cfg.device)), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        history["train_loss"].append(total / len(X_tr))
        val = _eval_loss(model, X_val, idx_val, y_val, cfg.device)
        history["val_loss"].append(val)
        if val < best - 1e-4:
            best, best_state, flat = val, copy.deepcopy(model.state_dict()), 0
        else:
            flat += 1
            if flat >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.history = history
    return model, history


def predict_proba(model, X, tickers, size=4096):
    #@ensures \result shape (N, 3), rows sum to 1
    device = next(model.parameters()).device
    idx = model.ticker_index(tickers)
    model.eval()
    out = []
    with torch.no_grad():
        for b in _batches(len(X), size):
            logits = model(torch.from_numpy(X[b]).to(device), idx[b].to(device))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 3), np.float32)


def save_model(model, path):
    #@ensures path holds weights + everything needed to rebuild the model
    torch.save({
        "state": model.state_dict(),
        "hparams": model.hparams,
        "ticker_vocab": model.ticker_vocab,
        "train_config": getattr(model, "train_config", {}),
        "history": getattr(model, "history", {}),
    }, path)


def load_model(path, device="cpu"):
    #@requires path written by save_model
    #@ensures \result is the same model in eval mode on device
    blob = torch.load(path, map_location=device, weights_only=False)
    model = CadenceCNN(ticker_vocab=blob["ticker_vocab"], **blob["hparams"])
    model.load_state_dict(blob["state"])
    model.train_config = blob["train_config"]
    model.history = blob["history"]
    return model.to(device).eval()
