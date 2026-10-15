import numpy as np
import pandas as pd
import torch

from musical.listener import (
    HORIZON_K, TARGET_COLS, Listener, ListenerConfig, load_listener,
    make_windows, save_listener, train_listener,
)

INPUTS = ["ret_atr", "range_atr", "vol_surprise", "onset", "tod_sin"]


def tiny_cfg(**kw):
    base = dict(window=64, channels=16, levels=4, emb_dim=2, dropout=0.0,
                lr=3e-3, batch_size=32, steps_per_epoch=40, epochs=3,
                patience=5, device="cpu", seed=0)
    base.update(kw)
    return ListenerConfig(**base)


def periodic_frame(n=4000, period=5, seed=0):
    rng = np.random.default_rng(seed)
    onset = rng.normal(0, 0.05, n)
    onset[::period] += 1.0
    idx = pd.date_range("2016-01-04 09:30", periods=n, freq="15min")
    return pd.DataFrame({
        "ret_atr": rng.normal(0, 1, n),
        "range_atr": rng.gamma(2.0, 0.5, n),
        "vol_surprise": rng.normal(0, 0.3, n),
        "onset": onset,
        "tod_sin": np.sin(np.arange(n) / 26 * 2 * np.pi),
    }, index=idx)


def test_listener_output_at_t_ignores_future_inputs():
    torch.manual_seed(0)
    model = Listener(len(INPUTS), ["AAA"], tiny_cfg()).eval()
    x = torch.randn(2, len(INPUTS), 200)
    idx = model.ticker_index(["AAA", "ZZZ"])
    out, enc = model(x, idx), model.encode(x, idx)
    x2 = x.clone()
    x2[..., 100:] = torch.randn_like(x2[..., 100:]) * 10
    assert out.shape == (2, len(TARGET_COLS) * HORIZON_K, 200)
    assert enc.shape == (2, 16, 200)
    assert torch.allclose(out[..., :100], model(x2, idx)[..., :100], atol=1e-6)
    assert torch.allclose(enc[..., :100], model.encode(x2, idx)[..., :100], atol=1e-6)


def test_unseen_ticker_maps_to_slot_zero():
    model = Listener(len(INPUTS), ["AAA", "BBB"], tiny_cfg())
    assert model.ticker_index(["BBB", "NEW"]).tolist() == [2, 0]


def test_windows_skip_nan_and_respect_split_end():
    df = periodic_frame(n=1000)
    df.iloc[300, df.columns.get_loc("ret_atr")] = np.nan
    end = df.index[900]
    wins = make_windows({"AAA": df}, INPUTS, df.index[0], end, window=64)
    ends = np.array([e for _, e in wins])
    assert len(ends) > 0
    assert not ((ends - 63 <= 300) & (ends >= 300)).any()       # no window covers the NaN
    assert (df.index[ends + HORIZON_K] < end).all()              # window + K stays inside
    assert (ends - 63 >= 0).all()


def test_listener_learns_tiny_periodic_series():
    df = periodic_frame()
    split = df.index[3000]
    model, hist = train_listener({"AAA": df}, INPUTS, tiny_cfg(),
                                 (df.index[0], split), (split, df.index[-1]))
    # standardized targets: predicting the mean scores ~1.0; a listener that
    # hears the period-5 pulse must do far better on the onset target
    assert hist["val_loss_onset"][-1] < 0.5
    assert len(hist["train_loss"]) == len(hist["val_loss"]) >= 1
    assert model.input_cols == INPUTS


def test_save_load_round_trip_gives_same_predictions(tmp_path):
    df = periodic_frame(n=1500)
    split = df.index[1000]
    model, _ = train_listener({"AAA": df}, INPUTS, tiny_cfg(epochs=1, steps_per_epoch=5),
                              (df.index[0], split), (split, df.index[-1]))
    path = tmp_path / "listener.pt"
    save_listener(model, path)
    again = load_listener(path)
    x = torch.randn(1, len(INPUTS), 128)
    idx = model.ticker_index(["AAA"])
    model.eval()
    assert torch.allclose(model.cpu()(x, idx), again(x, idx), atol=1e-6)
    assert again.input_cols == INPUTS
