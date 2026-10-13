import numpy as np
import torch

from musical.cnn import CadenceCNN, TrainConfig, load_model, predict_proba, save_model, train


def toy_data(n=600, seed=0):
    # Label is learnable from the input: sign of the window's mean in
    # channel 0 -> DOWN(0) / UP(2); near zero -> NONE(1)
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 8, 32)).astype(np.float32)
    drift = rng.choice([-1.0, 0.0, 1.0], size=n)
    X[:, 0, :] += drift[:, None]
    y = (drift + 1).astype(np.int64)
    tickers = rng.choice(["AAA", "BBB"], size=n)
    return X, y, tickers


def test_forward_gives_class_probabilities():
    X, _, tickers = toy_data(10)
    model = CadenceCNN(n_channels=8, ticker_vocab=["AAA", "BBB"])
    probs = predict_proba(model, X, tickers)
    assert probs.shape == (10, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_unknown_ticker_maps_to_reserved_slot():
    model = CadenceCNN(n_channels=8, ticker_vocab=["AAA"])
    assert model.ticker_index(["AAA", "ZZZ"]).tolist() == [1, 0]


def test_training_learns_a_learnable_task():
    X, y, tk = toy_data()
    cfg = TrainConfig(epochs=15, patience=5, device="cpu", seed=0)
    model, history = train(X[:450], y[:450], tk[:450],
                           X[450:], y[450:], tk[450:], cfg)
    acc = (predict_proba(model, X[450:], tk[450:]).argmax(1) == y[450:]).mean()
    assert acc > 0.8
    assert history["val_loss"][-1] < history["val_loss"][0]


def test_save_load_round_trip_gives_same_predictions(tmp_path):
    X, y, tk = toy_data(200)
    cfg = TrainConfig(epochs=2, device="cpu", seed=0)
    model, _ = train(X[:150], y[:150], tk[:150], X[150:], y[150:], tk[150:], cfg)
    path = tmp_path / "m.pt"
    save_model(model, path)
    again = load_model(path)
    np.testing.assert_allclose(predict_proba(model, X, tk),
                               predict_proba(again, X, tk), atol=1e-6)


def test_input_is_standardized_with_training_stats():
    X, y, tk = toy_data(200)
    X = X * 50 + 1000
    cfg = TrainConfig(epochs=1, device="cpu", seed=0)
    model, _ = train(X[:150], y[:150], tk[:150], X[150:], y[150:], tk[150:], cfg)
    mean = model.norm_mean.cpu().numpy().ravel()
    assert np.allclose(mean, X[:150].mean(axis=(0, 2)), rtol=1e-3)


def test_zero_embedding_dim_disables_ticker_embedding():
    # zero-width nn.Embedding crashes on CUDA, so it must not be built
    model = CadenceCNN(n_channels=8, ticker_vocab=["AAA"], emb_dim=0)
    assert model.embed is None
    X, _, tickers = toy_data(4)
    assert predict_proba(model, X, tickers).shape == (4, 3)
