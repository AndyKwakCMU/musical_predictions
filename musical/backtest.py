##############################################################################

#           backtest
#           Andy Kwak

##############################################################################

#       Event-driven backtester for the gate -> classify flow.
#       Trades exactly what labels.py labels: signal at bar t close,
#       enter at t+1 open, stop/target at entry -/+ k * ATR(t) (mirrored
#       for shorts), time exit at bar t+horizon close.
#
#       Conservative fill rules:
#         - a gap through the stop/target fills at that bar's open
#         - stop and target inside one bar -> assume the stop hit first
#         - one open position per ticker; overlapping signals skipped
#
#       Equity is marked only when trades close (no intra-trade marking),
#       so drawdowns are slightly understated.


#       @requires protocol bars + ATR per ticker, signals with direction
#       @ensures trade log, equity curve, metrics

##############################################################################

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    commission_per_share: float = 0.0035   # IBKR fixed-ish US stock rate
    commission_min: float = 0.35
    slippage_bps: float = 2.0              # per side

    def commission(self, shares):
        #@ensures \result is the commission for one fill of `shares`
        return max(self.commission_min, self.commission_per_share * shares)

    def slip(self, price, direction, entering):
        #@ensures \result is price moved against us by slippage_bps
        sign = direction if entering else -direction
        return price * (1 + sign * self.slippage_bps / 10_000)


def _exit_for(bars, t, direction, entry, risk, horizon):
    #@requires bars positional arrays, entry at bar t+1 open
    #@ensures (exit_bar, exit_price, reason)
    opens, highs, lows, closes = bars
    stop = entry - direction * risk
    target = entry + direction * risk
    for j in range(t + 1, t + horizon + 1):
        if j > t + 1:   # the entry bar's open is the entry itself
            gap_stop = opens[j] <= stop if direction > 0 else opens[j] >= stop
            gap_target = opens[j] >= target if direction > 0 else opens[j] <= target
            if gap_stop:
                return j, opens[j], "stop"
            if gap_target:
                return j, opens[j], "target"
        hit_stop = lows[j] <= stop if direction > 0 else highs[j] >= stop
        hit_target = highs[j] >= target if direction > 0 else lows[j] <= target
        if hit_stop:
            return j, stop, "stop"
        if hit_target:
            return j, target, "target"
    return t + horizon, closes[t + horizon], "time"


def simulate_trades(bars_by_ticker, atr_by_ticker, signals, k, horizon, cost):
    #@requires signals has ticker, timestamp, direction (+1 / -1)
    #@ensures \result has one row per taken trade (per-share, unsized)
    rows = []
    signals = signals.sort_values("timestamp")
    for ticker, sig in signals.groupby("ticker", sort=False):
        bars = bars_by_ticker[ticker]
        arrays = tuple(bars[c].to_numpy(float) for c in ["open", "high", "low", "close"])
        atr = atr_by_ticker[ticker].reindex(bars.index).to_numpy(float)
        positions = bars.index.get_indexer(sig["timestamp"])
        busy_until = -1
        for t, direction in zip(positions, sig["direction"]):
            if t < 0 or t < busy_until or t + horizon >= len(bars):
                continue
            if not np.isfinite(atr[t]) or atr[t] <= 0:
                continue
            entry = arrays[0][t + 1]
            risk = k * atr[t]
            exit_bar, exit_price, reason = _exit_for(arrays, t, direction,
                                                     entry, risk, horizon)
            busy_until = exit_bar
            rows.append({
                "ticker": ticker,
                "signal_time": bars.index[t],
                "entry_time": bars.index[t + 1],
                "exit_time": bars.index[exit_bar],
                "direction": int(direction),
                "entry": entry,
                "exit": exit_price,
                "exit_reason": reason,
                "risk_per_share": risk,
                "r_multiple": direction * (exit_price - entry) / risk,
                "entry_fill": cost.slip(entry, direction, entering=True),
                "exit_fill": cost.slip(exit_price, direction, entering=False),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["entry_time", "ticker"]).reset_index(drop=True)
    return out


def run_portfolio(trades, cost, start_equity=100_000.0, risk_frac=0.01,
                  max_positions=10):
    #@requires trades from simulate_trades (or same columns)
    #@ensures (log, equity): sized trades taken, equity marked at exits.
    #         Each trade risks risk_frac of equity; notional is capped at
    #         equity / max_positions so gross exposure never exceeds 1x.
    equity = start_equity
    open_trades = []          # (exit_time, pnl)
    curve = []
    log = []

    def close_until(when):
        nonlocal equity, open_trades
        done = sorted([o for o in open_trades if o[0] < when], key=lambda o: o[0])
        open_trades = [o for o in open_trades if o[0] >= when]
        for exit_time, pnl in done:
            equity += pnl
            curve.append((exit_time, equity))

    if len(trades):
        curve.append((trades["entry_time"].iloc[0], equity))
    for tr in trades.itertuples(index=False):
        close_until(tr.entry_time)
        if len(open_trades) >= max_positions:
            continue
        by_risk = equity * risk_frac / tr.risk_per_share
        by_notional = equity / max_positions / tr.entry_fill
        shares = int(np.floor(min(by_risk, by_notional)))
        if shares <= 0:
            continue
        fees = 2 * cost.commission(shares)
        pnl = tr.direction * (tr.exit_fill - tr.entry_fill) * shares - fees
        open_trades.append((tr.exit_time, pnl))
        row = tr._asdict()
        row.update(shares=shares, fees=fees, pnl=pnl,
                   r_net=pnl / (tr.risk_per_share * shares))
        log.append(row)
    close_until(pd.Timestamp.max)

    curve = pd.Series(dict(curve) if curve else {}, dtype=float)
    curve = curve.groupby(level=0).last().sort_index()
    return pd.DataFrame(log), curve


def metrics(log, equity):
    #@requires log has pnl and r_net, equity indexed by time
    #@ensures dict of headline stats
    if len(log) == 0:
        return {"trades": 0}
    daily = equity.resample("B").last().ffill()
    rets = daily.pct_change().dropna()
    sharpe = np.sqrt(252) * rets.mean() / rets.std() if rets.std() > 0 else 0.0
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    wins, losses = log[log["pnl"] > 0], log[log["pnl"] <= 0]
    return {
        "trades": len(log),
        "hit_rate": len(wins) / len(log),
        "avg_win_r": wins["r_net"].mean() if len(wins) else 0.0,
        "avg_loss_r": losses["r_net"].mean() if len(losses) else 0.0,
        "expectancy_r": log["r_net"].mean(),
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
        "cagr": (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1,
        "sharpe": sharpe,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
    }
