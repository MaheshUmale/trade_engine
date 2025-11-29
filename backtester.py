"""
backtester.py

Simple 1-min backtester that plugs RuleEngine + MSEAnalyst + simple risk sizing + simulated execution.

Usage:
  python backtester.py --csv data/BANKNIFTY_1m.csv --strategy "VWAP_Reclaim_Long"

CSV must have columns: datetime, open, high, low, close, volume
Datetime ISO or parseable; timezone naive is fine.

Outputs:
  - trades.csv (all trades)
  - prints summary metrics
"""
from tvDatafeed import TvDatafeed, Interval
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json
from math import floor

# import your rule engine and MSEAnalyst from existing files
from rule_engine import RuleEngine, IndicatorEngine, PatternEngine, EXAMPLE_STRATEGIES
from mse import MSEAnalyst, compute_trade_plan_from_ctx  # ensure mse.py exports these

# config
ACCOUNT = 200000.0
RISK_PCT = 0.01
MIN_RR = 2.0
MIN_QTY = 1

def compute_qty(entry, sl, account=ACCOUNT, risk_pct=RISK_PCT):
    tick_risk = abs(entry - sl)
    if tick_risk <= 0:
        return 0
    max_risk = account * risk_pct
    q = floor(max_risk / tick_risk)
    if q < MIN_QTY:
        return 0
    return q

def run_backtest( strategy_json, symbol="NIFTY", n_bars_lookback=2000):
    # df = pd.read_csv(csvfile, parse_dates=['datetime'])
    
    tv = TvDatafeed()
    df = tv.get_hist(symbol,exchange='NSE',interval=Interval.in_15_minute,n_bars=n_bars_lookback)
    # df['datetime'] = df.index
    df = df.sort_values('datetime').reset_index(drop=False)
    # print(df.head())
    # keep rolling list of candles as dicts
    recent = []
    ind = IndicatorEngine()
    pat = PatternEngine()
    rule_engine = RuleEngine()
    trades = []

    # which timeframe does strategy expect? assume 1m
    for idx, row in df.iterrows():
        candle = {
            'datetime': row['datetime'],
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': int(row['volume'])
        }
        recent.append(candle)
        if len(recent) < 20:
            continue
        # limit lookback to n_bars_lookback
        if len(recent) > n_bars_lookback:
            recent = recent[-n_bars_lookback:]

        # build market_context
        indicators = {}
        closes = [c['close'] for c in recent]
        indicators['ema9'] = ind.ema(closes[-30:], 9) if len(closes)>=9 else None
        indicators['ema20'] = ind.ema(closes[-60:], 20) if len(closes)>=20 else None
        indicators['vwap'] = ind.vwap(recent[-120:]) if len(recent)>=1 else recent[-1]['close']
        indicators['rvol'] = ind.rvol(recent, lookback=20)
        indicators['atr'] = ind.atr(recent, period=14)
        indicators['volume_spike'] = ind.volume_spike(recent, window=20, mult=1.5)

        patterns = {}
        if len(recent) >= 2:
            patterns['bullish_engulfing'] = pat.bullish_engulfing(recent[-2], recent[-1])
            patterns['pinbar'] = pat.pinbar(recent[-1])
            patterns['hammer'] = pat.hammer(recent[-1])

        market_context = {
            'ohlcv_recent': recent,
            'ohlcv_for_vwap': recent[-120:],
            'indicators': indicators,
            'patterns': patterns
        }

        # evaluate strategy
        res = rule_engine.evaluate_strategy(strategy_json, market_context)
        if not res.get('match'):
            continue

        # POTENTIAL_SIGNAL detected at this candle close (entry candidate next open)
        signal_time = candle['datetime']
        ltp = candle['close']

        # run MSEAnalyst on the same recent candles to produce SL/TP
        analyst = MSEAnalyst(pd.DataFrame(recent), symbol=symbol, signal_dir=strategy_json.get('entry_direction','LONG'))
        ctx = analyst.build_context()
        plan, reason = compute_trade_plan_from_ctx({'signal_dir': strategy_json.get('entry_direction','long')}, pd.DataFrame(recent), ctx, rr_min=MIN_RR)
        if plan is None:
            # skip
            continue

        entry = plan['entry']
        sl = plan['sl']
        tp = plan['tp']
        rr = plan['rr']

        qty = compute_qty(entry, sl)
        if qty == 0:
            continue

        # Simulate entry at next candle open if exists
        if idx+1 >= len(df):
            break
        next_open = float(df.loc[idx+1, 'open'])
        fill_price = next_open  # simple fill model
        # record trade open
        trade = {
            'symbol': symbol,
            'open_time': signal_time,
            'entry_price': fill_price,
            'qty': qty,
            'sl': sl,
            'tp': tp,
            'rr': rr,
            'close_time': None,
            'close_price': None,
            'pnl': None,
            'outcome': None
        }

        # Now simulate until SL/TP hit using subsequent candles
        closed = False
        for j in range(idx+1, len(df)):
            h = float(df.loc[j, 'high'])
            l = float(df.loc[j, 'low'])
            close_j = float(df.loc[j, 'close'])
            # long
            if h >= tp:
                close_price = tp
                pnl = (close_price - fill_price) * qty
                trade.update({'close_time': df.loc[j, 'datetime'], 'close_price': close_price, 'pnl': pnl, 'outcome': 'TP'})
                closed = True
                break
            if l <= sl:
                close_price = sl
                pnl = (close_price - fill_price) * qty
                trade.update({'close_time': df.loc[j, 'datetime'], 'close_price': close_price, 'pnl': pnl, 'outcome': 'SL'})
                closed = True
                break
            # optional: auto square off at market close (not implemented)
        if not closed:
            # close at last close
            last_close = float(df.loc[len(df)-1, 'close'])
            pnl = (last_close - fill_price) * qty
            trade.update({'close_time': df.loc[len(df)-1, 'datetime'], 'close_price': last_close, 'pnl': pnl, 'outcome': 'EXIT'})
        trades.append(trade)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("No trades taken")
        return None, None

    # compute metrics
    total_pnl = trades_df['pnl'].sum()
    wins = trades_df[trades_df['pnl']>0]
    win_rate = len(wins) / len(trades_df)
    avg_pnl = trades_df['pnl'].mean()
    # equity curve
    eq = trades_df['pnl'].cumsum()
    peak = eq.cummax()
    drawdown = (eq - peak).min()
    # simplified Sharpe: mean(daily returns)/std(daily returns); here use trade returns
    if trades_df['pnl'].std() == 0:
        sharpe = float('nan')
    else:
        sharpe = trades_df['pnl'].mean() / trades_df['pnl'].std() * np.sqrt(len(trades_df))

    summary = {
        'trades': len(trades_df),
        'total_pnl': float(total_pnl),
        'win_rate': float(win_rate),
        'avg_pnl': float(avg_pnl),
        'max_drawdown': float(drawdown),
        'sharpe_like': float(sharpe)
    }
    return trades_df, summary

if __name__ == "__main__":
    print('--strategy ="VWAP_Reclaim_Long", help="Strategy name (from EXAMPLE_STRATEGIES or custom JSON file)"')
    with open('strategies.json', 'r') as file:
        strategies = json.load(file)
    symbollist = ['NIFTY','BANKNIFTY','TCS','INFY','BHARATFORG','63MOONS','WELSPUNLIV','GRSE','M_M']
    for symbol in symbollist :
        for strategy in strategies:            
            trades_df, summary = run_backtest( strategy , symbol=symbol)
            if trades_df is not None:
                trades_df.to_csv("backtest_trades.csv", index=False,mode='a',header=False)
                print("Saved trades to backtest_trades.csv")
            print("Summary:", summary)
