import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import mse
from tvDatafeed import TvDatafeed, Interval
import time
from rule_engine import RuleEngine, IndicatorEngine, PatternEngine
from dataclasses import dataclass
from mse import detect_trend_vectorized

@dataclass
class Candle:
    time: int; open: float; high: float; low: float; close: float; volume: float

def load_candles_from_df(df :pd.DataFrame):
    df['time'] = df.index.astype(np.int64) // 10**9
    candle_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    return [Candle(**row) for row in df[candle_cols].to_dict('records')]

def resample_df(df, rule):
    return df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

def extract_order_blocks(df: pd.DataFrame) -> List[Dict]:
    # This is still not fully vectorized, but much faster
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(df['high'], distance=5)
    troughs, _ = find_peaks(-df['low'], distance=5)

    zones = []
    for p in peaks: zones.append({"type": "supply", "high": df['high'].iloc[p], "low": df['low'].iloc[p], "index": p})
    for t in troughs: zones.append({"type": "demand", "high": df['high'].iloc[t], "low": df['low'].iloc[t], "index": t})
    return zones

def extract_fvgs(df: pd.DataFrame) -> List[Dict]:
    low_gt_prev_high = df['low'] > df['high'].shift(2)
    high_lt_prev_low = df['high'] < df['low'].shift(2)

    fvgs = []
    for i in df.index[low_gt_prev_high]:
        idx = df.index.get_loc(i)
        if idx > 1: fvgs.append({"type": "fvg_up", "high": df['high'].iloc[idx-2], "low": df['low'].iloc[idx], "index": idx})
    for i in df.index[high_lt_prev_low]:
        idx = df.index.get_loc(i)
        if idx > 1: fvgs.append({"type": "fvg_down", "high": df['high'].iloc[idx], "low": df['low'].iloc[idx-2], "index": idx})
    return fvgs

class Backtester:
    def __init__(self, strategies: List[Dict], mse_analyst: mse.MSEAnalyst):
        self.strategies = strategies
        self.mse = mse_analyst
        self.rule_engine = RuleEngine()
        self.indicator_engine = IndicatorEngine()
        self.pattern_engine = PatternEngine()

    def run(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        # --- Pre-computation ---
        self.indicator_engine.compute_all_indicators(df, self.strategies)
        all_patterns = self.pattern_engine.compute_all_patterns(df)
        zones = extract_order_blocks(df)
        fvgs = extract_fvgs(df)

        # --- Trend Pre-computation ---
        df_5m = resample_df(df, '5T')
        df_15m = resample_df(df, '15T')
        trends = {
            'ltf': detect_trend_vectorized(df).reindex(df.index, method='ffill'),
            'mtf': detect_trend_vectorized(df_5m).reindex(df.index, method='ffill'),
            'htf': detect_trend_vectorized(df_15m).reindex(df.index, method='ffill')
        }

        trades = []
        i = 50
        while i < len(df):
            candle_time = df.index[i]

            historical_zones = [z for z in zones if z['index'] <= i]
            historical_fvgs = [f for f in fvgs if f['index'] <= i]

            context = self.mse.build_context(
                symbol=symbol, i=i, now_price=df['close'].iloc[i],
                trends=trends, atr=self.indicator_engine.indicators['atr'].iloc[i],
                zones=historical_zones, fvgs=historical_fvgs,
                timestamp=str(candle_time)
            )

            if context.score < 1 or context.compression:
                i += 1
                continue

            direction = context.structure_bias
            if direction not in ('long', 'short'):
                i += 1
                continue

            indicator_values = self.indicator_engine.get_indicator_values_at_candle(i)
            pattern_values = {p: s.iloc[i] for p, s in all_patterns.items()}

            trade_opened = False
            for strategy in self.strategies:
                if strategy.get("entry_direction", "LONG").lower() != direction: continue

                if self.rule_engine.evaluate_strategy(strategy, indicator_values, pattern_values):
                    entry_price = df['close'].iloc[i]
                    sl, tp = context.nearest_support, context.nearest_resistance

                    if sl == entry_price or tp == entry_price or abs(entry_price - sl) == 0:
                         i += 1; continue

                    rr = abs(tp - entry_price) / abs(entry_price - sl)
                    if rr < 1.0:
                        i+=1; continue

                    exit_found = False
                    for j in range(i + 1, len(df)):
                        outcome, exit_price = None, 0
                        if direction == 'long':
                            if df['low'].iloc[j] <= sl: outcome, exit_price = 'loss', sl
                            elif df['high'].iloc[j] >= tp: outcome, exit_price = 'win', tp
                        else:
                            if df['high'].iloc[j] >= sl: outcome, exit_price = 'loss', sl
                            elif df['low'].iloc[j] <= tp: outcome, exit_price = 'win', tp

                        if outcome:
                            pnl = (exit_price - entry_price) if direction == 'long' else (entry_price - exit_price)
                            trades.append({
                                'symbol': symbol, 'side': direction, 'strategy': strategy['strategy_name'],
                                'open_time': str(candle_time), 'entry_price': entry_price, 'sl': sl, 'tp': tp,
                                'exit_time': str(df.index[j]), 'exit_price': exit_price, 'outcome': outcome, 'pnl': pnl
                            })
                            i = j
                            exit_found = True
                            break

                    if exit_found:
                        trade_opened = True
                        break

            if not trade_opened:
                i += 1

        return trades

def getDF(symbol="NIFTY", timeframes=None, max_retries=3):
    tv = TvDatafeed()
    interval_str = (timeframes[0] if timeframes and timeframes[0] else "1").replace("m","")
    interval = getattr(Interval, f"in_{interval_str}_minute", Interval.in_1_minute)

    for attempt in range(max_retries):
        try:
            df = tv.get_hist(symbol=symbol, exchange='NSE', interval=interval, n_bars=2500)
            if df is not None and not df.empty: return df.sort_index()
        except Exception as e: print(f"Error fetching {symbol}: {e}")
        time.sleep(2 * (attempt + 1))
    return None

def run_backtest(strategy: Dict, symbol: str, df: pd.DataFrame):
    bt = Backtester([strategy], mse.MSEAnalyst())
    return bt.run(symbol, df)
