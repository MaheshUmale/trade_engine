import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import mse_enhanced

# ------------------- Strategy Engine -------------------
class Strategy:
    def __init__(self, config: Dict[str, Any]):
        self.name = config["strategy_name"]
        self.enabled = config.get("enabled", True)
        self.direction = config.get("entry_direction", "LONG").lower()
        self.timeframes = config.get("timeframes", ["1m"])
        self.logic = config["logic"]

    def evaluate_entry(self, candle: Dict[str, Any], indicators: Dict[str, Any], patterns: List[str]) -> bool:
        def check_rule(rule):
            if 'operator' in rule:
                if rule['operator'] == 'AND':
                    return all(check_rule(r) for r in rule['rules'])
                elif rule['operator'] == 'OR':
                    return any(check_rule(r) for r in rule['rules'])
            elif 'indicator' in rule:
                val = indicators.get(rule['indicator'])
                target = rule['value']
                if isinstance(target, str):
                    target = indicators.get(target, target)
                if rule['condition'] == '>': return val > target
                if rule['condition'] == '<': return val < target
                if rule['condition'] == '>=': return val >= target
                if rule['condition'] == '<=': return val <= target
                if rule['condition'] == '==': return val == target
                return False
            elif 'pattern' in rule:
                return rule['pattern'] in patterns
            return False

        return check_rule(self.logic)


from mse_enhanced  import MSEAnalyst


# ----------------------------------------------------
# Candle structure
# ----------------------------------------------------

from dataclasses import dataclass

@dataclass
class Candle:
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float

import pandas as pd
# ----------------------------------------------------
# Candle Loader
# ----------------------------------------------------
def load_candles_DF(df :pd.DataFrame()):
    
    # Correct way to assign the epoch time in seconds to a new column:
    df['time'] = df.index.astype(np.int64) // 10**9
    print(df.head())
    candles = []
    
    # Iterate over the rows using .itertuples(), which yields a named tuple for each row
    for row in df.itertuples():
        candles.append(Candle(
            # Access column data using attribute access (row.time, row.open, etc.)
            time=int(row.time),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        ))

    candles.sort(key=lambda x: x.time)
    return candles
def load_candles(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candle file not found: {path}")

    data = json.loads(p.read_text())

    candles = []
    for c in data:
        candles.append(Candle(
            time=int(c["time"]),
            open=float(c["open"]),
            high=float(c["high"]),
            low=float(c["low"]),
            close=float(c["close"]),
            volume=float(c.get("volume", 0)),
        ))

    candles.sort(key=lambda x: x.time)
    return candles


# ----------------------------------------------------
# Dummy Zone + FVG generater (you replace with your real detector)
# ----------------------------------------------------
def extract_order_blocks(candles):
    """
    You should replace this with your OB detection,
    but for now I generate simple zigzag OBs for testing.
    """

    zones = []
    for i in range(2, len(candles) - 2):
        c = candles[i]

        # simple detection: swing low = demand OB
        if candles[i].low < candles[i - 1].low and candles[i].low < candles[i + 1].low:
            zones.append({
                "type": "demand",
                "high": candles[i].high,
                "low": candles[i].low,
                "created_time": datetime.utcfromtimestamp(c.time).isoformat(),
                "created_index": i
            })

        # swing high = supply OB
        if candles[i].high > candles[i - 1].high and candles[i].high > candles[i + 1].high:
            zones.append({
                "type": "supply",
                "high": candles[i].high,
                "low": candles[i].low,
                "created_time": datetime.utcfromtimestamp(c.time).isoformat(),
                "created_index": i
            })

    return zones


def extract_fvgs(candles):
    """
    Dummy FVG detection: if candle gaps past previous high/low.
    You should replace with your actual FVG detector.
    """
    fvgs = []
    for i in range(2, len(candles)):
        prev = candles[i - 1]
        cur = candles[i]

        if cur.low > prev.high:  # upside gap
            fvgs.append({
                "type": "fvg_up",
                "high": prev.high,
                "low": cur.low,
                "created_time": datetime.utcfromtimestamp(cur.time).isoformat(),
                "created_index": i
            })

        if cur.high < prev.low:  # downside gap
            fvgs.append({
                "type": "fvg_down",
                "high": prev.low,
                "low": cur.high,
                "created_time": datetime.utcfromtimestamp(cur.time).isoformat(),
                "created_index": i
            })

    return fvgs
# ------------------- Backtester -------------------
class Backtester:
    def __init__(self, strategies_file: str, mse : mse_enhanced, one_entry_per_candle=True, score_threshold=0):
        with open(strategies_file, 'r') as f:
            self.strategies = [Strategy(s) for s in json.load(f) if s.get('enabled', True)]
        self.mse = mse
        self.ONE_ENTRY_PER_CANDLE = one_entry_per_candle
        self.SCORE_THRESHOLD = score_threshold

    def run(self, symbol: str, candles: List[Dict], zones_ltf, zones_mtf, zones_htf, fvgs) -> List[Dict]:
        trades = []
        last_entry_index = None

        for i in range(50, len(candles)):
            candle = candles[i]
            now = candle.close
            context = self.mse.build_context(
                symbol=symbol,
                now_price=now,
                candles_ltf=candles[max(0, i-200):i],
                candles_mtf=candles[max(0, i-600):i],
                candles_htf=candles[max(0, i-1500):i],
                zones_ltf=[z for z in zones_ltf],
                zones_mtf=[z for z in zones_mtf],
                zones_htf=[z for z in zones_htf],
                fvgs=fvgs,
                timestamp=str(datetime.now())
            )

            if self.ONE_ENTRY_PER_CANDLE and last_entry_index == i:
                continue

            if context.score < self.SCORE_THRESHOLD:
                continue

            if any(b['type'] == 'compression' for b in context.blockers):
                continue

            direction = context.structure_bias
            if direction not in ('long', 'short'):
                continue

            if not (context.htf_trend == context.mtf_trend == context.ltf_trend):
                continue

            # Evaluate strategies
            for strategy in self.strategies:
                if strategy.direction != direction:
                    continue
                # Simplified: indicators & patterns dict
                indicators = context.indicators
                patterns = context.patterns
                if strategy.evaluate_entry(candle, indicators, patterns):
                    last_entry_index = i
                    entry_price = now
                    if direction == 'long':
                        sl = context.nearest_support
                        tp = context.nearest_resistance + context.atr * 1.5
                    else:
                        sl = context.nearest_resistance
                        tp = context.nearest_support - context.atr * 1.5

                    trades.append({
                        'symbol': symbol,
                        'side': direction,
                        'strategy': strategy.name,
                        'open_time': candle['time'],
                        'entry_price': entry_price,
                        'qty': 1,
                        'sl': sl,
                        'tp': tp,
                        'mse_context': context.__dict__
                    })
        return trades

    # ------------------- Performance -------------------
    def generate_performance_report(self, trades: List[Dict], out_file='performance_report.html'):
        df = pd.DataFrame(trades)
        df['pnl'] = np.where(df['side']=='long', df['tp']-df['entry_price'], df['entry_price']-df['tp'])
        win_rate = (df['pnl']>0).mean()
        avg_rr = df['pnl'].mean()
        max_dd = df['pnl'].cumsum().cummax() - df['pnl'].cumsum()

        html = f"""
        <h1>Backtest Performance Report</h1>
        <p>Win rate: {win_rate:.2%}</p>
        <p>Average R/R: {avg_rr:.4f}</p>
        <p>Max Drawdown: {max_dd.max():.4f}</p>
        """

        # Equity curve plot
        plt.figure(figsize=(10,5))
        plt.plot(df['pnl'].cumsum())
        plt.title('Equity Curve')
        plt.xlabel('Trades')
        plt.ylabel('Cumulative PnL')
        equity_curve_file = Path(out_file).with_suffix('.png')
        plt.savefig(equity_curve_file)
        plt.close()

        html += f'<img src="{equity_curve_file.name}" alt="Equity Curve">'
        Path(out_file).write_text(html)
        print(f'Performance report saved to {out_file}')

# ------------------- Example Usage -------------------
# bt = Backtester('strategies.json', mse)
# trades = bt.run('AAPL', candles, zones_ltf, zones_mtf, zones_htf, fvgs)
# bt.generate_performance_report(trades)



from tvDatafeed import TvDatafeed, Interval
import time
def getDF(symbol="NIFTY",  timeframes=None):
     
    if TvDatafeed is None:
        raise RuntimeError("TvDatafeed not available and no df passed in.")
    tv = TvDatafeed()
    n_bars_lookback = 2500
    interval = "1"
    if timeframes is not None and timeframes[0] is not None:
        interval = str(timeframes[0])
        df = tv.get_hist(symbol, exchange='NSE', interval=Interval(interval), n_bars=n_bars_lookback)
    else:
        df = tv.get_hist(symbol, exchange='NSE', interval=Interval.in_1_minute, n_bars=n_bars_lookback)
    if df is None:
        time.sleep(3)
        df = tv.get_hist(symbol, exchange='NSE', interval=Interval.in_1_minute, n_bars=n_bars_lookback)
    time.sleep(1)

    df = df.sort_values('datetime').reset_index(drop=False)
    return df

import mse_enhanced as mse
    # ---------- REPLACE the bottom of your backtester_event.py with this block ----------
if __name__ == "__main__":
    
    analyst = mse.MSEAnalyst()
    bt = Backtester('strategies.json', analyst )
    # list of symbols to test
    symbollist = ['63MOONS']  # change as needed
    for symbol in symbollist:

        df = getDF(symbol=symbol,timeframes =["1"])
        
        candles = load_candles_DF(df)

        # zone & fvg extraction (replace with your real detectors later)
        zones_ltf = extract_order_blocks(candles)
        zones_mtf = extract_order_blocks(candles)  # placeholder
        zones_htf = extract_order_blocks(candles)  # placeholder
        fvgs = extract_fvgs(candles)

        # engine = bt
        # trades = engine.run(symbol, candles, zones_ltf, zones_mtf, zones_htf, fvgs)

        
        trades = bt.run(symbol, candles, zones_ltf, zones_mtf, zones_htf, fvgs)
        bt.generate_performance_report(trades)
        