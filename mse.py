import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MSEContext:
    symbol: str; timestamp: str; htf_trend: str; mtf_trend: str; ltf_trend: str
    nearest_resistance: float; nearest_support: float
    compression: bool; impulse: bool; score: float
    structure_bias: str; atr: float

def detect_swing_points(series, order=5):
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(series, distance=order)
    troughs, _ = find_peaks(-series, distance=order)
    return peaks, troughs

def detect_trend_vectorized(df: pd.DataFrame):
    high_peaks, _ = detect_swing_points(df['high'])
    _, low_troughs = detect_swing_points(df['low'])

    swing_highs = pd.Series(np.nan, index=df.index)
    swing_highs.iloc[high_peaks] = df['high'].iloc[high_peaks]
    swing_highs = swing_highs.ffill()

    swing_lows = pd.Series(np.nan, index=df.index)
    swing_lows.iloc[low_troughs] = df['low'].iloc[low_troughs]
    swing_lows = swing_lows.ffill()

    higher_highs = (swing_highs > swing_highs.shift(1)).fillna(False)
    higher_lows = (swing_lows > swing_lows.shift(1)).fillna(False)
    lower_highs = (swing_highs < swing_highs.shift(1)).fillna(False)
    lower_lows = (swing_lows < swing_lows.shift(1)).fillna(False)

    uptrend = (higher_highs & higher_lows)
    downtrend = (lower_highs & lower_lows)

    trend = pd.Series("neutral", index=df.index)
    trend[uptrend] = "up"
    trend[downtrend] = "down"

    return trend

class MSEAnalyst:
    def get_nearest_sr(self, now_price: float, zones: List[Dict]) -> (float, float):
        supports = [z['low'] for z in zones if z['type'] == 'demand' and z['low'] < now_price]
        resistances = [z['high'] for z in zones if z['type'] == 'supply' and z['high'] > now_price]
        return max(supports) if supports else now_price * 0.98, min(resistances) if resistances else now_price * 1.02

    def build_context(self, symbol: str, i: int, now_price: float,
                      trends: Dict[str, pd.Series], atr: float,
                      zones: List[Dict], fvgs: List[Dict], timestamp: str) -> MSEContext:

        htf_trend = trends['htf'].iloc[i]
        mtf_trend = trends['mtf'].iloc[i]
        ltf_trend = trends['ltf'].iloc[i]

        structure_bias = "long" if ltf_trend == "up" else "short" if ltf_trend == "down" else "neutral"

        nearest_support, nearest_resistance = self.get_nearest_sr(now_price, zones)

        # Simplified for now, can be vectorized as well
        compression = False
        impulse = bool(fvgs)

        score = 0
        if ltf_trend == "up": score += 3
        if ltf_trend == "down": score -= 3
        if mtf_trend == htf_trend: score += 2
        if impulse: score += 1

        return MSEContext(
            symbol=symbol, timestamp=timestamp, htf_trend=htf_trend, mtf_trend=mtf_trend, ltf_trend=ltf_trend,
            nearest_resistance=nearest_resistance, nearest_support=nearest_support,
            compression=compression, impulse=impulse, score=score,
            structure_bias=structure_bias, atr=atr
        )
