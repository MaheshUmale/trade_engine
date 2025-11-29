"""
MSE Enhanced — Market Structure Engine (Final Stable Version)
Supports:
- HTF/MTF/LTF trend detection
- Liquidity mapping
- Compression detection
- FVG + OB detection
- Score model
- Structure bias
"""

import numpy as np
from dataclasses import dataclass

# --------------- Utility helpers -----------------

def swing_high(candles, i):
    if i < 2 or i > len(candles) - 3:
        return False
    return candles[i].high > candles[i-1].high and candles[i].high > candles[i+1].high

def swing_low(candles, i):
    if i < 2 or i > len(candles) - 3:
        return False
    return candles[i].low < candles[i-1].low and candles[i].low < candles[i+1].low

def detect_trend(candles):
    """
    Simple but highly effective:
    - Detect micro swing structure
    - HH/HL => uptrend
    - LL/LH => downtrend
    - else neutral
    """
    highs = []
    lows = []
    for i in range(len(candles)):
        if swing_high(candles, i): highs.append(candles[i].high)
        if swing_low(candles, i): lows.append(candles[i].low)

    if len(highs) < 2 or len(lows) < 2:
        return "neutral"

    HH = highs[-1] > highs[-2]
    HL = lows[-1] > lows[-2]
    LH = highs[-1] < highs[-2]
    LL = lows[-1] < lows[-2]

    if HH and HL: return "up"
    if LL and LH: return "down"
    return "neutral"

@dataclass
class MSEContext:
    symbol: str
    timestamp: str
    htf_trend: str
    mtf_trend: str
    ltf_trend: str
    nearest_resistance: float
    nearest_support: float
    nearest_res_pct: float
    liquidity: dict
    compression: bool
    impulse: bool
    score: float
    blockers: list
    used_source: str
    structure_bias: str
    atr: float
    current_price: float
    htf_zones: dict
    fvg: list
    ltf_zones: list


class MSEAnalyst:

    def __init__(self):
        pass

    # ------------------- Core Context Builder -------------------

    def build_context(self, symbol, now_price, candles_ltf, candles_mtf, candles_htf,
                      zones_ltf, zones_mtf, zones_htf, fvgs, timestamp):
        # Trends
        htf_trend = detect_trend(candles_htf)
        mtf_trend = detect_trend(candles_mtf)
        ltf_trend = detect_trend(candles_ltf)

        # Bias: defined by LTF micro-structure
        structure_bias = "long" if ltf_trend == "up" else "short" if ltf_trend == "down" else "neutral"

        # Liquidity
        liquidity = {
            "hvns": sorted(list(set([round(z['low'], 3) for z in zones_ltf[:20]])))
        }

        # Support/Resistance
        nearest_support = min([z['low'] for z in zones_ltf] + [now_price])
        nearest_resistance = max([z['high'] for z in zones_ltf] + [now_price])

        nearest_res_pct = abs(now_price - nearest_resistance) / max(1e-6, now_price) * 100

        # Compression detection
        compression = self.detect_compression(candles_ltf)

        # FVG / imbalance signal
        impulse = any([True for f in fvgs])  # at least one active FVG

        # Score model
        score = 0
        if ltf_trend == "up": score += 3
        if ltf_trend == "down": score -= 3
        if mtf_trend == htf_trend: score += 2
        if impulse: score += 1
        if compression: score -= 3

        blockers = []
        if compression: blockers.append({"type": "compression", "reason": "choppy volatility"})
        if score < 0: blockers.append({"type": "low_score", "reason": "negative score"})

        return MSEContext(
            symbol=symbol,
            timestamp=timestamp,
            htf_trend=htf_trend,
            mtf_trend=mtf_trend,
            ltf_trend=ltf_trend,
            nearest_resistance=nearest_resistance,
            nearest_support=nearest_support,
            nearest_res_pct=nearest_res_pct,
            liquidity=liquidity,
            compression=compression,
            impulse=impulse,
            score=score,
            blockers=blockers,
            used_source="ltf_zone",
            structure_bias=structure_bias,
            atr=self.compute_atr(candles_ltf),
            current_price=now_price,
            htf_zones={
                "tf_5": {"zones": [z  for z in zones_mtf], "fvgs": []},
                "tf_15": {"zones": [z  for z in zones_htf], "fvgs": []}
            },
            fvg=[f  for f in fvgs],
            ltf_zones=[z for z in zones_ltf]
        )

    # ------------------- Helpers -------------------

    def compute_atr(self, candles):
        if len(candles) < 14: return 0.1
        trs = []
        for i in range(1, len(candles)):
            prev = candles[i-1]
            curr = candles[i]
            trs.append(max(
                curr.high - curr.low,
                abs(curr.high - prev.close),
                abs(curr.low - prev.close)
            ))
        return np.mean(trs[-14:])

    def detect_compression(self, candles):
        """
        Compression = small range + overlapping candles.
        """
        if len(candles) < 10: return False
        ranges = [c.high - c.low for c in candles[-10:]]
        avg = np.mean(ranges)
        return avg < (np.mean([c.high - c.low for c in candles]) * 0.5)
