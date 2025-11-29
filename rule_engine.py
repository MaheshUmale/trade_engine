"""
rule_engine.py

Rule Engine for Module 2 (Custom Logic Screener)
- Evaluate nested JSON strategies (AND/OR)
- Leaf evaluators: indicator comparisons, boolean flags, patterns
- IndicatorEngine + PatternEngine helpers (lightweight)

API:
  engine = RuleEngine()
  ctx = indicator_engine.compute_indicators(df_recent)   # compute once per new candle
  result = engine.evaluate_strategy(strategy_json, market_context=ctx)

Returns:
  {
    "match": True/False,
    "reason": "matched" or "first failing reason",
    "trace": [ ... detailed leaf results ... ]
  }
"""

from typing import Any, Dict, List, Tuple, Union
import math

# ---- Utilities ----
def safe_get(d: Dict, k: str, default=None):
    return d.get(k, default)

def _cmp(a, op: str, b) -> bool:
    """Compare helper supporting float/int/bool/string"""
    try:
        # allow numeric comparison if both convertible to float
        if op == "==":
            return a == b
        if op == "!=":
            return a != b
        # Try numeric comparisons
        a_f = float(a)
        b_f = float(b)
        if op == ">":
            return a_f > b_f
        if op == "<":
            return a_f < b_f
        if op == ">=":
            return a_f >= b_f
        if op == "<=":
            return a_f <= b_f
    except Exception:
        # fallback to python ops
        if op == "==":
            return a == b
        if op == "!=":
            return a != b
        # unsupported types for >/< etc.
        return False
    return False

# ---- Indicator Engine (lightweight) ----
class IndicatorEngine:
    """
    Compute only the indicators you need and expose them as a dict.
    This is designed to be called once per new candle (eg on 1m close)
    and to be fast.
    """

    @staticmethod
    def ema(series: List[float], period: int) -> float:
        # simple EMA on list (last value), fast iterative ok for small windows
        if not series or period <= 0:
            return None
        k = 2.0 / (period + 1.0)
        ema_val = series[0]
        for v in series[1:]:
            ema_val = v * k + ema_val * (1 - k)
        return ema_val

    @staticmethod
    def vwap(ohlc: List[Dict[str, Any]]) -> float:
        # expects list of candles with keys open/high/low/close/volume
        # compute VWAP on provided list (e.g., session or lookback)
        pv_sum = 0.0
        vol_sum = 0.0
        for c in ohlc:
            typical = (c['high'] + c['low'] + c['close']) / 3.0
            pv_sum += typical * c.get('volume', 0)
            vol_sum += c.get('volume', 0)
        return (pv_sum / vol_sum) if vol_sum > 0 else ohlc[-1]['close']

    @staticmethod
    def rvol(ohlc_recent: List[Dict[str, Any]], lookback: int = 20) -> float:
        # RVOL = current volume / avg(volume over lookback)
        if not ohlc_recent:
            return 1.0
        cur_vol = ohlc_recent[-1].get('volume', 0)
        hist = [c.get('volume', 0) for c in ohlc_recent[-lookback:]]
        avg = sum(hist) / len(hist) if hist else 1.0
        return (cur_vol / avg) if avg > 0 else 1.0

    @staticmethod
    def atr(ohlc: List[Dict[str, Any]], period: int = 14) -> float:
        # approximate ATR over provided candle list
        if len(ohlc) < 2:
            return 0.0
        trs = []
        prev_close = ohlc[0]['close']
        for c in ohlc[1:]:
            high = c['high']; low = c['low']; close = c['close']
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
            prev_close = close
        # simple moving average of TRs (fast)
        n = min(period, len(trs))
        return sum(trs[-n:]) / n if trs else 0.0

    @staticmethod
    def volume_spike(ohlc_recent: List[Dict[str,Any]], window: int = 20, mult: float = 1.5) -> bool:
        if not ohlc_recent:
            return False
        cur = ohlc_recent[-1]['volume']
        hist = [c['volume'] for c in ohlc_recent[-window:-0]] if len(ohlc_recent) > 1 else []
        avg = sum(hist) / len(hist) if hist else 0
        # if no history, fallback to false
        if avg <= 0:
            return False
        return cur >= mult * avg

# ---- Pattern Engine (candlestick patterns) ----
class PatternEngine:
    """
    Basic candle pattern detectors operating on a sequence of candles.
    Each function returns True/False. Patterns are intentionally conservative.
    """

    @staticmethod
    def _body_and_wicks(c):
        body = abs(c['close'] - c['open'])
        rng = c['high'] - c['low'] if (c['high'] - c['low']) > 0 else 1e-9
        upper = c['high'] - max(c['open'], c['close'])
        lower = min(c['open'], c['close']) - c['low']
        return body, upper, lower, rng

    @staticmethod
    def hammer(candle):
        body, upper, lower, rng = PatternEngine._body_and_wicks(candle)
        # small upper wick, long lower wick, body near high
        if lower > 2*body and upper < 0.5*body and (body / rng) < 0.6:
            return True
        return False

    @staticmethod
    def pinbar(candle):
        body, upper, lower, rng = PatternEngine._body_and_wicks(candle)
        # one side wick dominates
        if (lower >= 2*body and upper <= 0.2*body) or (upper >= 2*body and lower <= 0.2*body):
            return True
        return False

    @staticmethod
    def bullish_engulfing(prev_c, cur_c):
        # prev bearish, cur bullish and body engulfs
        if prev_c['close'] < prev_c['open'] and cur_c['close'] > cur_c['open']:
            if cur_c['close'] >= prev_c['open'] and cur_c['open'] <= prev_c['close']:
                return True
        return False

    @staticmethod
    def bearish_engulfing(prev_c, cur_c):
        if prev_c['close'] > prev_c['open'] and cur_c['close'] < cur_c['open']:
            if cur_c['open'] >= prev_c['close'] and cur_c['close'] <= prev_c['open']:
                return True
        return False

    @staticmethod
    def doji(candle, thresh=0.1):
        body, upper, lower, rng = PatternEngine._body_and_wicks(candle)
        return (body / rng) <= thresh

    @staticmethod
    def inside_bar(prev_c, cur_c):
        return (cur_c['high'] <= prev_c['high']) and (cur_c['low'] >= prev_c['low'])

# ---- Rule Engine ----
class RuleEngine:
    def __init__(self, indicator_engine: IndicatorEngine = None, pattern_engine: PatternEngine = None):
        self.ind = indicator_engine or IndicatorEngine()
        self.pat = pattern_engine or PatternEngine()

    def eval_leaf(self, leaf: Dict[str, Any], market_context: Dict[str, Any]) -> Tuple[bool, str, Dict]:
        """
        Evaluate a leaf node and return (pass_bool, reason_str, debug_info)
        leaf can be:
          - {"indicator": "ema9", "condition": ">", "value": "ema20"}
          - {"indicator": "rvol", "condition": ">=", "value": 2.0}
          - {"pattern": "bullish_engulfing"}
          - {"indicator": "volume_spike", "condition": "==", "value": true}
        market_context expected to include:
          - 'ohlcv_recent' : list of candle dicts (old...new)
          - 'indicators': dict with precomputed indicators (optional fallback)
          - 'patterns': dict with precomputed pattern flags (optional)
        """
        # Debug container
        dbg = {"leaf": leaf}
        # Pattern leaf
        if 'pattern' in leaf:
            pat = leaf['pattern']
            # allow precomputed pattern or compute on the fly
            if market_context.get('patterns') and pat in market_context['patterns']:
                res = bool(market_context['patterns'][pat])
                return res, f"pattern:{pat}", {"pattern": pat, "value": res}
            # compute minimal patterns
            recent = market_context.get('ohlcv_recent', [])
            if not recent:
                return False, f"pattern:{pat}:no_data", {"pattern": pat}
            if pat == 'hammer':
                res = self.pat.hammer(recent[-1])
            elif pat == 'pinbar':
                res = self.pat.pinbar(recent[-1])
            elif pat == 'bullish_engulfing':
                if len(recent) >= 2:
                    res = self.pat.bullish_engulfing(recent[-2], recent[-1])
                else:
                    res = False
            elif pat == 'bearish_engulfing':
                if len(recent) >= 2:
                    res = self.pat.bearish_engulfing(recent[-2], recent[-1])
                else:
                    res = False
            elif pat == 'doji':
                res = self.pat.doji(recent[-1])
            elif pat == 'inside_bar':
                if len(recent) >= 2:
                    res = self.pat.inside_bar(recent[-2], recent[-1])
                else:
                    res = False
            else:
                # unknown pattern: false
                res = False
            return res, f"pattern:{pat}", {"pattern": pat, "value": res}

        # Indicator leaf
        if 'indicator' in leaf:
            left_key = leaf['indicator']
            left_val = None
            # prefer precomputed indicators dict
            if market_context.get('indicators') and left_key in market_context['indicators']:
                left_val = market_context['indicators'][left_key]
            else:
                # support few built-ins by name
                if left_key == 'close':
                    left_val = market_context.get('ohlcv_recent', [])[-1]['close'] if market_context.get('ohlcv_recent') else None
                elif left_key == 'open':
                    left_val = market_context.get('ohlcv_recent', [])[-1]['open'] if market_context.get('ohlcv_recent') else None
                elif left_key == 'high':
                    left_val = market_context.get('ohlcv_recent', [])[-1]['high'] if market_context.get('ohlcv_recent') else None
                elif left_key == 'low':
                    left_val = market_context.get('ohlcv_recent', [])[-1]['low'] if market_context.get('ohlcv_recent') else None
                elif left_key == 'volume':
                    left_val = market_context.get('ohlcv_recent', [])[-1]['volume'] if market_context.get('ohlcv_recent') else None
                elif left_key == 'vwap':
                    # compute session VWAP if provided candles
                    left_val = self.ind.vwap(market_context.get('ohlcv_for_vwap', market_context.get('ohlcv_recent', [])))
                elif left_key == 'rvol':
                    left_val = self.ind.rvol(market_context.get('ohlcv_recent', []), lookback=leaf.get('lookback', 20))
                elif left_key.startswith('ema'):
                    # emaN: parse N
                    try:
                        n = int(left_key.replace('ema',''))
                        series = [c['close'] for c in market_context.get('ohlcv_recent',[])]
                        left_val = self.ind.ema(series[-(n*3):] if series else series, n)  # small window fallback
                    except Exception:
                        left_val = None
                elif left_key == 'atr':
                    left_val = self.ind.atr(market_context.get('ohlcv_recent',[]), period=leaf.get('period',14))
                elif left_key == 'volume_spike':
                    left_val = self.ind.volume_spike(market_context.get('ohlcv_recent',[]), window=leaf.get('window',20), mult=leaf.get('mult',1.5))
                else:
                    # fallback to indicators dict if present
                    left_val = market_context.get('indicators', {}).get(left_key)
            # right side value
            value = leaf.get('value')
            right_val = None
            if isinstance(value, (int, float, bool)):
                right_val = value
            elif isinstance(value, str):
                # if string and references indicator name, fetch
                if market_context.get('indicators') and value in market_context['indicators']:
                    right_val = market_context['indicators'][value]
                else:
                    # support 'ema20' referenced as string
                    if value.startswith('ema'):
                        try:
                            n = int(value.replace('ema',''))
                            series = [c['close'] for c in market_context.get('ohlcv_recent',[])]
                            right_val = self.ind.ema(series[-(n*3):] if series else series, n)
                        except Exception:
                            right_val = None
                    elif value == 'vwap':
                        right_val = self.ind.vwap(market_context.get('ohlcv_for_vwap', market_context.get('ohlcv_recent', [])))
                    elif value == 'close':
                        right_val = market_context.get('ohlcv_recent', [])[-1]['close'] if market_context.get('ohlcv_recent') else None
                    else:
                        # attempt numeric conversion
                        try:
                            right_val = float(value)
                        except Exception:
                            right_val = market_context.get('indicators', {}).get(value)
            else:
                right_val = None

            cond = leaf.get('condition','==')
            ok = _cmp(left_val, cond, right_val)
            return ok, f"indicator:{left_key}{cond}{value}", {"left": left_val, "right": right_val, "cond": cond}

        # Unknown leaf
        return False, "unknown_leaf", {"leaf": leaf}

    def eval_node(self, node: Dict[str, Any], market_context: Dict[str, Any]) -> Tuple[bool, str, List[Dict]]:
        """
        Recursively evaluate a node. Return (result_bool, reason, trace_list)
        trace_list contains debug records for leaves
        """
        trace = []
        # leaf node if contains indicator or pattern
        if 'indicator' in node or 'pattern' in node:
            ok, reason, info = self.eval_leaf(node, market_context)
            trace.append({"leaf": node, "result": ok, "reason": reason, "info": info})
            return ok, reason, trace

        # composite node expected
        op = node.get('operator', 'AND').upper()
        rules = node.get('rules', [])
        if not rules:
            return False, "empty_rules", []

        if op == 'AND':
            for child in rules:
                ok, reason, child_trace = self.eval_node(child, market_context)
                trace.extend(child_trace)
                if not ok:
                    return False, reason, trace
            return True, "AND_all_true", trace

        elif op == 'OR':
            reasons = []
            for child in rules:
                ok, reason, child_trace = self.eval_node(child, market_context)
                trace.extend(child_trace)
                if ok:
                    return True, "OR_one_true", trace
                reasons.append(reason)
            # none true
            return False, f"OR_none_true: {reasons}", trace

        else:
            # unsupported operator
            return False, f"unsupported_op:{op}", []

    def evaluate_strategy(self, strategy_json: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the strategy logic against market_context.
        Returns dict with keys:
          - match: bool
          - reason: str
          - trace: list (detailed leaf results)
        """
        logic = strategy_json.get('logic')
        if not logic:
            return {"match": False, "reason": "no_logic", "trace": []}
        ok, reason, trace = self.eval_node(logic, market_context)
        return {"match": bool(ok), "reason": reason, "trace": trace}

# ---- Example strategies (same as earlier) ----
EXAMPLE_STRATEGIES = [
    {
      "strategy_name": "VWAP_Reclaim_Long",
      "enabled": True,
      "entry_direction": "LONG",
      "timeframes": ["1m", "5m"],
      "logic": {
        "operator": "AND",
        "rules": [
          {"indicator": "close", "condition": ">", "value": "vwap"},
          {"indicator": "rvol",  "condition": ">=", "value": 2.0},
          {
            "operator": "OR",
            "rules": [
              {"pattern": "bullish_engulfing"},
              {"indicator": "volume_spike", "condition": "==", "value": True}
            ]
          }
        ]
      }
    },
    {
      "strategy_name": "EMA_Stacked_Pullback_Long",
      "enabled": True,
      "entry_direction": "LONG",
      "timeframes": ["1m", "5m"],
      "logic": {
        "operator": "AND",
        "rules": [
          {"indicator": "ema9", "condition": ">", "value": "ema20"},
          {"indicator": "close", "condition": ">", "value": "ema9"},
          {"pattern": "pinbar"}
        ]
      }
    },
    {
      "strategy_name": "ORB_Long",
      "enabled": True,
      "entry_direction": "LONG",
      "timeframes": ["1m", "5m"],
      "logic": {
        "operator": "AND",
        "rules": [
          {"indicator": "time", "condition": ">=", "value": "09:20:00"},
          {"indicator": "close", "condition": ">", "value": "orb_high"},
          {"indicator": "volume_spike", "condition": "==", "value": True},
          {"indicator": "rvol", "condition": ">=", "value": 1.5}
        ]
      }
    },
    {
      "strategy_name": "Breakout_Retest_Long",
      "enabled": True,
      "entry_direction": "LONG",
      "logic": {
        "operator": "AND",
        "rules": [
          {"indicator": "close", "condition": ">", "value": "prev_swing_high"},
          {"indicator": "pullback_depth", "condition": "<=", "value": 0.382},
          {"pattern": "bullish_engulfing"}
        ]
      }
    },
    {
      "strategy_name": "RSI_Divergence_Long",
      "enabled": True,
      "entry_direction": "LONG",
      "logic": {
        "operator": "AND",
        "rules": [
          {"indicator": "bullish_rsi_div", "condition": "==", "value": True},
          {"indicator": "ema9", "condition": ">", "value": "ema20"}
        ]
      }
    },
    {
      "strategy_name": "Volume_Climax_Reversal",
      "enabled": True,
      "entry_direction": "LONG",
      "logic": {
        "operator": "AND",
        "rules": [
          {"indicator": "climax_volume", "condition": "==", "value": True},
          {"pattern": "pinbar"},
          {"indicator": "close", "condition": ">", "value": "ema9"}
        ]
      }
    }
]

# ---- Minimal example runner for local test ----
if __name__ == "__main__":
    # Simple synthetic candles to pass into engine
    candles = []
    import random, time
    base = 100.0
    for i in range(60):
        o = base + random.uniform(-0.5, 0.5)
        c = o + random.uniform(-0.7, 0.7)
        hi = max(o, c) + random.uniform(0, 0.3)
        lo = min(o, c) - random.uniform(0, 0.3)
        v = int(100 + abs(random.gauss(0,1))*200)
        candles.append({'open': o, 'high': hi, 'low': lo, 'close': c, 'volume': v, 'ts': i})
        base = c

    # Simulate indicators
    ind_engine = IndicatorEngine()
    indicators = {
        'ema9': ind_engine.ema([c['close'] for c in candles[-30:]], 9),
        'ema20': ind_engine.ema([c['close'] for c in candles[-60:]], 20),
        'vwap': ind_engine.vwap(candles[-30:]),
        'rvol': ind_engine.rvol(candles[-30:], lookback=20),
        'volume_spike': ind_engine.volume_spike(candles[-30:], window=20, mult=1.5)
    }
    patterns = {}
    pe = PatternEngine()
    if len(candles) >= 2:
        patterns['bullish_engulfing'] = pe.bullish_engulfing(candles[-2], candles[-1])
        patterns['pinbar'] = pe.pinbar(candles[-1])
        patterns['hammer'] = pe.hammer(candles[-1])
        patterns['doji'] = pe.doji(candles[-1])
        patterns['inside_bar'] = pe.inside_bar(candles[-2], candles[-1])

    context = {
        'ohlcv_recent': candles,
        'indicators': indicators,
        'patterns': patterns,
        'ohlcv_for_vwap': candles[-30:]
    }

    engine = RuleEngine(indicator_engine=ind_engine, pattern_engine=pe)
    print("Indicators:", indicators)
    for strat in EXAMPLE_STRATEGIES:
        r = engine.evaluate_strategy(strat, context)
        print(f"Strategy {strat['strategy_name']}: match={r['match']} reason={r['reason']}")
        # optional debug trace
        # import json; print(json.dumps(r['trace'], default=str, indent=2))
