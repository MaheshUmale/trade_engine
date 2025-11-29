from typing import Any, Dict, List, Tuple
import pandas as pd
import pandas_ta as ta

# ---- Indicator Engine (Vectorized) ----
class IndicatorEngine:
    def __init__(self):
        self.indicators = {}

    def get_indicator_values_at_candle(self, i: int) -> Dict:
        return {k: v.iloc[i] for k, v in self.indicators.items()}

    def compute_all_indicators(self, df: pd.DataFrame, strategies: List[Dict]):
        self.indicators = {}

        # --- Compute indicators individually ---
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.vwap(append=True)
        df.ta.rsi(append=True)
        df.ta.atr(append=True)
        df.ta.bbands(append=True)

        # --- Rename columns to match strategy keys ---
        rename_map = {
            "EMA_9": "ema9",
            "EMA_20": "ema20",
            "RSI_14": "rsi",
            "ATRr_14": "atr",
            "BBu_20_2.0": "bollinger_upper",
            "BBl_20_2.0": "bollinger_lower"
        }
        df = df.rename(columns=rename_map)

        # Handle VWAP column name, which can be 'VWAP_D' or 'VWAP'
        if 'VWAP_D' in df.columns:
            df = df.rename(columns={'VWAP_D': 'vwap'})
        elif 'VWAP' in df.columns:
            df = df.rename(columns={'VWAP': 'vwap'})

        # --- Custom indicators ---
        df['rvol'] = df['volume'] / df['volume'].rolling(window=20).mean()
        df['volume_spike'] = df['volume'] > df['volume'].rolling(window=20).mean() * 1.5

        # --- Store all computed columns ---
        for col in df.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                self.indicators[col] = df[col]

        self.indicators.update({
            'open': df['open'], 'high': df['high'], 'low': df['low'],
            'close': df['close'], 'volume': df['volume']
        })

# ---- Pattern Engine ----
class PatternEngine:
    def bullish_engulfing(self, df):
        prev = df.shift(1)
        return (prev['close'] < prev['open']) & (df['close'] > df['open']) & \
               (df['close'] >= prev['open']) & (df['open'] <= prev['close'])

    def pinbar(self, df):
        body = abs(df['close'] - df['open'])
        rng = df['high'] - df['low']
        upper = df['high'] - df[['open', 'close']].max(axis=1)
        lower = df[['open', 'close']].min(axis=1) - df['low']
        return ((lower >= 2 * body) & (upper <= 0.2 * body)) | \
               ((upper >= 2 * body) & (lower <= 0.2 * body))

    def compute_all_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        return {
            "bullish_engulfing": self.bullish_engulfing(df),
            "pinbar": self.pinbar(df)
        }

# ---- Rule Engine ----
class RuleEngine:
    def eval_leaf(self, leaf: Dict, indicators: Dict, patterns: Dict) -> bool:
        if 'pattern' in leaf:
            return patterns.get(leaf['pattern'], False)

        if 'indicator' in leaf:
            left_val = indicators.get(leaf['indicator'])
            value = leaf.get('value')
            right_val = value if not isinstance(value, str) else indicators.get(value)

            if left_val is None or right_val is None: return False

            op = leaf.get('condition', '==')
            if op == ">": return left_val > right_val
            if op == "<": return left_val < right_val
            if op == ">=": return left_val >= right_val
            if op == "<=": return left_val <= right_val
            if op == "==": return left_val == right_val
        return False

    def eval_node(self, node: Dict, indicators: Dict, patterns: Dict) -> bool:
        if 'indicator' in node or 'pattern' in node:
            return self.eval_leaf(node, indicators, patterns)

        op = node.get('operator', 'AND').upper()
        rules = node.get('rules', [])

        if op == 'AND': return all(self.eval_node(r, indicators, patterns) for r in rules)
        if op == 'OR': return any(self.eval_node(r, indicators, patterns) for r in rules)
        return False

    def evaluate_strategy(self, strategy: Dict, indicators: Dict, patterns: Dict) -> bool:
        return self.eval_node(strategy.get('logic', {}), indicators, patterns)
