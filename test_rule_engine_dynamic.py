import unittest
from rule_engine import RuleEngine, IndicatorEngine

class TestRuleEngineDynamic(unittest.TestCase):
    def setUp(self):
        self.engine = RuleEngine()
        # Create synthetic OHLC data
        self.ohlc = []
        base = 100.0
        for i in range(50):
            self.ohlc.append({
                'open': base, 'high': base+2, 'low': base-2, 'close': base+1, 'volume': 1000
            })
            base += 1

    def test_dynamic_dispatch(self):
        # Test calling a method dynamically via strategy JSON
        # donchian_high(period=10)
        strategy = {
            "logic": {
                "indicator": "donchian_high(period=10)",
                "condition": ">",
                "value": 0
            }
        }
        ctx = {'ohlcv_recent': self.ohlc}
        res = self.engine.evaluate_strategy(strategy, ctx)
        self.assertTrue(res['match'])
        # Check trace to see if it actually called the method
        trace = res['trace'][0]
        self.assertIsNotNone(trace['info']['left'])
        print(f"Donchian High: {trace['info']['left']}")

    def test_bollinger(self):
        strategy = {
            "logic": {
                "operator": "AND",
                "rules": [
                    {"indicator": "bollinger_upper(period=20, std_dev=2.0)", "condition": ">", "value": "close"},
                    {"indicator": "bollinger_lower", "condition": "<", "value": "close"}
                ]
            }
        }
        ctx = {'ohlcv_recent': self.ohlc}
        res = self.engine.evaluate_strategy(strategy, ctx)
        # We just want to ensure it runs without error and returns values
        self.assertIsNotNone(res['trace'][0]['info']['left'])
        print(f"Bollinger Upper: {res['trace'][0]['info']['left']}")

    def test_keltner(self):
        strategy = {
            "logic": {
                "indicator": "keltner_mid(period=20)",
                "condition": ">",
                "value": 0
            }
        }
        ctx = {'ohlcv_recent': self.ohlc}
        res = self.engine.evaluate_strategy(strategy, ctx)
        self.assertIsNotNone(res['trace'][0]['info']['left'])
        print(f"Keltner Mid: {res['trace'][0]['info']['left']}")

if __name__ == '__main__':
    unittest.main()
