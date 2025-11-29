"""
screener.py

Async Screener Module (Module 2) for the intraday trading engine.
- Loads strategy JSON rules from MongoDB (or local file fallback)
- Aggregates ticks to 1-min candles
- Computes indicators & patterns using RuleEngine helpers
- Evaluates strategies with RuleEngine
- Publishes POTENTIAL_SIGNAL to Redis channel SIGNALS.POTENTIAL

Usage:
    python screener.py

Configuration:
    - Edit CONFIG at top for Redis/Mongo URIs and symbol list.
    - Replace UpstoxWSSStub with a real WSS client later (implements same async iterator interface).
"""

import asyncio
import json
import logging
import os
from collections import defaultdict, deque
from datetime import datetime, timedelta,timezone,UTC
from typing import Dict, List, Any

# Attempt imports for async redis and motor (Mongo). Provide fallbacks if not installed.
try:
    import redis.asyncio as aioredis
except Exception:
    aioredis = None

try:
    import motor.motor_asyncio as motor
except Exception:
    motor = None

# Import RuleEngine (assumes rule_engine.py in same folder)
from rule_engine import RuleEngine, IndicatorEngine, PatternEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------- CONFIG ----------------
CONFIG = {
    "SYMBOLS": ["BANKNIFTY"],               # list of symbols to watch
    "REDIS_URI": "redis://localhost:6379",
    "REDIS_CHANNEL": "SIGNALS.POTENTIAL",
    "MONGO_URI": "mongodb://localhost:27017",
    "MONGO_DB": "trading",
    "MONGO_COLLECTION": "strategies",
    "STRATEGIES_LOCAL_FILE": "strategies.json",  # fallback file if Mongo not available
    "CANDLE_TF_SECONDS": 60,  # 1-minute candles
    "AGG_WINDOW": 500,        # number of historical candles to keep in memory per symbol
    "LOG_MATCH_VERBOSE": False
}
# ----------------------------------------

# ----------------- Helpers & Fallbacks -----------------
async def load_strategies_from_mongo(mongo_uri: str, db_name: str, collection_name: str) -> List[Dict]:
    """Load strategies from MongoDB (async)."""
    if motor is None:
        logging.warning("motor not installed; cannot load from MongoDB.")
        return []
    client = motor.AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    coll = db[collection_name]
    docs = []
    async for doc in coll.find({"enabled": True}):
        doc.pop("_id", None)
        docs.append(doc)
    return docs

def load_strategies_from_file(path: str) -> List[Dict]:
    """Fallback: load strategies from a local JSON file."""
    if not os.path.exists(path):
        logging.warning("No strategies file found at %s", path)
        return []
    with open(path, "r") as f:
        data = json.load(f)
    # expect list of strategy dicts
    return [s for s in data if s.get("enabled", True)]

# ----------------- Upstox WSS Stub -----------------
class UpstoxWSSStub:
    """
    Async stub that mimics a WSS client producing tick events.
    Each tick is a dict:
      {"symbol": "BANKNIFTY", "ts": datetime, "price": float, "size": int}
    This stub emits synthetic ticks for demo; you can replace with real Upstox WSS implementation.
    """
    def __init__(self, symbols: List[str], tick_interval_sec: float = 0.2):
        self.symbols = symbols
        self.tick_interval = tick_interval_sec
        self.running = True

    async def connect(self):
        logging.info("UpstoxWSSStub connecting (simulated)...")
        await asyncio.sleep(0.01)

    async def __aiter__(self):
        """Allows: async for tick in client: """
        # naive synthetic price generator
        import random
        base_prices = {s: 10000.0 + random.uniform(-50, 50) for s in self.symbols}
        while self.running:
            await asyncio.sleep(self.tick_interval)
            now = datetime.now(timezone.utc)
            # emit a tick per symbol (round robin)
            for s in self.symbols:
                # small random walk
                base_prices[s] += random.uniform(-1, 1)
                tick = {
                    "symbol": s,
                    "ts": now,
                    "price": float(round(base_prices[s], 2)),
                    "size": int(abs(random.gauss(0, 1)) * 100)
                }
                yield tick

    async def close(self):
        self.running = False
        logging.info("UpstoxWSSStub closed.")

# ----------------- Candle Aggregator -----------------
class CandleAggregator:
    """
    Aggregate streaming ticks into 1-minute candles (or tf configured).
    Maintains a rolling buffer of last N candles per symbol.
    """
    def __init__(self, tf_seconds: int = 60, maxlen: int = 500):
        self.tf = tf_seconds
        self.buffers = {}  # symbol -> current candle dict being built
        self.history = defaultdict(lambda: deque(maxlen=maxlen))  # symbol -> deque(candles)

    def _candle_period_start(self, ts: datetime) -> datetime:
        # Align to tf seconds
        secs = int(ts.timestamp())
        period_start = secs - (secs % self.tf)
        return datetime.fromtimestamp(period_start, UTC) #datetime.utcfromtimestamp()

    def ingest_tick(self, tick: Dict[str, Any]):
        """
        Ingest a tick. Return completed_candle if a candle finished, else None.
        Candle dict keys: datetime (period start), open, high, low, close, volume
        """
        sym = tick["symbol"]
        ts = tick["ts"]
        price = tick["price"]
        size = tick.get("size", 0)
        period = self._candle_period_start(ts)

        cur = self.buffers.get(sym)
        if cur is None or cur["datetime"] != period:
            # close previous candle (if exists)
            completed = None
            if cur is not None:
                completed = cur.copy()
                self.history[sym].append(completed)
            # start new candle
            self.buffers[sym] = {"datetime": period, "open": price, "high": price, "low": price, "close": price, "volume": size}
            return completed
        else:
            # update current candle
            cur["high"] = max(cur["high"], price)
            cur["low"] = min(cur["low"], price)
            cur["close"] = price
            cur["volume"] += size
            return None

    def get_recent_candles(self, symbol: str, n: int = 200) -> List[Dict]:
        """Return list of candles including current incomplete candle if exists (most recent last)."""
        items = list(self.history[symbol])
        if symbol in self.buffers:
            # append a snapshot of current candle
            items.append(self.buffers[symbol].copy())
        return items[-n:]

# ----------------- Screener Core -----------------
class Screener:
    def __init__(self, config: Dict):
        self.config = config
        self.symbols = config["SYMBOLS"]
        self.agg = CandleAggregator(tf_seconds=config["CANDLE_TF_SECONDS"], maxlen=config["AGG_WINDOW"])
        self.rule_engine = RuleEngine()
        self.ind_engine = IndicatorEngine()
        self.pat_engine = PatternEngine()
        self.redis = None
        self.mongo_client = None
        self.strategies = []  # in-memory list of strategy dicts
        self.redis_channel = config["REDIS_CHANNEL"]

    # ----------------- External connectors -----------------
    async def start_redis(self):
        if aioredis is None:
            logging.warning("redis.asyncio not installed; Redis publisher disabled.")
            self.redis = None
            return
        self.redis = aioredis.from_url(self.config["REDIS_URI"], decode_responses=True)
        # Test connection
        try:
            await self.redis.ping()
            logging.info("Connected to Redis at %s", self.config["REDIS_URI"])
        except Exception as e:
            logging.warning("Failed to connect to Redis: %s", e)
            self.redis = None

    async def start_mongo(self):
        if motor is None:
            logging.warning("motor (Mongo async) not installed; using local strategies file.")
            self.mongo_client = None
            return
        try:
            client = motor.AsyncIOMotorClient(self.config["MONGO_URI"])
            # quick ping
            await client.admin.command('ping')
            self.mongo_client = client
            logging.info("Connected to MongoDB at %s", self.config["MONGO_URI"])
        except Exception as e:
            logging.warning("Failed to connect to MongoDB: %s", e)
            self.mongo_client = None

    async def load_strategies(self):
        """Load strategies, prefer Mongo then fallback file."""
        if self.mongo_client:
            try:
                db = self.mongo_client[self.config["MONGO_DB"]]
                coll = db[self.config["MONGO_COLLECTION"]]
                docs = []
                async for doc in coll.find({"enabled": True}):
                    doc.pop("_id", None)
                    docs.append(doc)
                self.strategies = docs
                logging.info("Loaded %d strategies from MongoDB", len(docs))
                return
            except Exception as e:
                logging.warning("Error loading strategies from Mongo: %s", e)
        # fallback to file
        file_path = self.config["STRATEGIES_LOCAL_FILE"]
        self.strategies = load_strategies_from_file(file_path)
        logging.info("Loaded %d strategies from local file %s", len(self.strategies), file_path)

    # ----------------- Publishing signals -----------------
    async def publish_signal(self, payload: Dict[str, Any]):
        payload_json = json.dumps(payload, default=str)
        if self.redis:
            try:
                await self.redis.publish(self.redis_channel, payload_json)
                logging.info("Published POTENTIAL_SIGNAL %s %s", payload.get("symbol"), payload.get("strategy"))
                return True
            except Exception as e:
                logging.warning("Failed to publish to Redis: %s", e)
        # fallback: print to log
        logging.info("POTENTIAL_SIGNAL (local): %s", payload_json)
        return False

    # ----------------- Core tick handler -----------------
    async def handle_tick(self, tick: Dict[str, Any]):
        """Process incoming tick: aggregate, and when candle completes evaluate strategies."""
        completed_candle = self.agg.ingest_tick(tick)
        if completed_candle:
            sym = completed_candle.get("symbol") if "symbol" in completed_candle else None
            # The aggregator returns completed candle but doesn't attach symbol; we can infer by checking buffers.
            # Simpler: use tick.symbol for symbol that closed.
            sym = tick["symbol"]
            # get recent candles for symbol
            recent = self.agg.get_recent_candles(sym, n=200)
            # build market_context: ohlcv_recent list of dicts with open/high/low/close/volume
            market_context = {
                "ohlcv_recent": recent,
                "ohlcv_for_vwap": recent[-120:],  # use last 120 bars for VWAP if needed
                "indicators": {},
                "patterns": {}
            }
            # compute common indicators once per candle
            try:
                # EMA9/EMA20
                closes = [c['close'] for c in recent]
                if len(closes) >= 9:
                    market_context['indicators']['ema9'] = self.ind_engine.ema(closes[-30:], 9)
                if len(closes) >= 20:
                    market_context['indicators']['ema20'] = self.ind_engine.ema(closes[-60:], 20)
                # VWAP
                market_context['indicators']['vwap'] = self.ind_engine.vwap(market_context['ohlcv_for_vwap'])
                # RVOL
                market_context['indicators']['rvol'] = self.ind_engine.rvol(recent, lookback=20)
                # ATR
                market_context['indicators']['atr'] = self.ind_engine.atr(recent, period=14)
                # volume_spike
                market_context['indicators']['volume_spike'] = self.ind_engine.volume_spike(recent, window=20, mult=1.5)
            except Exception as e:
                logging.exception("Indicator compute error: %s", e)

            # compute patterns
            try:
                if len(recent) >= 2:
                    market_context['patterns']['bullish_engulfing'] = self.pat_engine.bullish_engulfing(recent[-2], recent[-1])
                    market_context['patterns']['bearish_engulfing'] = self.pat_engine.bearish_engulfing(recent[-2], recent[-1])
                    market_context['patterns']['pinbar'] = self.pat_engine.pinbar(recent[-1])
                    market_context['patterns']['hammer'] = self.pat_engine.hammer(recent[-1])
                    market_context['patterns']['doji'] = self.pat_engine.doji(recent[-1])
                    market_context['patterns']['inside_bar'] = self.pat_engine.inside_bar(recent[-2], recent[-1])
            except Exception as e:
                logging.exception("Pattern compute error: %s", e)

            # Evaluate strategies (only those enabled & matching timeframe logic if specified)
            for strat in self.strategies:
                try:
                    # basic TF filter: if strategy requires 5m but aggregator is 1m, we still evaluate since MSE pipeline may pass multi-TF.
                    # For now assume strategies in list are compatible.
                    result = self.rule_engine.evaluate_strategy(strat, market_context)
                    if result.get("match"):
                        # craft POTENTIAL_SIGNAL payload
                        payload = {
                            "symbol": sym,
                            "strategy": strat.get("strategy_name"),
                            "signal_time": datetime.now(timezone.utc).isoformat(),
                            "ltp": recent[-1]['close'],
                            "direction": strat.get("entry_direction", "BOTH"),
                            "confidence": "rule_match",
                            "trace": result.get("trace") if self.config.get("LOG_MATCH_VERBOSE") else None,
                            "indicators": market_context['indicators']
                        }
                        await self.publish_signal(payload)
                except Exception as e:
                    logging.exception("Strategy eval error for %s: %s", strat.get("strategy_name"), e)

    # ----------------- Main runner -----------------
    async def run(self, wss_client):
        # start connectors
        await self.start_redis()
        await self.start_mongo()
        await self.load_strategies()

        # connect to WSS
        await wss_client.connect()
        logging.info("Screener started. Listening to ticks...")

        # iterate ticks
        try:
            async for tick in wss_client:
                # attach symbol info for completed candle closure detection
                # ingest tick and evaluate
                await self.handle_tick(tick)
        except asyncio.CancelledError:
            logging.info("Screener run cancelled")
        except Exception as e:
            logging.exception("Error in WSS loop: %s", e)
        finally:
            await wss_client.close()
            if self.redis:
                await self.redis.close()

# ----------------- Entry point -----------------
async def main():
    screener = Screener(CONFIG)
    # use stub; replace with real upstox client when ready
    wss = UpstoxWSSStub(CONFIG["SYMBOLS"], tick_interval_sec=0.05)
    # ensure strategies.json exists with some example strategies if mongo not used
    if not os.path.exists(CONFIG["STRATEGIES_LOCAL_FILE"]):
        # create example strategies file from rule_engine.EXAMPLE_STRATEGIES
        try:
            from rule_engine import EXAMPLE_STRATEGIES
            with open(CONFIG["STRATEGIES_LOCAL_FILE"], "w") as f:
                json.dump(EXAMPLE_STRATEGIES, f, indent=2)
            logging.info("Wrote example strategies to %s", CONFIG["STRATEGIES_LOCAL_FILE"])
        except Exception:
            pass
    # run screener
    await screener.run(wss)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Screener terminated by user")
