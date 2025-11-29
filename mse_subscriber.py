"""
mse_subscriber.py

Listens to SIGNALS.POTENTIAL from Redis,
runs Market Structure Engine (MSE / Analyst),
publishes TRADE_PLAN_PROPOSED to SIGNALS.MSE_OUTPUT.

Modules needed:
- Redis (redis.asyncio)
- Your existing MSE code (mse.py)

Usage:
    python mse_subscriber.py
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

# Redis async client
try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

from mse_enhanced import MarketStructureEngine  # <-- your existing MSE module

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------------- CONFIG ----------------
CONFIG = {
    "REDIS_URI": "redis://localhost:6379",
    "IN_CHANNEL": "SIGNALS.POTENTIAL",
    "OUT_CHANNEL": "SIGNALS.MSE_OUTPUT",
    "LOG_CONTEXT": False
}
# ----------------------------------------


class MSESubscriber:

    def __init__(self, config):
        self.config = config
        self.redis = None
        self.pubsub = None
        self.mse = MarketStructureEngine()  # your MSE class instance

    async def start_redis(self):
        if aioredis is None:
            raise RuntimeError("redis.asyncio is not installed.")

        self.redis = aioredis.from_url(self.config["REDIS_URI"], decode_responses=True)
        await self.redis.ping()
        logging.info(f"Connected to Redis at {self.config['REDIS_URI']}")

        self.pubsub = self.redis.pubsub()
        await self.pubsub.subscribe(self.config["IN_CHANNEL"])
        logging.info(f"Subscribed to Redis channel: {self.config['IN_CHANNEL']}")

    # ---------------- PUBLISH TRADE_PLAN ----------------
    async def publish_mse_output(self, payload: dict):
        payload_json = json.dumps(payload, default=str)
        await self.redis.publish(self.config["OUT_CHANNEL"], payload_json)
        logging.info(f"Published TRADE_PLAN_PROPOSED for {payload.get('symbol')}")

    # ---------------- MAIN LOOP ----------------
    async def run(self):
        await self.start_redis()
        logging.info("MSE Subscriber started. Waiting for signals...")

        try:
            while True:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is None:
                    await asyncio.sleep(0.01)
                    continue

                raw = message.get("data")
                try:
                    signal = json.loads(raw)
                except:
                    logging.warning(f"Invalid JSON: {raw}")
                    continue

                logging.info(f"Received POTENTIAL_SIGNAL: {signal}")

                # Extract fields
                symbol = signal.get("symbol")
                ltp = float(signal.get("ltp"))
                strategy = signal.get("strategy")

                # ---------------- RUN MSE ----------------
                try:
                    mse_result = await self.mse.process_symbol(symbol, ltp)
                except Exception as e:
                    logging.exception(f"MSE error for {symbol}: {e}")
                    continue

                if mse_result is None:
                    logging.info(f"MSE returned None — skipping plan for {symbol}")
                    continue

                # ---------------- BUILD TRADE_PLAN_PROPOSED ----------------
                trade_plan = {
                    "symbol": symbol,
                    "strategy": strategy,
                    "entry": mse_result["entry"],
                    "sl": mse_result["sl"],
                    "tp": mse_result["tp"],
                    "rr": mse_result.get("rr"),
                    "blockers": mse_result.get("blockers", []),
                    "mse_context": mse_result.get("mse_context") if self.config["LOG_CONTEXT"] else None,
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }

                # Publish to next stage
                await self.publish_mse_output(trade_plan)

        except asyncio.CancelledError:
            logging.info("MSE Subscriber cancelled.")
        except Exception as e:
            logging.exception(f"Subscriber error: {e}")
        finally:
            await self.pubsub.close()
            if self.redis:
                await self.redis.close()


async def main():
    sub = MSESubscriber(CONFIG)
    await sub.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("MSE subscriber stopped.")
