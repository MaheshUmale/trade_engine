"""
event_logger.py

Subscribe to Redis event channels and persist all events to MongoDB collections for auditing.

Runs locally, writes to MongoDB DB 'trading_journal', collections per event type:
  - events.potential_signals
  - events.mse_output
  - events.risk_validated
  - events.execution_order_placed
  - events.execution_order_filled
  - events.execution_position_closed

Usage:
  python event_logger.py
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

# try async redis + motor; fallback to sync
try:
    import redis.asyncio as aioredis
    import motor.motor_asyncio as motor
    ASYNC = True
except Exception:
    import redis
    import pymongo
    ASYNC = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# config
REDIS_URI = "redis://localhost:6379"
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "trading_journal"
CHANNELS = [
    "SIGNALS.POTENTIAL",
    "SIGNALS.MSE_OUTPUT",
    "SIGNALS.RISK_VALIDATED",
    "SIGNALS.RISK_REJECTED",
    "EXECUTION.ORDER_PLACED",
    "EXECUTION.ORDER_FILLED",
    "EXECUTION.POSITION_CLOSED"
]

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

# ---------------- async implementation ----------------
if ASYNC:
    async def run_async():
        # motor client
        mclient = motor.AsyncIOMotorClient(MONGO_URI)
        db = mclient[MONGO_DB]
        r = aioredis.from_url(REDIS_URI, decode_responses=True)
        pub = r.pubsub()
        await pub.subscribe(*CHANNELS)
        logging.info("Subscribed (async) to channels: %s", CHANNELS)
        try:
            while True:
                msg = await pub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg is None:
                    await asyncio.sleep(0.01)
                    continue
                ch = msg.get("channel")
                data = msg.get("data")
                if not data:
                    continue
                try:
                    payload = json.loads(data)
                except Exception:
                    payload = {"raw": data}
                doc = {
                    "channel": ch,
                    "payload": payload,
                    "received_at": _now_iso()
                }
                # safe collection name
                colname = f"events_{ch.replace('.', '_').lower()}"
                await db[colname].insert_one(doc)
                logging.info("Logged event -> %s (%s)", colname, str(payload.get("symbol", payload.get("order_id","-"))))
        except asyncio.CancelledError:
            logging.info("Event logger cancelled")
        finally:
            await pub.close()
            await r.close()
            mclient.close()

    if __name__ == "__main__":
        try:
            asyncio.run(run_async())
        except KeyboardInterrupt:
            logging.info("Stopped (async)")

# ---------------- sync fallback ----------------
else:
    def run_sync():
        r = redis.Redis.from_url(REDIS_URI, decode_responses=True)
        client = pymongo.MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        pub = r.pubsub()
        pub.subscribe(*CHANNELS)
        logging.info("Subscribed (sync) to channels: %s", CHANNELS)
        try:
            for msg in pub.listen():
                if msg is None:
                    continue
                if msg['type'] not in ('message',):
                    continue
                ch = msg['channel']
                data = msg['data']
                try:
                    payload = json.loads(data)
                except Exception:
                    payload = {"raw": data}
                doc = {
                    "channel": ch,
                    "payload": payload,
                    "received_at": _now_iso()
                }
                colname = f"events_{ch.replace('.', '_').lower()}"
                db[colname].insert_one(doc)
                logging.info("Logged event -> %s (%s)", colname, str(payload.get("symbol", payload.get("order_id","-"))))
        except KeyboardInterrupt:
            logging.info("Stopped (sync)")
        finally:
            pub.close()
            client.close()

    if __name__ == "__main__":
        run_sync()
