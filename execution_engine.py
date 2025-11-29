"""
execution_engine.py

Simulated Execution Engine (Module 5) for local end-to-end testing.

Listens to:
  SIGNALS.RISK_VALIDATED

Publishes:
  EXECUTION.ORDER_PLACED
  EXECUTION.ORDER_FILLED
  EXECUTION.POSITION_CLOSED

Behavior:
 - Accepts validated plans (entry, sl, tp, qty)
 - Simulates a market fill after a short delay (slippage applied)
 - Keeps a simulated market-price walker task that moves price until TP or SL hit
 - On position close, publishes POSITION_CLOSED and releases intraday lock TG:LOCK:<symbol>

Notes:
 - For integration with real broker, replace simulation logic with REST order calls and real fill/listener logic.
 - Run this module alongside screener.py, mse_subscriber.py, risk_guardian.py.
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from typing import Dict, Any

try:
    import redis.asyncio as aioredis
except Exception:
    aioredis = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------- CONFIG ----------------
CONFIG = {
    "REDIS_URI": "redis://localhost:6379",
    "IN_CHANNEL": "SIGNALS.RISK_VALIDATED",
    "OUT_CHANNEL_ORDER_PLACED": "EXECUTION.ORDER_PLACED",
    "OUT_CHANNEL_ORDER_FILLED": "EXECUTION.ORDER_FILLED",
    "OUT_CHANNEL_POSITION_CLOSED": "EXECUTION.POSITION_CLOSED",
    # Simulation knobs
    "FILL_DELAY_SEC": 0.5,            # delay before simulated fill
    "SLIPPAGE_PTS": 0.0,              # absolute slippage in price units
    "SLIPPAGE_PCT": 0.0005,           # percentage slippage (fraction of price)
    "PRICE_STEP_SEC": 0.3,            # market step interval for price sim
    "PRICE_STEP_VOLatility": 0.0015,  # step volatility fraction of price (std dev)
    "RELEASE_LOCK_ON_CLOSE": True,
}
# ----------------------------------------

class ExecutionEngineSim:
    def __init__(self, cfg):
        self.cfg = cfg
        self.redis = None
        self.pubsub = None
        self.active_tasks = {}  # symbol -> asyncio.Task (monitor)
        self.positions = {}     # local cache: symbol -> position dict

    # ---------------- Redis helpers ----------------
    async def start_redis(self):
        if aioredis is None:
            logging.warning("redis.asyncio not installed; running in console-only mode.")
            self.redis = None
            self.pubsub = None
            return
        self.redis = aioredis.from_url(self.cfg["REDIS_URI"], decode_responses=True)
        try:
            await self.redis.ping()
            logging.info("Connected to Redis at %s", self.cfg["REDIS_URI"])
            self.pubsub = self.redis.pubsub()
            await self.pubsub.subscribe(self.cfg["IN_CHANNEL"])
            logging.info("Subscribed to channel: %s", self.cfg["IN_CHANNEL"])
        except Exception as e:
            logging.warning("Redis connection failed: %s — running console-only.", e)
            self.redis = None
            self.pubsub = None

    async def publish(self, channel: str, payload: Dict[str, Any]):
        payload_json = json.dumps(payload, default=str)
        if self.redis:
            try:
                await self.redis.publish(channel, payload_json)
                logging.info("Published %s %s %s", channel, payload.get("symbol"), payload.get("order_id",""))
                return True
            except Exception as e:
                logging.warning("Failed publish to redis (%s): %s", channel, e)
        # fallback console
        logging.info("PUBLISH[%s] %s", channel, payload_json)
        return False

    # ---------------- Order lifecycle ----------------
    async def handle_validated_order(self, order: Dict[str, Any]):
        """
        order: expected keys: symbol, entry, sl, tp, qty, order_id (optional)
        """
        symbol = order.get("symbol")
        if not symbol:
            logging.warning("Received invalid order without symbol: %s", order)
            return

        entry = float(order.get("entry"))
        sl = float(order.get("sl"))
        tp = float(order.get("tp"))
        qty = int(order.get("qty", 0))
        source = order.get("source", "SIM")
        timestamp = datetime.now(timezone.utc).isoformat()

        # build order_id if not present
        order_id = order.get("order_id") or f"EXE-{symbol}-{int(datetime.now(timezone.utc).timestamp())}"

        # Step 1: publish ORDER_PLACED
        placed = {
            "order_id": order_id,
            "symbol": symbol,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "qty": qty,
            "source": source,
            "status": "PLACED",
            "placed_at": timestamp
        }
        await self.publish(self.cfg["OUT_CHANNEL_ORDER_PLACED"], placed)

        # Step 2: simulate fill delay and slippage
        await asyncio.sleep(self.cfg["FILL_DELAY_SEC"])
        # slippage: combine absolute and percentage
        slip = self.cfg["SLIPPAGE_PTS"] + abs(entry) * self.cfg["SLIPPAGE_PCT"] * (random.uniform(-1,1))
        # choose fill price direction: for longs, slip up; for shorts, slip down. Use sign of qty
        fill_price = entry + slip if qty > 0 else entry - slip

        filled = {
            "order_id": order_id,
            "symbol": symbol,
            "filled_price": round(fill_price, 2),
            "filled_qty": qty,
            "filled_at": datetime.now(timezone.utc).isoformat(),
            "status": "FILLED"
        }
        await self.publish(self.cfg["OUT_CHANNEL_ORDER_FILLED"], filled)

        # store active position
        position = {
            "symbol": symbol,
            "order_id": order_id,
            "entry": float(filled["filled_price"]),
            "sl": sl,
            "tp": tp,
            "qty": qty,
            "opened_at": filled["filled_at"],
            "source": source
        }
        # cache locally
        self.positions[symbol] = position
        # persist to redis for others to read
        if self.redis:
            try:
                await self.redis.hset(f"POS:{symbol}", mapping={k: json.dumps(v, default=str) if isinstance(v,(dict,list)) else v for k,v in position.items()})
            except Exception:
                logging.exception("Failed to persist position to redis.")

        # start monitor task (cancel existing if any)
        if symbol in self.active_tasks:
            task = self.active_tasks[symbol]
            task.cancel()
        task = asyncio.create_task(self._monitor_position(symbol))
        self.active_tasks[symbol] = task

    async def _monitor_position(self, symbol: str):
        """
        Simulate market price moves and close position on TP or SL.
        The simulation uses small random steps around current entry price.
        """
        pos = self.positions.get(symbol)
        if not pos:
            return

        entry = float(pos["entry"])
        sl = float(pos["sl"])
        tp = float(pos["tp"])
        qty = int(pos["qty"])
        # seed simulated price at entry
        price = float(entry)
        step_sec = self.cfg["PRICE_STEP_SEC"]
        volatility = self.cfg["PRICE_STEP_VOLatility"]
        logging.info("Starting price monitor for %s entry=%s sl=%s tp=%s", symbol, entry, sl, tp)

        try:
            while True:
                await asyncio.sleep(step_sec)
                # random step: normal with 0 mean, std = volatility * price
                step = random.gauss(0, 1) * volatility * price
                price = price + step
                # check hits (for longs assume qty>0)
                # for safety allow small epsilon
                if qty > 0:
                    if price >= tp:
                        reason = "TP_HIT"
                    elif price <= sl:
                        reason = "SL_HIT"
                    else:
                        reason = None
                else:
                    # short case
                    if price <= tp:
                        reason = "TP_HIT"
                    elif price >= sl:
                        reason = "SL_HIT"
                    else:
                        reason = None

                if reason:
                    closed = {
                        "symbol": symbol,
                        "order_id": pos["order_id"],
                        "qty": qty,
                        "closed_at": datetime.now(timezone.utc).isoformat(),
                        "close_price": round(price, 2),
                        "reason": reason,
                        "pnl": round((price - entry) * qty, 2)  # simplistic PnL (no fees)
                    }
                    await self.publish(self.cfg["OUT_CHANNEL_POSITION_CLOSED"], closed)
                    # cleanup
                    self.positions.pop(symbol, None)
                    # remove persisted pos
                    if self.redis:
                        try:
                            await self.redis.delete(f"POS:{symbol}")
                        except Exception:
                            pass
                    # release intraday lock if configured
                    if self.cfg.get("RELEASE_LOCK_ON_CLOSE") and self.redis:
                        try:
                            await self.redis.delete(f"TG:LOCK:{symbol}")
                            logging.info("Released TG:LOCK:%s", symbol)
                        except Exception:
                            pass
                    return
                # else continue loop
        except asyncio.CancelledError:
            logging.info("Monitor task for %s cancelled", symbol)
            return
        except Exception as e:
            logging.exception("Monitor error for %s: %s", symbol, e)
            return

    # ---------------- main loop ----------------
    async def run(self):
        await self.start_redis()
        logging.info("Execution Engine (SIM) started, listening for validated orders...")
        if self.pubsub is None:
            # no redis pubsub — console-only mode: nothing to do
            logging.warning("No Redis pubsub available; running in console mode (no input).")
            while True:
                await asyncio.sleep(60)
            return

        try:
            while True:
                msg = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg is None:
                    await asyncio.sleep(0.01)
                    continue
                raw = msg.get("data")
                try:
                    payload = json.loads(raw)
                except Exception:
                    logging.warning("Invalid JSON from risk channel: %s", raw)
                    continue
                logging.info("Received VALIDATED ORDER: %s %s", payload.get("symbol"), payload.get("order_id",""))
                # handle order
                # if payload contains 'qty' and entry etc. pass through
                asyncio.create_task(self.handle_validated_order(payload))

        except asyncio.CancelledError:
            logging.info("Execution Engine cancelled.")
        except Exception as e:
            logging.exception("Main run loop error: %s", e)
        finally:
            # cancel monitor tasks
            for t in list(self.active_tasks.values()):
                t.cancel()
            if self.pubsub:
                await self.pubsub.close()
            if self.redis:
                await self.redis.close()

# ---------------- CLI ----------------
async def main():
    eng = ExecutionEngineSim(CONFIG)
    await eng.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Execution Engine stopped by user.")
