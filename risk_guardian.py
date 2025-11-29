"""
risk_guardian.py

Risk Guardian Subscriber

- Subscribes to Redis channel SIGNALS.MSE_OUTPUT
- Validates incoming trade plans (R:R, SL distance, risk per trade)
- Computes position sizing (quantity) based on account size & per-trade risk
- Enforces intraday symbol lock to prevent opening a second trade on the same symbol
- Publishes approved orders to SIGNALS.RISK_VALIDATED
- Publishes rejected plans to SIGNALS.RISK_REJECTED (with reason)
- Uses timezone-aware datetimes

Requires:
    pip install redis

If redis.asyncio not installed, runs in "console mode" and prints messages.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from math import floor

# Optional: async redis
try:
    import redis.asyncio as aioredis
except Exception:
    aioredis = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------- CONFIG ----------------
CONFIG = {
    "REDIS_URI": "redis://localhost:6379",
    "IN_CHANNEL": "SIGNALS.MSE_OUTPUT",
    "OUT_CHANNEL_APPROVED": "SIGNALS.RISK_VALIDATED",
    "OUT_CHANNEL_REJECTED": "SIGNALS.RISK_REJECTED",
    # Risk params (tweak for your account)
    "ACCOUNT_BALANCE": 200000.0,   # INR or account currency
    "RISK_PER_TRADE_PCT": 0.01,    # 1% risk of account per trade
    "MIN_RR": 2.0,                 # minimum reward:risk required
    "MIN_SL_DISTANCE_PTS": 2.0,    # absolute minimum SL distance in price points (avoid zero)
    "INTRADAY_LOCK_TTL_SEC": 24 * 3600,  # keep lock for symbol for the trading day (sec)
    # Instrument specifics (if futures/options, adapt lot_size)
    "DEFAULT_LOT_SIZE": 1,         # number of underlying units per contract (set to 1 for stocks)
    "MIN_QTY": 1,                  # minimum quantity to place
}
# ----------------------------------------

class RiskGuardian:
    def __init__(self, config):
        self.config = config
        self.redis = None
        self.pubsub = None

    async def start_redis(self):
        if aioredis is None:
            logging.warning("redis.asyncio not installed — running in console-only mode.")
            self.redis = None
            return
        self.redis = aioredis.from_url(self.config["REDIS_URI"], decode_responses=True)
        try:
            await self.redis.ping()
            logging.info("Connected to Redis at %s", self.config["REDIS_URI"])
            self.pubsub = self.redis.pubsub()
            await self.pubsub.subscribe(self.config["IN_CHANNEL"])
            logging.info("Subscribed to channel: %s", self.config["IN_CHANNEL"])
        except Exception as e:
            logging.warning("Failed Redis connection: %s — running console-only.", e)
            self.redis = None
            self.pubsub = None

    async def publish(self, channel: str, payload: dict):
        payload_json = json.dumps(payload, default=str)
        if self.redis:
            try:
                await self.redis.publish(channel, payload_json)
                logging.info("Published to %s: %s %s", channel, payload.get("symbol"), payload.get("order_id", ""))
                return True
            except Exception as e:
                logging.warning("Failed publish to redis (%s): %s", channel, e)
                # fallback to console
        # Console fallback
        logging.info("PUBLISH[%s] %s", channel, payload_json)
        return False

    async def _acquire_intraday_lock(self, symbol: str) -> bool:
        """
        Ensure there is no active trade for this symbol.
        Implemented via Redis SETNX with TTL.
        Returns True if lock acquired (no active trade), False if already locked.
        """
        lock_key = f"TG:LOCK:{symbol}"
        ttl = self.config["INTRADAY_LOCK_TTL_SEC"]
        if self.redis is None:
            # simple local memory based lock: we won't implement local persistent lock in this version
            # to keep behavior predictable we return True (no lock) — caller should be careful
            return True
        try:
            # SET key NX EX ttl
            res = await self.redis.set(lock_key, "1", nx=True, ex=ttl)
            return bool(res)
        except Exception as e:
            logging.warning("Lock acquisition error for %s: %s", symbol, e)
            # safe fallback: disallow to avoid double risk
            return False

    async def _release_intraday_lock(self, symbol: str):
        """Remove the lock if needed (we may choose to keep for full day)."""
        if self.redis:
            try:
                await self.redis.delete(f"TG:LOCK:{symbol}")
            except Exception:
                pass

    def _compute_quantity(self, entry: float, sl: float) -> int:
        """
        Computes quantity based on account risk and SL distance.
        qty = floor( (account_balance * risk_pct) / abs(entry - sl) )
        Applies minimum lot size and MIN_QTY.
        """
        acc = self.config["ACCOUNT_BALANCE"]
        risk_pct = self.config["RISK_PER_TRADE_PCT"]
        max_risk = acc * risk_pct
        tick_risk = abs(entry - sl)
        if tick_risk < self.config["MIN_SL_DISTANCE_PTS"]:
            # avoid extremely tiny SL; treat as invalid
            return 0
        raw_qty = max_risk / tick_risk
        # adjust for lot size if needed (for futures, multiply by lot)
        lot = self.config.get("DEFAULT_LOT_SIZE", 1)
        # floor to nearest integer contract / share
        qty = floor(raw_qty / lot) * lot
        if qty < self.config.get("MIN_QTY", 1):
            return 0
        return int(qty)

    def _validate_basic_structure(self, plan: dict) -> (bool, str):
        """
        Basic validation of MSE plan presence/shape.
        Expects fields: symbol, entry, sl, tp, rr
        """
        required = ["symbol", "entry", "sl", "tp", "rr"]
        for r in required:
            if r not in plan or plan[r] is None:
                return False, f"Missing_required_field:{r}"
        try:
            entry = float(plan["entry"])
            sl = float(plan["sl"])
            tp = float(plan["tp"])
            rr = float(plan["rr"])
        except Exception:
            return False, "Invalid_numeric_field"
        if rr < self.config["MIN_RR"]:
            return False, f"R:R_too_low:{rr:.2f}<{self.config['MIN_RR']}"
        if abs(entry - sl) < self.config["MIN_SL_DISTANCE_PTS"]:
            return False, "SL_too_close"
        return True, "ok"

    async def handle_mse_plan(self, plan: dict):
        """
        Validate and possibly approve an MSE trade plan.
        On approval publish VALIDATED_ORDER_REQUEST, else publish rejection.
        """
        # plan may come wrapped e.g. {'symbol':.., 'valid':True, 'entry':..., ...}
        symbol = plan.get("symbol")
        if symbol is None:
            logging.warning("Received MSE payload without symbol: %s", plan)
            return

        # basic structural validation
        ok, reason = self._validate_basic_structure(plan)
        if not ok:
            reject_payload = {
                "symbol": symbol,
                "reason": reason,
                "mse_payload": plan,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self.publish(self.config["OUT_CHANNEL_REJECTED"], reject_payload)
            return

        # Intraday lock — prevent second trade on same symbol
        lock_ok = await self._acquire_intraday_lock(symbol)
        if not lock_ok:
            reject_payload = {
                "symbol": symbol,
                "reason": "intraday_lock_active",
                "mse_payload": plan,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self.publish(self.config["OUT_CHANNEL_REJECTED"], reject_payload)
            return

        # compute qty
        entry = float(plan["entry"])
        sl = float(plan["sl"])
        tp = float(plan["tp"])
        rr = float(plan["rr"])
        qty = self._compute_quantity(entry, sl)

        if qty <= 0:
            # cannot size trade within risk budget
            # release lock since we didn't create a trade
            await self._release_intraday_lock(symbol)
            reject_payload = {
                "symbol": symbol,
                "reason": "insufficient_qty_for_risk",
                "computed_qty": qty,
                "mse_payload": plan,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self.publish(self.config["OUT_CHANNEL_REJECTED"], reject_payload)
            return

        # Optional: additional checks (max notional, max open trades, etc.)
        max_notional = self.config.get("MAX_NOTIONAL_PER_TRADE")
        if max_notional:
            if qty * entry > max_notional:
                await self._release_intraday_lock(symbol)
                reject_payload = {
                    "symbol": symbol,
                    "reason": "exceeds_max_notional",
                    "computed_qty": qty,
                    "notional": qty * entry,
                    "mse_payload": plan,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                await self.publish(self.config["OUT_CHANNEL_REJECTED"], reject_payload)
                return

        # If we get here, trade is approved
        validated = {
            "symbol": symbol,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "qty": qty,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mse_context": plan.get("mse_context"),
            "source": "MSE",
            "order_id": f"VG-{symbol}-{int(datetime.now(timezone.utc).timestamp())}"
        }
        await self.publish(self.config["OUT_CHANNEL_APPROVED"], validated)
        # Note: keep the intraday lock until manual release or order exit logic runs.
        # The execution engine should release the lock when the trade is fully closed (or Risk Guardian can monitor and release).

    async def run(self):
        await self.start_redis()
        logging.info("Risk Guardian running, listening for MSE output...")

        if self.pubsub is None:
            # Console-only fallback: do nothing (or could poll a file/stdin)
            logging.warning("No Redis pubsub; Risk Guardian waiting in console mode (no input).")
            # In fallback mode, exit or sleep
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
                    logging.warning("Invalid JSON from MSE channel: %s", raw)
                    continue
                logging.info("Received MSE plan: %s %s", payload.get("symbol"), payload.get("valid", ""))
                # If payload indicates invalid plan from MSE, publish rejection and skip
                if payload.get("valid") is False:
                    reject = {
                        "symbol": payload.get("symbol"),
                        "reason": payload.get("reason") or "mse_invalid",
                        "mse_payload": payload,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    await self.publish(self.config["OUT_CHANNEL_REJECTED"], reject)
                    continue

                # Otherwise handle plan
                await self.handle_mse_plan(payload)

        except asyncio.CancelledError:
            logging.info("Risk Guardian cancelled.")
        except Exception as e:
            logging.exception("Risk Guardian loop error: %s", e)
        finally:
            if self.pubsub:
                await self.pubsub.close()
            if self.redis:
                await self.redis.close()

# ----------------- Entrypoint -----------------
async def main():
    rg = RiskGuardian(CONFIG)
    await rg.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Risk Guardian stopped by user.")
