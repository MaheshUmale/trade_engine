"""
dashboard.py

Simple Streamlit UI showing:
 - Live positions (from Redis POS:<symbol>)
 - Recent events (from MongoDB trading_journal.events_* collections)

Run:
  streamlit run dashboard.py
"""

import streamlit as st
import time
import json
from datetime import datetime, timezone

# sync clients (Streamlit runs synchronously)
import redis
import pymongo

# config
REDIS_URI = "redis://localhost:6379"
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "trading_journal"
WATCH_SYMBOLS = ["BANKNIFTY"]  # expand as needed

r = redis.Redis.from_url(REDIS_URI, decode_responses=True)
mongo = pymongo.MongoClient(MONGO_URI)
db = mongo[MONGO_DB]

def now_iso():
    return datetime.now(timezone.utc).isoformat()

st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("Intraday Trading Dashboard — Local")

col1, col2 = st.columns([1,2])

with col1:
    st.header("Live Positions")
    pos_placeholder = st.empty()
    st.write("Redis keys: POS:<SYMBOL>")

with col2:
    st.header("Recent Events (Mongo)")
    events_placeholder = st.empty()

# event selection
collections = sorted([c for c in db.list_collection_names() if c.startswith("events_")])
sel = st.selectbox("Choose event collection", options=collections or ["events_signals_potential"])

REFRESH = st.sidebar.slider("Refresh (seconds)", 2, 10, 3)

def fetch_positions():
    rows = []
    # look for keys POS:<symbol> or scan pattern
    keys = r.keys("POS:*") if r else []
    for k in keys:
        try:
            h = r.hgetall(k)
            # convert JSON-like fields if present
            parsed = {kk: (json.loads(v) if (v and (v.strip().startswith("{") or v.strip().startswith("["))) else v) for kk,v in h.items()}
            parsed['key'] = k
            rows.append(parsed)
        except Exception:
            rows.append({"key": k, "raw": h})
    return rows

def fetch_recent_events(colname, limit=50):
    col = db[colname]
    docs = list(col.find().sort("received_at", -1).limit(limit))
    return docs

# main loop (streamlit auto-runs; use st.experimental_rerun via a timer)
while True:
    # positions
    pos_rows = fetch_positions()
    if pos_rows:
        pos_placeholder.write(pos_rows)
    else:
        pos_placeholder.write("No active positions")

    # events
    docs = fetch_recent_events(sel, limit=30)
    if docs:
        # show simplified table
        simplified = []
        for d in docs:
            payload = d.get("payload")
            symbol = None
            if isinstance(payload, dict):
                symbol = payload.get("symbol") or payload.get("order_id")
            simplified.append({
                "ts": d.get("received_at"),
                "channel": d.get("channel"),
                "symbol": symbol,
                "payload": json.dumps(payload) if isinstance(payload, dict) else str(payload)
            })
        events_placeholder.table(simplified)
    else:
        events_placeholder.write("No events yet")

    time.sleep(REFRESH)
    # Streamlit will re-run the script, but we keep loop for simplicity (this is acceptable for local dev)
