# backtester_event.py (fixed)
"""
Patched event-driven backtester that uses mse_enhanced.MarketStructureEngine when available,
and writes plot payload containing order_blocks, fvgs, hvns, trades, and candles.

Key fixes:
 - Map MSE 'created_time' to a valid created_index inside the candles array
 - Clamp indices to visible candle range
 - Filter zones that are far outside the visible price range
 - Deduplicate zones and limit number of zones plotted
"""

import argparse, json, asyncio
import pandas as pd
import numpy as np
from math import floor
from datetime import datetime, timedelta
from dateutil import parser as dateparser

# import rule engine
try:
    from rule_engine import RuleEngine, IndicatorEngine, PatternEngine, EXAMPLE_STRATEGIES
except Exception as e:
    raise RuntimeError("rule_engine.py required") from e

# try import MSE enhanced
MSE_AVAILABLE = False
try:
    from mse_enhanced import MarketStructureEngine
    MSE_AVAILABLE = True
except Exception:
    MarketStructureEngine = None
    MSE_AVAILABLE = False

def fallback_plan(df_recent):
    high = df_recent['high']; low = df_recent['low']; close = df_recent['close']
    tr = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
    atr = tr.tail(14).mean()
    price = float(df_recent['close'].iloc[-1])
    sl = price - 1.5 * atr
    tp = price + 2.5 * atr
    rr = (tp - price) / (price - sl) if (price - sl) > 0 else 0
    return {'entry': price, 'sl': sl, 'tp': tp, 'rr': rr}

def compute_qty(entry, sl, account=200000.0, risk_pct=0.01, min_qty=1):
    tick_risk = abs(entry - sl)
    if tick_risk <= 0: return 0
    max_risk = account * risk_pct
    q = floor(max_risk / tick_risk)
    if q < min_qty: return 0
    return int(q)

# tvDatafeed import is optional in many environments; keep same pattern
try:
    from tvDatafeed import TvDatafeed, Interval
except Exception:
    TvDatafeed = None
    Interval = None

import time, bisect

# -------------------- helper utilities for mapping --------------------
def iso_to_datetime(s):
    if s is None:
        return None
    try:
        return dateparser.parse(s)
    except Exception:
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

def build_candle_datetime_list(candles_plot):
    # candles_plot has {'time': unix_seconds, ...}
    return [datetime.utcfromtimestamp(c['time']) for c in candles_plot]

def find_nearest_index_by_time(candle_dt_list, created_dt):
    """
    candle_dt_list: list of datetime objects sorted ascending
    created_dt: datetime
    returns nearest integer index (clamped)
    """
    if created_dt is None or len(candle_dt_list) == 0:
        return None
    # convert to naive UTC if tz-aware
    if created_dt.tzinfo is not None:
        created_dt = created_dt.astimezone(tz=None).replace(tzinfo=None)
    times = candle_dt_list
    pos = bisect.bisect_left(times, created_dt)
    if pos == 0:
        return 0
    if pos >= len(times):
        return len(times) - 1
    before = times[pos-1]; after = times[pos]
    # choose nearest by absolute delta
    if abs((created_dt - before)) <= abs((after - created_dt)):
        return pos-1
    else:
        return pos

def clamp_index(idx, n):
    if idx is None:
        return None
    if idx < 0:
        return 0
    if idx >= n:
        return n-1
    return int(idx)

def zone_in_price_range(zone, visible_low, visible_high, pad_pct=0.4):
    """
    Return True if zone price range intersects visible range widened by pad_pct of the visible span.
    This avoids plotting zones that are far off-screen.
    """
    span = max(visible_high - visible_low, 1e-6)
    pad = span * pad_pct
    low = float(zone.get('low', zone.get('price', visible_low)))
    high = float(zone.get('high', zone.get('price', visible_high)))
    return not (high < (visible_low - pad) or low > (visible_high + pad))

def normalize_zone_obj(z):
    """Return a sanitized zone dict with numeric high/low and a string type."""
    try:
        return {
            'type': str(z.get('type', '')).strip(),
            'high': float(z.get('high')),
            'low': float(z.get('low')),
            'created_index': z.get('created_index', None),
            'created_time': z.get('created_time', None),
            'strength': float(z.get('strength', 0)) if z.get('strength') is not None else 0.0
        }
    except Exception:
        return None

# -------------------- backtest runner --------------------
async def run_backtest(symbol="NIFTY", strategy_name=None, out_prefix="backtest", timeframes=None, df=None):
    if df is None:
        if TvDatafeed is None:
            raise RuntimeError("TvDatafeed not available and no df passed in.")
        tv = TvDatafeed()
        n_bars_lookback = 500
        interval = "1"
        if timeframes is not None and timeframes[0] is not None:
            interval = str(timeframes[0])
            df = tv.get_hist(symbol, exchange='NSE', interval=Interval(interval), n_bars=n_bars_lookback)
        else:
            df = tv.get_hist(symbol, exchange='NSE', interval=Interval.in_1_minute, n_bars=n_bars_lookback)
        if df is None:
            time.sleep(3)
            df = tv.get_hist(symbol, exchange='NSE', interval=Interval.in_1_minute, n_bars=n_bars_lookback)
        time.sleep(1)

    df = df.sort_values('datetime').reset_index(drop=False)

    strategies = EXAMPLE_STRATEGIES
    strategy = None
    if strategy_name:
        for s in strategies:
            if s.get('strategy_name') == strategy_name:
                strategy = s; break
    if strategy is None:
        strategy = strategies[0]

    rule_engine = RuleEngine()
    ind = IndicatorEngine()
    pat = PatternEngine()

    mse_engine = None
    if MSE_AVAILABLE:
        tf = timeframes or ['1','5','15']
        bars_per = {t: 500 for t in tf}
        mse_engine = MarketStructureEngine(timeframes=tf, bars_per_tf=bars_per)

    trades = []
    candles_plot = []
    order_blocks_plot = []
    fvgs_plot = []
    hvns_plot = set()
    events = []

    recent = []
    n_bars = 800

    # iterate and build candles_plot list (this is the array used for the chart)
    for idx, row in df.iterrows():
        candle = {
            'time': int(row['datetime'].timestamp()),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': int(row['volume'])
        }
        recent.append({
            'datetime': row['datetime'],
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume']
        })
        if len(recent) > n_bars:
            recent = recent[-n_bars:]
        candles_plot.append(candle)

        if len(recent) < 30:
            continue

        # indicators
        closes = [c['close'] for c in recent]
        indicators = {}
        indicators['ema9'] = ind.ema(closes[-30:], 9) if len(closes) >= 9 else None
        indicators['ema20'] = ind.ema(closes[-60:], 20) if len(closes) >= 20 else None
        indicators['vwap'] = ind.vwap(recent[-120:]) if len(recent) >= 1 else recent[-1]['close']
        indicators['rvol'] = ind.rvol(recent, lookback=20)
        indicators['atr'] = ind.atr(recent, period=14)
        patterns = {}
        if len(recent) >= 2:
            patterns['bullish_engulfing'] = pat.bullish_engulfing(recent[-2], recent[-1])

        market_context = {'ohlcv_recent': recent, 'ohlcv_for_vwap': recent[-120:], 'indicators': indicators, 'patterns': patterns}

        res = rule_engine.evaluate_strategy(strategy, market_context)
        if not res.get('match'):
            continue

        events.append({'type': 'potential_signal', 'time': row['datetime'].isoformat(), 'ltp': row['close']})

        plan = None; mse_ctx = None
        if mse_engine:
            try:
                mse_res = await mse_engine.process_symbol(symbol=symbol, ltp=row['close'], custom_df=pd.DataFrame(recent))
                if mse_res.get('valid'):
                    plan = {
                        'entry': mse_res.get('entry'),
                        'sl': mse_res.get('sl'),
                        'tp': mse_res.get('tp'),
                        'rr': mse_res.get('rr')
                    }
                    mse_ctx = mse_res.get('mse_context', {})
                    events.append({'type': 'mse_approved', 'time': row['datetime'].isoformat(), 'mode': mse_res.get('mode'), 'rr': plan['rr']})
                else:
                    events.append({'type': 'mse_reject', 'time': row['datetime'].isoformat(), 'reason': mse_res.get('reason')})
                    continue
            except Exception as e:
                events.append({'type':'mse_error','time':row['datetime'].isoformat(),'err': str(e)})
                plan = fallback_plan(pd.DataFrame(recent))
        else:
            plan = fallback_plan(pd.DataFrame(recent))

        if plan is None:
            continue

        entry = plan['entry']; sl = plan['sl']; tp = plan['tp']
        qty = compute_qty(entry, sl)
        if qty <= 0:
            events.append({'type': 'risk_reject', 'time': row['datetime'].isoformat(), 'reason':'qty_zero'}); continue

        # fill at next candle open
        if idx + 1 >= len(df):
            break
        next_open = float(df.loc[idx+1,'open'])
        fill_price = next_open
        open_time = df.loc[idx+1,'datetime']
        trade = {'symbol':'BACKTEST', 'side': 'long', 'strategy': strategy.get('strategy_name'),
                 'open_time': open_time.isoformat(), 'entry_price': fill_price, 'qty': qty,
                 'sl': sl, 'tp': tp, 'close_time': None, 'close_price': None, 'pnl': None, 'outcome': None,
                 'mse_context': mse_ctx}
        events.append({'type':'order_filled','time':open_time.isoformat(),'entry':fill_price,'qty':qty})

        # walk forward to find exit
        closed=False
        for j in range(idx+1, len(df)):
            hj = float(df.loc[j,'high']); lj = float(df.loc[j,'low']); closej = float(df.loc[j,'close'])
            if hj >= tp:
                trade.update({'close_time': df.loc[j,'datetime'].isoformat(), 'close_price': tp, 'pnl': (tp - fill_price)*qty, 'outcome':'TP'}); events.append({'type':'trade_closed','time':df.loc[j,'datetime'].isoformat(),'reason':'TP'}); closed=True; break
            if lj <= sl:
                trade.update({'close_time': df.loc[j,'datetime'].isoformat(), 'close_price': sl, 'pnl': (sl - fill_price)*qty, 'outcome':'SL'}); events.append({'type':'trade_closed','time':df.loc[j,'datetime'].isoformat(),'reason':'SL'}); closed=True; break
        if not closed:
            last_close = float(df.loc[len(df)-1,'close'])
            trade.update({'close_time': df.loc[len(df)-1,'datetime'].isoformat(), 'close_price': last_close, 'pnl': (last_close - fill_price)*qty, 'outcome':'EXIT'}); events.append({'type':'trade_closed','time':df.loc[len(df)-1,'datetime'].isoformat(),'reason':'END'} )
        trades.append(trade)

        # --------------- collect plot shapes from mse_ctx (robust mapping) -----------------
        if mse_ctx:
            # precompute candle datetime list for mapping
            candle_dt_list = build_candle_datetime_list(candles_plot)

            # visible price min/max
            visible_low = min(c['low'] for c in candles_plot)
            visible_high = max(c['high'] for c in candles_plot)

            # helper to map/validate zone
            def map_zone(z, max_extend_bars=20):
                nz = normalize_zone_obj(z)
                if nz is None: return None
                # prefer created_time -> created_index mapping
                ctime = nz.get('created_time')
                idx_mapped = None
                if ctime:
                    dt = iso_to_datetime(ctime)
                    idx_mapped = find_nearest_index_by_time(candle_dt_list, dt)
                # fall back to provided created_index if present
                if idx_mapped is None and nz.get('created_index') is not None:
                    try:
                        idx_mapped = int(nz.get('created_index'))
                    except Exception:
                        idx_mapped = None
                if idx_mapped is None:
                    # default: place it near the end (last visible region) but keep it valid
                    idx_mapped = max(0, len(candles_plot)-5)
                # clamp
                idx_mapped = clamp_index(idx_mapped, len(candles_plot))
                nz['created_index'] = int(idx_mapped)
                # filter by price range to avoid off-screen garbage
                if not zone_in_price_range(nz, visible_low, visible_high, pad_pct=0.5):
                    return None
                # limit high/low numeric sanity
                try:
                    nz['high'] = float(nz['high'])
                    nz['low'] = float(nz['low'])
                except Exception:
                    return None
                # ensure high>=low
                if nz['high'] < nz['low']:
                    # swap if inverted
                    tmp = nz['high']; nz['high'] = nz['low']; nz['low'] = tmp
                # add a display end index (not stored in JSON, only used locally)
                nz['_end_index'] = min(len(candles_plot)-1, nz['created_index'] + max_extend_bars)
                return nz

            # LTF zones (direct list)
            ltf_z = mse_ctx.get('ltf_zones') or mse_ctx.get('ltf_zones_list') or mse_ctx.get('zones') or []
            processed = []
            for z in ltf_z:
                pz = map_zone(z, max_extend_bars=20)
                if pz:
                    processed.append(pz)
            # dedupe by (type, high, low) keeping strongest if strength available
            seen = {}
            deduped = []
            for z in processed:
                key = (z['type'], round(z['high'], 6), round(z['low'], 6))
                prev = seen.get(key)
                if prev is None or z.get('strength',0) > prev.get('strength',0):
                    seen[key] = z
            deduped = list(seen.values())
            # limit to most recent N
            deduped = sorted(deduped, key=lambda x: x['created_index'])[-12:]
            for z in deduped:
                order_blocks_plot.append({
                    'type': z['type'],
                    'high': z['high'],
                    'low': z['low'],
                    'created_index': int(z['created_index']),
                    'created_time': z.get('created_time')
                })

            # FVGs
            fvg_list = mse_ctx.get('fvg') or []
            processed_fvgs = []
            for f in fvg_list:
                pf = map_zone(f, max_extend_bars=12)
                if pf:
                    processed_fvgs.append(pf)
            # dedupe & limit
            seenf = {}
            for f in processed_fvgs:
                key = (round(f['high'],6), round(f['low'],6))
                if key not in seenf or f.get('strength',0) > seenf[key].get('strength',0):
                    seenf[key] = f
            final_fvgs = list(seenf.values())[-10:]
            for f in final_fvgs:
                fvgs_plot.append({
                    'low': f['low'],
                    'high': f['high'],
                    'created_index': int(f['created_index']),
                    'created_time': f.get('created_time')
                })

            # HTF zones (structured per timeframe)
            htf_zones = mse_ctx.get('htf_zones') or {}
            for tf_key, data in htf_zones.items():
                zones_list = data.get('zones', []) if isinstance(data, dict) else data
                processed_htf = []
                for ob in zones_list:
                    pz = map_zone(ob, max_extend_bars=30)
                    if pz:
                        # attach TF key into type label for clarity
                        pz['type'] = f"{pz['type']} ({tf_key})"
                        processed_htf.append(pz)
                # dedupe per tf and limit
                seen_htf = {}
                for z in processed_htf:
                    key = (z['type'], round(z['high'],6), round(z['low'],6))
                    if key not in seen_htf or z.get('strength',0) > seen_htf[key].get('strength',0):
                        seen_htf[key] = z
                chosen = list(seen_htf.values())[-6:]
                for z in chosen:
                    order_blocks_plot.append({
                        'type': z['type'],
                        'high': z['high'],
                        'low': z['low'],
                        'created_index': int(z['created_index']),
                        'created_time': z.get('created_time')
                    })

            # HVNs (liquidity)
            hvn_list = mse_ctx.get('liquidity', {}).get('hvns', []) if mse_ctx.get('liquidity') else mse_ctx.get('hvns', [])
            for h in (hvn_list or []):
                try:
                    hv = float(h)
                    # only keep if within a reasonable band near visible price
                    if (hv >= visible_low - (visible_high-visible_low)*0.6) and (hv <= visible_high + (visible_high-visible_low)*0.6):
                        hvns_plot.add(round(hv, 9))
                except Exception:
                    continue

    # summarize
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        summary = {'trades':0,'total_pnl':0.0,'win_rate':0.0,'max_drawdown':0.0}
    else:
        total_pnl = trades_df['pnl'].sum(); wins = trades_df[trades_df['pnl']>0]; win_rate = len(wins)/len(trades_df)
        eq = trades_df['pnl'].cumsum(); peak = eq.cummax(); dd = (eq - peak).min() if not eq.empty else 0.0
        summary = {'trades': len(trades_df), 'total_pnl': float(total_pnl), 'win_rate': float(win_rate), 'avg_pnl': float(trades_df['pnl'].mean()), 'max_drawdown': float(dd)}

    # outputs
    trades_df.to_csv(f"{out_prefix}_trades.csv", index=False)
    plot_payload = {
        'candles': candles_plot,
        'order_blocks': order_blocks_plot,
        'fvgs': fvgs_plot,
        'hvns': sorted(list(hvns_plot)),
        'trades': trades,
        'summary': summary,
        'events': events
    }
    with open(f"{out_prefix}_plot_data.json","w") as f:
        json.dump(plot_payload, f, default=str, indent=2)
    print("Backtest done:", summary)
    return trades_df, summary

# simple main to run multiple symbols/strategies (keeps original behavior)
import chart_lightweight as chart
def main():
    with open('strategies.json', 'r') as file:
        strategies = json.load(file)
    symbollist = [ '63MOONS','WELSPUNLIV','GRSE','M_M']
    for symbol in symbollist:
        for strategy in strategies:
            strategyName = strategy["strategy_name"]
            out_prefix = str('./out/'+symbol+'_'+strategyName)
            tfs = [5,15,30]
            trades, summary = asyncio.run(run_backtest(symbol=symbol ,strategy_name=strategyName, out_prefix=out_prefix, timeframes=tfs))
            if trades is not None and not trades.empty:
                print("-------------FOUND TRADE :", f"{out_prefix}_trades.csv")
                chart.main(f"{out_prefix}_plot_data.json", f"{out_prefix}.html")

if __name__ == "__main__":
    main()
