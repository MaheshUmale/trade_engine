# mse.py
# Market-Structure Engine (MSE) with breakout filters:
# - ATH / 52-week detection
# - Breakout candle strength
# - Volume confirmation
# - Optional retest confirmation
# - Imbalance detection
#
# Usage: run the file or import MSEAnalyst and compute_trade_plan_from_ctx as before.

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# plotting libs used in demo
import matplotlib.pyplot as plt

# ----------------- Configuration (tunable) -----------------
CONFIG = {
    # ATR multipliers & thresholds
    "MIN_HTF_DISTANCE_ATR_MULT": 1.5,
    "MIN_LTF_DISTANCE_ATR_MULT": 1.0,
    "MIN_TP_DISTANCE_ATR_MULT": 1.5,
    "MIN_RR": 2.0,
    # Liquidity pool thresholds
    "LIQUIDITY_MIN_WICK": 2,
    "LIQUIDITY_MIN_VOL_RATIO": 0.5,
    # Breakout filters
    "BREAKOUT_ATR_MULT": 2.0,          # TP = price + BREAKOUT_ATR_MULT * ATR for ATH fallback
    "BREAKOUT_BODY_ATR_MULT": 1.0,     # breakout candle body must be >= X * ATR
    "BREAKOUT_BODY_RATIO": 0.6,        # body / range >= this ratio to be strong body
    "VOLUME_CONFIRM_MULT": 1.5,        # breakout volume >= this * avg_vol
    "RETEST_ENABLED": True,
    "RETEST_LOOKBACK_BARS": 12,        # look-back window to consider retest
    "RETEST_MAX_PULLBACK_PCT": 0.5,    # max allowed pullback as % of ATR (in %)
    "IMBALANCE_THRESHOLD": 0.6,       # proportion of range as imbalance indicator
    "ATH_PROX_PCT": 0.002,            # within 0.2% of ATH considered ATH proximity
    "WEEK52_BARS": 252*24*60//5,      # approx bars - (unused if not available)
}

# ----------------- Data retrieval utility (tvDatafeed fallback) -----------------
def get_ohlcv(symbol='BANKNIFTY', exchange='NSE', n_bars=400, interval_minutes=5):
    """
    Try tvDatafeed first; otherwise synthetic fallback.
    interval_minutes only affects fallback synthetic generator.
    """
    try:
        from tvDatafeed import TvDatafeed, Interval
        tv = TvDatafeed()
        if interval_minutes == 1:
            tv_interval = Interval.in_1_minute
        elif interval_minutes == 5:
            tv_interval = Interval.in_5_minute
        elif interval_minutes == 15:
            tv_interval = Interval.in_15_minute
        elif interval_minutes == 60:
            tv_interval = Interval.in_1_hour
        else:
            tv_interval = Interval.in_1_minute
        df = tv.get_hist(symbol, exchange, interval=tv_interval, n_bars=n_bars)
        df = df.reset_index().rename(columns={'index':'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[['datetime','open','high','low','close','volume']].copy()
        return df
    except Exception as e:
        # fallback synthetic
        rng = pd.date_range(end=pd.Timestamp.now(), periods=n_bars, freq=f'{interval_minutes}T')
        np.random.seed(42)
        price = 200 + np.cumsum(np.random.randn(n_bars) * 0.3 * (interval_minutes/1.0))
        high = price + np.random.rand(n_bars) * 0.5
        low = price - np.random.rand(n_bars) * 0.5
        openp = price + (np.random.rand(n_bars)-0.5)*0.2
        close = price + (np.random.rand(n_bars)-0.5)*0.2
        volume = (np.abs(np.random.randn(n_bars))*1000*(interval_minutes/1.0)).astype(int)
        df = pd.DataFrame({'datetime': rng, 'open': openp, 'high': high, 'low': low, 'close': close, 'volume': volume})
        return df

# ----------------- Basic indicators -----------------
def compute_atr(df, n=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
    atr = tr.rolling(n, min_periods=1).mean()
    return atr

def avg_volume(df, window=20):
    return df['volume'].rolling(window, min_periods=1).mean().iloc[-1]

# ----------------- Swing detection -----------------
def detect_swings(df, lookback=5):
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    pivots = []
    for i in range(lookback, n - lookback):
        hi_segment = highs[i-lookback:i+lookback+1]
        lo_segment = lows[i-lookback:i+lookback+1]
        if highs[i] == max(hi_segment) and list(hi_segment).count(highs[i]) == 1:
            pivots.append({'idx': i, 'type': 'H', 'price': float(highs[i]), 'ts': df['datetime'].iloc[i]})
        if lows[i] == min(lo_segment) and list(lo_segment).count(lows[i]) == 1:
            pivots.append({'idx': i, 'type': 'L', 'price': float(lows[i]), 'ts': df['datetime'].iloc[i]})
    return pivots

# ----------------- Order-block detection -----------------
def detect_order_blocks(df, atr, vol_window=20, impulse_mult=1.0, vol_mult=1.5):
    n = len(df)
    avg_vol = df['volume'].rolling(vol_window, min_periods=1).mean().fillna(0)
    zones = []
    for i in range(1, n):
        if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
            continue
        rng = df['high'].iloc[i] - df['low'].iloc[i]
        if rng > impulse_mult * atr.iloc[i] and df['volume'].iloc[i] > vol_mult * max(1.0, avg_vol.iloc[i]):
            prev = df.iloc[i-1]
            if df['close'].iloc[i] > df['open'].iloc[i]:
                zb_high = max(prev['open'], prev['close'], prev['high'])
                zb_low = min(prev['open'], prev['close'], prev['low'])
                strength = rng * df['volume'].iloc[i]
                zones.append({'type':'demand', 'high': float(zb_high), 'low': float(zb_low), 'created_idx': i-1, 'strength': float(strength)})
            else:
                prev = df.iloc[i-1]
                zb_high = max(prev['open'], prev['close'], prev['high'])
                zb_low = min(prev['open'], prev['close'], prev['low'])
                strength = rng * df['volume'].iloc[i]
                zones.append({'type':'supply', 'high': float(zb_high), 'low': float(zb_low), 'created_idx': i-1, 'strength': float(strength)})
    return zones

# ----------------- Liquidity mapping -----------------
def detect_liquidity_pools(df, bins=60, min_wick=2, min_vol_ratio=0.5):
    prices = np.concatenate([df['low'].values, df['high'].values])
    pmin, pmax = prices.min(), prices.max()
    if pmax == pmin:
        return {'above': [], 'below': [], 'all_pools': []}
    bin_size = (pmax - pmin) / bins
    centers = [pmin + (i+0.5)*bin_size for i in range(bins)]
    wick_count = np.zeros(bins, dtype=int)
    vol_sum = np.zeros(bins, dtype=float)
    for _, row in df.iterrows():
        hi = row['high']
        lo = row['low']
        vol = row['volume']
        lo_bin = int((lo - pmin) / bin_size)
        hi_bin = int((hi - pmin) / bin_size)
        lo_bin = max(0, min(bins-1, lo_bin))
        hi_bin = max(0, min(bins-1, hi_bin))
        wick_count[hi_bin] += 1
        wick_count[lo_bin] += 1
        span = max(1, hi_bin - lo_bin + 1)
        vol_per_bin = vol / span
        vol_sum[lo_bin:hi_bin+1] += vol_per_bin
    mean_vol = vol_sum.mean() if vol_sum.mean() > 0 else 1.0
    strong_bins = [i for i in range(bins) if (wick_count[i] >= min_wick and (vol_sum[i] / (mean_vol + 1e-9)) >= min_vol_ratio)]
    if not strong_bins:
        score = wick_count + (vol_sum / (mean_vol + 1e-9))
        top_idx = np.argsort(score)[-6:][::-1]
        pools = [centers[i] for i in top_idx]
    else:
        sorted_bins = sorted(strong_bins, key=lambda i: (wick_count[i], vol_sum[i]), reverse=True)
        top_idx = sorted_bins[:8]
        pools = [centers[i] for i in top_idx]
    last_close = df['close'].iloc[-1]
    above = [p for p in pools if p > last_close]
    below = [p for p in pools if p < last_close]
    return {'above': sorted(above), 'below': sorted(below), 'all_pools': sorted(pools)}

# ----------------- Compression and impulse -----------------
def detect_compression(df, atr):
    recent_atr = atr.iloc[-10:].mean()
    price = df['close'].iloc[-1]
    if price == 0 or np.isnan(recent_atr):
        return False
    return (recent_atr / price) < 0.0015

def detect_impulse(df, atr):
    n = len(df)
    if n < 4:
        return False
    last_ranges = (df['high'] - df['low']).iloc[-3:]
    atr_recent = atr.iloc[-3:]
    if (last_ranges.values > atr_recent.values).sum() >= 2:
        closes = df['close'].iloc[-4:]
        return closes.iloc[-1] > closes.iloc[-3]
    return False

# ----------------- HTF integration -----------------
def build_htf_zones(symbol, exchange='NSE', tv=None):
    zones = {'htf_15': [], 'htf_60': [], 'htf_d1': []}
    try:
        if tv is None:
            from tvDatafeed import TvDatafeed, Interval
            tv = TvDatafeed()
        # fetch HTF frames
        df15 = tv.get_hist(symbol, exchange, interval=tv.Interval.in_15_minute, n_bars=400).reset_index().rename(columns={'index':'datetime'})
        df60 = tv.get_hist(symbol, exchange, interval=tv.Interval.in_1_hour, n_bars=400).reset_index().rename(columns={'index':'datetime'})
        dfd = tv.get_hist(symbol, exchange, interval=tv.Interval.in_daily, n_bars=300).reset_index().rename(columns={'index':'datetime'})
        for df, key, impulse_mult in [(df15, 'htf_15', 1.0), (df60, 'htf_60', 1.0), (dfd, 'htf_d1', 1.2)]:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df[['datetime','open','high','low','close','volume']].copy()
            atr = compute_atr(df)
            z = detect_order_blocks(df, atr, vol_window=20, impulse_mult=impulse_mult, vol_mult=1.3)
            filtered = []
            for zz in z:
                width_pct = (zz['high'] - zz['low']) / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 0
                if zz.get('strength', 0) > 0 and width_pct > 0.00005:
                    filtered.append(zz)
            zones[key] = filtered
    except Exception:
        pass
    return zones

# ----------------- structural level selection -----------------
def nearest_structural_levels(live_df, ltf_zones, ltf_pivots, htf_zones_dict, atr_series):
    last_price = live_df['close'].iloc[-1]
    atr_now = atr_series.iloc[-1] if not atr_series.isna().iloc[-1] else (last_price * 0.001)
    candidates_res = []
    candidates_sup = []
    # HTF priority
    for tf in ['htf_d1', 'htf_60', 'htf_15']:
        zones = htf_zones_dict.get(tf, [])
        for z in zones:
            if z['type'] == 'supply':
                dist_pts = z['low'] - last_price
                if dist_pts > (CONFIG['MIN_HTF_DISTANCE_ATR_MULT'] * atr_now):
                    candidates_res.append({'price': z['low'], 'source': tf, 'dist': dist_pts, 'strength': z.get('strength',0)})
            else:
                dist_pts = last_price - z['high']
                if dist_pts > (CONFIG['MIN_HTF_DISTANCE_ATR_MULT'] * atr_now):
                    candidates_sup.append({'price': z['high'], 'source': tf, 'dist': dist_pts, 'strength': z.get('strength',0)})
    # fallback to LTF zones if none
    if not candidates_res or not candidates_sup:
        for z in ltf_zones:
            if z['type'] == 'supply':
                dist_pts = z['low'] - last_price
                if dist_pts > (CONFIG['MIN_LTF_DISTANCE_ATR_MULT'] * atr_now):
                    candidates_res.append({'price': z['low'], 'source': 'ltf_zone', 'dist': dist_pts, 'strength': z.get('strength', 0)})
            else:
                dist_pts = last_price - z['high']
                if dist_pts > (CONFIG['MIN_LTF_DISTANCE_ATR_MULT'] * atr_now):
                    candidates_sup.append({'price': z['high'], 'source': 'ltf_zone', 'dist': dist_pts, 'strength': z.get('strength', 0)})
    # pivots last resort
    if not candidates_res:
        for p in ltf_pivots:
            if p['type'] == 'H':
                dist_pts = p['price'] - last_price
                if dist_pts > (CONFIG['MIN_LTF_DISTANCE_ATR_MULT'] * atr_now):
                    candidates_res.append({'price': p['price'], 'source': 'ltf_pivot', 'dist': dist_pts, 'strength': 1})
    if not candidates_sup:
        for p in ltf_pivots:
            if p['type'] == 'L':
                dist_pts = last_price - p['price']
                if dist_pts > (CONFIG['MIN_LTF_DISTANCE_ATR_MULT'] * atr_now):
                    candidates_sup.append({'price': p['price'], 'source': 'ltf_pivot', 'dist': dist_pts, 'strength': 1})
    nearest_res = None
    nearest_sup = None
    used_source = None
    if candidates_res:
        sel = min(candidates_res, key=lambda x: x['dist'])
        nearest_res = sel['price']; used_source = sel['source']
    if candidates_sup:
        sel = min(candidates_sup, key=lambda x: x['dist'])
        nearest_sup = sel['price']; used_source = used_source or sel['source']
    nearest_res_pct = None
    if nearest_res is not None and last_price>0:
        nearest_res_pct = 100.0 * (nearest_res - last_price) / last_price
    return nearest_res, nearest_sup, nearest_res_pct, used_source

# ----------------- New breakout helper functions -----------------
def detect_ath(df, lookback_days=365):
    """
    Detect price is near ATH or 52-week high.
    lookback_days may be approximated by rows in df if DF is daily.
    Here we use entire df: if current close >= historical high * (1 - ATH_PROX_PCT) => ATH proximity
    """
    hist_high = df['high'].max()
    price = df['close'].iloc[-1]
    return price >= hist_high * (1.0 - CONFIG['ATH_PROX_PCT'])

def detect_breakout_candle(df, atr, body_atr_mult=None, body_ratio=None):
    """
    Check last candle qualifies as breakout candle:
    - body >= body_atr_mult * ATR
    - body/range >= body_ratio
    - close > prior high (for long breakout) or close < prior low (for short)
    """
    if body_atr_mult is None: body_atr_mult = CONFIG['BREAKOUT_BODY_ATR_MULT']
    if body_ratio is None: body_ratio = CONFIG['BREAKOUT_BODY_RATIO']
    if len(df) < 2: return False, "Insufficient bars"
    last = df.iloc[-1]
    prev = df.iloc[-2]
    atr_val = atr if not np.isnan(atr ) else max(1.0, last['close']*0.001)
    body = abs(last['close'] - last['open'])
    rng = last['high'] - last['low'] if (last['high'] - last['low']) > 0 else 1e-9
    # directional check for long
    if last['close'] <= prev['high']:
        return False, "Close not above prior high"
    if body < body_atr_mult * atr_val:
        return False, f"Body too small ({body:.2f} < {body_atr_mult}*ATR={body_atr_mult*atr_val:.2f})"
    if (body / rng) < body_ratio:
        return False, f"Body/Range ratio too low ({body/rng:.2f} < {body_ratio})"
    return True, "Breakout candle OK"

def volume_confirmation(df, vol_mult=None, window=20):
    if vol_mult is None: vol_mult = CONFIG['VOLUME_CONFIRM_MULT']
    if len(df) < window + 1:
        avg = avg_volume(df, window=window)
    else:
        avg = df['volume'].rolling(window, min_periods=1).mean().iloc[-2]
    last_vol = df['volume'].iloc[-1]
    return last_vol >= vol_mult * max(1.0, avg), f"vol {last_vol} vs avg {avg:.1f}"

def detect_retest(df, breakout_level, lookback=12, max_pullback_atr_mult=0.5):
    """
    Check if price pulled back to the breakout_level within recent lookback bars and then bounced above it.
    Returns (True/False, reason)
    """
    if lookback <= 0 or len(df) < 3:
        return False, "Retest disabled or insufficient bars"
    recent = df.iloc[-lookback-1:].reset_index(drop=True)
    atr = compute_atr(recent).iloc[-1] if len(recent)>1 else (recent['close'].iloc[-1]*0.001)
    # find if price touched or dipped slightly below breakout_level and then closed above it later
    touched_idx = None
    for i in range(len(recent)-1):
        if recent['low'].iloc[i] <= breakout_level <= recent['high'].iloc[i]:
            touched_idx = i
    if touched_idx is None:
        return False, "No retest touch"
    # ensure bounce after touch
    after = recent.iloc[touched_idx+1:]
    if after.empty:
        return False, "No bars after touch"
    # check if close after touch is above breakout_level and pullback size within limit
    pullback = breakout_level - recent['low'].iloc[touched_idx]
    if pullback > max_pullback_atr_mult * atr:
        return False, f"Pullback too deep ({pullback:.2f} > {max_pullback_atr_mult}*ATR)"
    # bounce check
    bounced = any(after['close'] > breakout_level)
    return bounced, ("bounced" if bounced else "no bounce")

def detect_imbalance(df, lookback=10, imbalance_threshold=CONFIG['IMBALANCE_THRESHOLD']):
    """
    Detect imbalance: a candle with large range and small wick on one side indicating sweep/inefficiency.
    Simple heuristic: recent impulse candle where body / range > threshold and the opposite wick small.
    """
    if len(df) < lookback+1:
        return False, "insufficient data"
    recent = df.iloc[-lookback:]
    for i in range(len(recent)):
        row = recent.iloc[i]
        body = abs(row['close'] - row['open'])
        rng = row['high'] - row['low'] if (row['high'] - row['low']) > 0 else 1e-9
        upper_wick = row['high'] - max(row['close'], row['open'])
        lower_wick = min(row['close'], row['open']) - row['low']
        # bullish imbalance (small lower wick, big body)
        if body / rng >= imbalance_threshold and lower_wick <= 0.1*body:
            return True, "bullish imbalance"
        if body / rng >= imbalance_threshold and upper_wick <= 0.1*body:
            return True, "bearish imbalance"
    return False, "no imbalance"

# ----------------- Scoring reused -----------------
def compute_structure_score(features, weights=None):
    if weights is None:
        weights = {
            'htf_alignment': 20,
            'clean_air': 15,
            'impulse': 10,
            'liquidity_penalty': -20,
            'resistance_penalty': -25,
            'trend_conflict_penalty': -20,
            'compression_penalty': -30,
            'volume_confirmation': 10
        }
    score = 0
    blockers = []
    if features.get('htf_trend') == features.get('signal_dir'):
        score += weights['htf_alignment']
    elif features.get('htf_trend') is None:
        pass
    else:
        score -= weights['htf_alignment']
        blockers.append({'type':'trend_conflict', 'reason':'HTF opposite'})
    if features.get('nearest_res_pct') is not None and features['nearest_res_pct'] >= features.get('clean_air_pct', 0.5):
        score += weights['clean_air']
    else:
        score += -weights['resistance_penalty'] if features.get('nearest_res_pct') is not None else 0
        if features.get('nearest_res_pct') is not None:
            blockers.append({'type':'resistance', 'reason':f"res within {features['nearest_res_pct']:.2f}%"})
    if features.get('impulse'):
        score += weights['impulse']
    if features.get('liquidity'):
        pools = features['liquidity'].get('above',[])
        entry = features.get('entry', features.get('price'))
        tp = features.get('tp', entry)
        for p in pools:
            if entry < p < tp:
                score += weights['liquidity_penalty']
                blockers.append({'type':'liquidity', 'reason':'liquidity pool between entry and TP'})
                break
    if features.get('compression'):
        score += weights['compression_penalty']
        blockers.append({'type':'compression','reason':'compression detected'})
    if features.get('volume_confirm'):
        score += weights['volume_confirmation']
    return score, blockers

# ----------------- Trade plan generator with breakout filters -----------------
def compute_trade_plan_from_ctx(signal, df, ctx, rr_min=None):
    """
    Returns (trade_plan_dict, None) if valid else (None, reason_str)
    Enhanced: handles breakout fallback with filters.
    """
    if rr_min is None:
        rr_min = CONFIG['MIN_RR']
    price = df['close'].iloc[-1]
    atr_series = compute_atr(df)
    atr = atr_series.iloc[-1] if not np.isnan(atr_series.iloc[-1]) else max(1.0, price*0.001)
    direction = ctx.get('structure_bias', signal.get('signal_dir', 'long'))
    nearest_res = ctx.get('nearest_resistance')
    nearest_sup = ctx.get('nearest_support')

    # define SL candidate (structural)
    sl_candidate = nearest_sup if direction == 'long' else nearest_res
    if sl_candidate is None:
        reason = "No structural SL candidate found"
        # If no SL candidate, we can still consider breakout only if we can compute a conservative SL (e.g., price - 2*ATR)
        # But prefer structural SL; here reject early to be safe.
        return None, reason

    sl_dist = abs(price - sl_candidate)
    if sl_dist < (CONFIG['MIN_LTF_DISTANCE_ATR_MULT'] * atr):
        return None, f"SL too tight ({sl_dist:.2f} < {CONFIG['MIN_LTF_DISTANCE_ATR_MULT']}*ATR)"

    # Normal case: structural TP exists
    if nearest_res is not None and direction == 'long':
        tp_candidate = nearest_res + 0.25 * atr
        tp_dist = tp_candidate - price
        if tp_dist < (CONFIG['MIN_TP_DISTANCE_ATR_MULT'] * atr):
            # TP too close -> consider using breakout-mode if breakout detected and validated
            pass
        else:
            rr = tp_dist / sl_dist
            if rr < rr_min:
                return None, f"R:R {rr:.2f} < {rr_min}"
            return {'entry': float(price), 'sl': float(sl_candidate), 'tp': float(tp_candidate), 'rr': float(rr), 'mode': 'structural'}, None

    # If we are here: either no structural TP or TP too close. Consider breakout handling.
    # Breakout detection & filters:
    filters_failed = []
    # 1) ATH proximity check
    ath_flag = detect_ath(df)
    # 2) breakout candle check
    ok_candle, reason_candle = detect_breakout_candle(df, atr, body_atr_mult=CONFIG['BREAKOUT_BODY_ATR_MULT'], body_ratio=CONFIG['BREAKOUT_BODY_RATIO'])
    if not ok_candle:
        filters_failed.append(f"breakout_candle:{reason_candle}")
    # 3) volume confirmation
    vol_ok, vol_reason = volume_confirmation(df, vol_mult=CONFIG['VOLUME_CONFIRM_MULT'])
    if not vol_ok:
        filters_failed.append(f"volume:{vol_reason}")
    # 4) imbalance detection
    imb_ok, imb_reason = detect_imbalance(df)
    if not imb_ok:
        # imbalance is helpful but not mandatory; mark as weak if not present
        # do not fail on imbalance absence, just note it
        pass
    # 5) optional retest confirmation (if enabled, require retest OR stronger candle/volume)
    retest_ok = True
    if CONFIG['RETEST_ENABLED']:
        # breakout level is prior high (approx prev high)
        breakout_level = df['high'].iloc[-2]  # approximate prior high as breakout level
        retest_ok, retest_reason = detect_retest(df, breakout_level, lookback=CONFIG['RETEST_LOOKBACK_BARS'], max_pullback_atr_mult=CONFIG['RETEST_MAX_PULLBACK_PCT'])
        if not retest_ok:
            # allow fallback if candle+volume+imbalance are strong
            if not (ok_candle and vol_ok and imb_ok):
                filters_failed.append(f"retest:{retest_reason}")

    # Evaluate filters
    if filters_failed:
        return None, f"Breakout filters failed: {filters_failed}"

    # If passed filters, define breakout TP using ATR multiple
    tp_candidate = price + CONFIG['BREAKOUT_ATR_MULT'] * atr if direction == 'long' else price - CONFIG['BREAKOUT_ATR_MULT'] * atr
    tp_dist = abs(tp_candidate - price)
    if tp_dist < (CONFIG['MIN_TP_DISTANCE_ATR_MULT'] * atr):
        return None, f"Breakout TP too close ({tp_dist:.2f} < {CONFIG['MIN_TP_DISTANCE_ATR_MULT']}*ATR)"
    rr = tp_dist / sl_dist
    if rr < rr_min:
        return None, f"R:R {rr:.2f} < {rr_min}"
    return {'entry': float(price), 'sl': float(sl_candidate), 'tp': float(tp_candidate), 'rr': float(rr), 'mode': 'breakout_atr', 'ath_proximity': ath_flag}, None

# ----------------- MSE Analyst (orchestrator) -----------------
class MSEAnalyst:
    def __init__(self, df_ltf, symbol='DEMO', exchange='NSE', tv=None, signal_dir='long'):
        self.symbol = symbol
        self.exchange = exchange
        self.df = df_ltf.copy().reset_index(drop=True)
        self.signal_dir = signal_dir
        self.atr = compute_atr(self.df)
        self.pivots = detect_swings(self.df, lookback=5)
        self.zones = detect_order_blocks(self.df, self.atr, vol_window=20, impulse_mult=0.8, vol_mult=2.0)
        self.liquidity = detect_liquidity_pools(self.df)
        self.compression = detect_compression(self.df, self.atr)
        self.impulse = detect_impulse(self.df, self.atr)
        try:
            self.htf_zones = build_htf_zones(self.symbol, self.exchange, tv=tv)
        except Exception:
            self.htf_zones = {'htf_15': [], 'htf_60': [], 'htf_d1': []}
        self.nearest_resistance, self.nearest_support, self.nearest_res_pct, self.used_source = nearest_structural_levels(self.df, self.zones, self.pivots, self.htf_zones, self.atr)
        self.structure_bias = 'long' if self.impulse else 'neutral'

    def build_context(self):
        features = {
            'htf_trend': None,
            'signal_dir': self.signal_dir,
            'price': self.df['close'].iloc[-1],
            'liquidity': self.liquidity,
            'nearest_res_pct': self.nearest_res_pct,
            'clean_air_pct': 0.5,
            'impulse': self.impulse,
            'compression': self.compression,
            'entry': self.df['close'].iloc[-1],
            'tp': self.nearest_resistance if self.nearest_resistance is not None else self.df['close'].iloc[-1] + 2*(self.df['close'].iloc[-1] - (self.nearest_support or (self.df['close'].iloc[-1]-1))),
            'volume_confirm': False
        }
        score, blockers = compute_structure_score(features)
        ctx = {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'htf_trend': features['htf_trend'],
            'mtf_trend': None,
            'ltf_trend': None,
            'nearest_resistance': self.nearest_resistance,
            'nearest_support': self.nearest_support,
            'nearest_res_pct': self.nearest_res_pct,
            'liquidity': self.liquidity,
            'compression': self.compression,
            'impulse': self.impulse,
            'score': score,
            'blockers': blockers,
            'used_source': getattr(self, 'used_source', None),
            'structure_bias': self.structure_bias,
            'atr': float(self.atr.iloc[-1]) if not self.atr.isna().iloc[-1] else None,
            'current_price': float(self.df['close'].iloc[-1])
        }
        return ctx


# ======================================================
# NEW: MarketStructureEngine Wrapper for External Modules
# ======================================================

class MarketStructureEngine:
    """
    Wrapper used by the downstream event pipeline.
    Provides:
        await engine.process_symbol(symbol, ltp)
    """

    def __init__(self, interval_minutes=5, n_bars=400):
        self.interval = interval_minutes
        self.n_bars = n_bars

    async def process_symbol(self, symbol: str, ltp: float, direction="long"):
        """
        Executes:
            - Fetch OHLCV (via tvDatafeed or fallback)
            - Run MSEAnalyst
            - Compute trade plan
        Returns:
            dict (trade plan + context)
            or None if invalid
        """
        try:
            # 1. Fetch latest LTF candles
            df = get_ohlcv(symbol, n_bars=self.n_bars, interval_minutes=self.interval)

            # 2. Build analyst
            analyst = MSEAnalyst(df, symbol=symbol, signal_dir=direction)

            # 3. Context
            ctx = analyst.build_context()

            # 4. Trade plan
            trade_plan, reason = compute_trade_plan_from_ctx(
                {"signal_dir": direction},
                df,
                ctx,
                rr_min=CONFIG["MIN_RR"]
            )

            if trade_plan is None:
                return {
                    "symbol": symbol,
                    "valid": False,
                    "reason": reason,
                    "mse_context": ctx,
                    "blockers": ctx.get("blockers", [])
                }

            # Successful
            return {
                "symbol": symbol,
                "valid": True,
                "entry": trade_plan["entry"],
                "sl": trade_plan["sl"],
                "tp": trade_plan["tp"],
                "rr": trade_plan["rr"],
                "mode": trade_plan.get("mode"),
                "ath": trade_plan.get("ath_proximity"),
                "mse_context": ctx,
                "blockers": ctx.get("blockers", [])
            }

        except Exception as e:
            return {
                "symbol": symbol,
                "valid": False,
                "reason": f"Exception: {e}"
            }

















# ----------------- Demo main -----------------
if __name__ == "__main__":
    symbol = 'M_M'
    df = get_ohlcv(symbol, n_bars=400, interval_minutes=5)
    # df=df.head(350)
    analyst = MSEAnalyst(df, symbol=symbol, exchange='NSE', tv=None, signal_dir='long')
    ctx = analyst.build_context()
    trade_plan, reason = compute_trade_plan_from_ctx({'signal_id':'demo', 'signal_dir':'long'}, df, ctx, rr_min=CONFIG['MIN_RR'])
    print("MSE Context Summary")
    print(pd.DataFrame([{
        'nearest_resistance': ctx['nearest_resistance'],
        'nearest_support': ctx['nearest_support'],
        'nearest_res_pct': ctx['nearest_res_pct'],
        'score': ctx['score'],
        'blockers': ctx['blockers'],
        'used_source': ctx.get('used_source'),
        'atr': ctx.get('atr'),
        'current_price': ctx.get('current_price')
    }]))
    if trade_plan is None:
        print("Trade plan invalid:", reason)
    else:
        print("Trade plan:", trade_plan)



    # Simple plotting (mplfinance) - keep as before but with improved levels
    try:
        import mplfinance as mpf
        df_plot = df.set_index('datetime').copy()
        df_plot.columns = map(str.capitalize, df_plot.columns)
        add_plots = []
        # pivots
        h_pivots = pd.Series(np.nan, index=df_plot.index)
        l_pivots = pd.Series(np.nan, index=df_plot.index)
        for p in analyst.pivots:
            if p['type'] == 'H':
                h_pivots.iloc[p['idx']] = p['price']
            else:
                l_pivots.iloc[p['idx']] = p['price']
        add_plots.append(mpf.make_addplot(h_pivots, type='scatter', marker='^', color='red', markersize=50, label='High Pivot'))
        add_plots.append(mpf.make_addplot(l_pivots, type='scatter', marker='v', color='green', markersize=50, label='Low Pivot'))
        # zones
        for i, z in enumerate(analyst.zones):
            label_text = 'Zone' if i == 0 else '_nolegend_'
            # draw zone as filled series
            add_plots.append(mpf.make_addplot(pd.Series(z['high'], index=df_plot.index), color='blue', alpha=0.08,
                                              fill_between=dict(y1=z['low'], y2=z['high'], color='blue', alpha=0.08),
                                              secondary_y=False, label=label_text))
        # liquidity pools
        for i, p in enumerate(analyst.liquidity.get('all_pools', [])[:8]):
            label = 'Liquidity Pool' if i == 0 else '_nolegend_'
            hline_series = pd.Series(p, index=df_plot.index)
            add_plots.append(mpf.make_addplot(hline_series, color='purple', linestyle='-.', width=0.6, label=label, secondary_y=False))
        # structural lines
        if ctx['nearest_resistance'] is not None:
            add_plots.append(mpf.make_addplot(pd.Series(ctx['nearest_resistance'], index=df_plot.index), color='orange', linestyle='--', label='Nearest Resistance'))
        if ctx['nearest_support'] is not None:
            add_plots.append(mpf.make_addplot(pd.Series(ctx['nearest_support'], index=df_plot.index), color='brown', linestyle=':', label='Nearest Support'))
        if trade_plan is not None:
            add_plots.append(mpf.make_addplot(pd.Series(trade_plan['entry'], index=df_plot.index), color='blue', linestyle='-', label='Entry'))
            add_plots.append(mpf.make_addplot(pd.Series(trade_plan['sl'], index=df_plot.index), color='red', linestyle='--', label='Stop Loss'))
            add_plots.append(mpf.make_addplot(pd.Series(trade_plan['tp'], index=df_plot.index), color='green', linestyle='-.', label='Take Profit'))
        fig, axes = mpf.plot(df_plot, type='candle',
                 title=f"MSE : {symbol} - Price with Structural Levels (source={ctx.get('used_source')})",
                 addplot=add_plots,
                 figratio=(14,6),
                 tight_layout=True,
                 volume=True,
                 style='yahoo',
                 returnfig=True)
        ax = axes[0]
        ax.text(df_plot.index[int(len(df_plot)*0.05)], ax.get_ylim()[1] * 0.95,
                f"Score: {ctx['score']}  Blockers: {len(ctx['blockers'])}  ATR: {ctx.get('atr')}",
                bbox=dict(facecolor='white', alpha=0.7), verticalalignment='top')
        mpf.show()
    except Exception as e:
        # fallback to simple matplotlib if mplfinance missing
        print("mplfinance plotting failed:", e)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.plot(df['datetime'], df['close'], label='close')
        if ctx['nearest_resistance'] is not None:
            plt.axhline(ctx['nearest_resistance'], linestyle='--', color='orange', label='Nearest Resistance')
        if ctx['nearest_support'] is not None:
            plt.axhline(ctx['nearest_support'], linestyle=':', color='brown', label='Nearest Support')
        if trade_plan is not None:
            plt.axhline(trade_plan['entry'], linestyle='-', color='blue', label='Entry')
            plt.axhline(trade_plan['sl'], linestyle='--', color='red', label='SL')
            plt.axhline(trade_plan['tp'], linestyle='-.', color='green', label='TP')
        plt.legend()
        plt.show()