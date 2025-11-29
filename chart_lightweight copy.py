"""
chart_lightweight_upgraded.py

Upgraded chart renderer for backtest_plot_data.json with:
 - TradingView-style translucent rectangles for OB / FVG
 - Floating toolbar with toggles for LTF OB, HTF OB, FVG, HVN, Trades
 - Limits zones drawn (most recent N)
 - Fade-out effect for older zones
 - Clean responsive UI, optimized for clarity

Usage:
  python chart_lightweight_upgraded.py <plot_json> <out_html>
"""

import json
from pathlib import Path
from datetime import datetime
import sys

TEMPLATE = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Backtest Chart — Upgraded UI</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0b0f14; --panel:#0f1720; --muted:#9aa4ad; --accent:#22c55e;
    --ob-long: rgba(34,197,94,0.12); --ob-long-border: rgba(34,197,94,0.28);
    --ob-short: rgba(239,68,68,0.12); --ob-short-border: rgba(239,68,68,0.28);
    --htf-long: rgba(16,185,129,0.10); --htf-short: rgba(249,115,22,0.10);
    --fvg: rgba(255,215,64,0.14); --fvg-border: rgba(255,215,64,0.28);
    --hvn: rgba(96,165,250,0.12); --toolbar: rgba(15,23,42,0.7);
  }
  html,body {height:100%; margin:0; background:var(--bg); color:#e6eef2; font-family:Inter, Arial, sans-serif;}
  #chart { position: absolute; inset:0; }
  .toolbar {
    position: absolute; top:14px; left:14px; z-index: 40;
    background: var(--toolbar); border-radius:10px; padding:8px; display:flex; gap:8px;
    box-shadow: 0 6px 18px rgba(2,6,23,0.6); backdrop-filter: blur(6px);
  }
  .tool-btn { border:0; background:transparent; color:var(--muted); padding:8px 10px; border-radius:8px; cursor:pointer; font-weight:600; font-size:13px; display:flex; gap:8px; align-items:center; }
  .tool-btn.active { color:#fff; background: rgba(255,255,255,0.03); box-shadow: inset 0 -1px 0 rgba(255,255,255,0.02); }
  .legend { position:absolute; right:18px; top:14px; z-index:40; color:var(--muted); background: rgba(11,15,20,0.5); padding:8px 10px; border-radius:8px; font-size:13px; }
  .summaryBox { position:absolute; left:14px; bottom:14px; z-index:40; background: rgba(7,10,13,0.7); padding:10px 12px; border-radius:10px; color:var(--muted); }
  .toggle-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:6px; }
  /* responsive */
  @media (max-width:900px){
    .toolbar { gap:6px; padding:6px; left:8px; top:8px; }
    .tool-btn { padding:6px 8px; font-size:12px; }
    .legend, .summaryBox { font-size:12px; padding:8px; }
  }
</style>
</head>
<body>
<div id="chart"></div>

<div class="toolbar" id="toolbar">
  <button class="tool-btn active" id="btn_ltf_ob">OB (LTF)</button>
  <button class="tool-btn active" id="btn_htf_ob">OB (HTF)</button>
  <button class="tool-btn active" id="btn_fvg">FVG</button>
  <button class="tool-btn" id="btn_hvn">HVN</button>
  <button class="tool-btn active" id="btn_trades">Trades</button>
</div>

<div class="legend" id="legend">Layers: OB LTF / OB HTF / FVG / HVN / Trades</div>
<div class="summaryBox" id="summaryBox">Loading...</div>

<!-- Lightweight Charts -->
<script src="https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js"></script>

<script>
(function(){
  const raw = %%PLOT_JSON%%;
  const data = raw;

  // Config - tune these to change appearance/limits
  const CONFIG = {
    MAX_LTF_ZONES: 8,
    MAX_HTF_ZONES: 6,
    ZONE_EXTEND_BARS: 20,    // how many bars forward the rectangles extend
    ZONE_FADE_AFTER_BARS: 16, // start fading (alpha) after this many bars
    RECT_ROUND: 6,
    OPACITY_BASE: 0.12,
  };

  // Parse candles
  function parseTimeVal(val){
    if (typeof val === 'number') return val;
    const dt = new Date(val);
    if (!isNaN(dt.getTime())) return Math.floor(dt.getTime()/1000);
    return null;
  }
  const candlesMap = new Map();
  (data.candles || []).forEach(c => {
    const t = parseTimeVal(c.time || c.time);
    if (!t) return;
    candlesMap.set(t, {
      time: t, open: +c.open, high:+c.high, low:+c.low, close:+c.close, volume:+(c.volume||0)
    });
  });
  const candles = Array.from(candlesMap.values()).sort((a,b)=>a.time-b.time);
  if (candles.length === 0){
    document.getElementById('summaryBox').innerText = 'No candle data found in JSON.';
    return;
  }

  // Chart
  const chart = LightweightCharts.createChart(document.getElementById('chart'), {
    layout: { backgroundColor: '#0b0f14', textColor: '#cfe3eb' },
    grid: { vertLines: { color: '#0f1720' }, horzLines: { color: '#0f1720' } },
    timeScale: { timeVisible: true, rightOffset: 8 },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal }
  });

  const candleSeries = chart.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350', wickUpColor:'#26a69a', wickDownColor:'#ef5350',
    borderVisible: false
  });
  candleSeries.setData(candles);

  // Overlay canvas for rects (pixel-perfect)
  const overlay = document.createElement('canvas');
  overlay.style.position = 'absolute';
  overlay.style.left = '0';
  overlay.style.top = '0';
  overlay.style.zIndex = 30;
  overlay.style.pointerEvents = 'none';
  document.getElementById('chart').appendChild(overlay);

  function resizeOverlay(){
    const root = document.getElementById('chart');
    overlay.width = root.clientWidth * devicePixelRatio;
    overlay.height = root.clientHeight * devicePixelRatio;
    overlay.style.width = root.clientWidth + 'px';
    overlay.style.height = root.clientHeight + 'px';
    overlay.getContext('2d').scale(devicePixelRatio, devicePixelRatio);
  }
  window.addEventListener('resize', ()=>{ chart.resize(document.getElementById('chart').clientWidth, document.getElementById('chart').clientHeight); resizeOverlay(); drawAll();});
  resizeOverlay();

  // Helpers to convert time/price <-> pixel using LightweightCharts API
  function timeToX(time) {
    // find index of time in candles
    const idx = candles.findIndex(c => c.time === time);
    if (idx === -1) {
      // approximate with nearest index using timeScale
      return chart.timeScale().indexToCoordinate ? chart.timeScale().indexToCoordinate(idx) : null;
    }
    if (!chart.timeScale().indexToCoordinate) return null;
    return chart.timeScale().indexToCoordinate(idx);
  }
  function indexToX(idx){
    if (!chart.timeScale().indexToCoordinate) return null;
    return chart.timeScale().indexToCoordinate(idx);
  }
  function priceToY(price){
    // use series priceToCoordinate
    if (!candleSeries.priceToCoordinate) return null;
    return candleSeries.priceToCoordinate(price);
  }

  // Build zone lists with limits and metadata
  function normalizeZones() {
    const ltf = (data.order_blocks || []).slice(-CONFIG.MAX_LTF_ZONES).map(z => ({...z, source:'ltf'}));
    const fvgs = (data.fvgs || []).slice(-Math.max(CONFIG.MAX_LTF_ZONES, CONFIG.MAX_HTF_ZONES)).map(f => ({...f, source:'fvg'}));
    // HTF: those were added as order_blocks with types like "supply (tf_x)" in backtester; try to extract
    const htf_raw = (data.order_blocks || []).filter(o => typeof o.type === 'string' && o.type.includes('(')).slice(-CONFIG.MAX_HTF_ZONES);
    const htf = htf_raw.map(z => ({...z, source:'htf'}));
    return {ltf, htf, fvgs};
  }
  let zones = normalizeZones();

  // Paint routine draws rectangles for zones
  function drawAll(){
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0,0,overlay.width,overlay.height);

    // If API missing, bail gracefully
    if (!chart.timeScale().indexToCoordinate || !candleSeries.priceToCoordinate) {
      // fallback: show small legend error
      ctx.fillStyle = 'rgba(255,140,0,0.6)';
      ctx.font = '12px Inter, sans-serif';
      ctx.fillText('Warning: chart API limited — zone overlay disabled', 12, 20);
      return;
    }

    const visibleRange = chart.timeScale().getVisibleRange();
    const firstIdx = Math.max(0, Math.floor(visibleRange.from || 0));
    const lastIdx  = Math.min(candles.length-1, Math.ceil(visibleRange.to || candles.length-1));

    function drawZoneRect(z, visualStyle){
      // created_index may be a plot index (we assume backtester used created_index vs candle array index)
      const startIdx = Math.max(0, (z.created_index !== undefined) ? z.created_index : 0);
      const fromIdx = startIdx;
      const toIdx = Math.min(candles.length-1, startIdx + CONFIG.ZONE_EXTEND_BARS);

      // If out of visible range, skip
      if (toIdx < firstIdx || fromIdx > lastIdx) return;

      // Compute x/y coords
      const x1 = chart.timeScale().indexToCoordinate(fromIdx);
      const x2 = chart.timeScale().indexToCoordinate(toIdx);
      const yTop = candleSeries.priceToCoordinate(z.high);
      const yBot = candleSeries.priceToCoordinate(z.low);

      if (x1 == null || x2 == null || yTop == null || yBot == null) return;

      // Alpha fade depending on how many bars since creation to visible left edge
      const barsSince = Math.max(0, lastIdx - startIdx);
      let alpha = CONFIG.OPACITY_BASE;
      if (barsSince > CONFIG.ZONE_FADE_AFTER_BARS) {
        // fade progressively
        const extra = barsSince - CONFIG.ZONE_FADE_AFTER_BARS;
        alpha = Math.max(0.03, CONFIG.OPACITY_BASE * Math.exp(-0.08 * extra));
      }

      ctx.save();
      ctx.beginPath();
      // rounded rect
      const rx = Math.max(2, Math.min(12, CONFIG.RECT_ROUND));
      const w = Math.max(1, x2 - x1);
      const h = Math.max(1, yBot - yTop);
      // fill
      ctx.fillStyle = visualStyle.fill.replace('ALPHA', alpha.toFixed(3));
      ctx.strokeStyle = visualStyle.border.replace('ALPHA', Math.min(0.9, (alpha*2)).toFixed(3));
      ctx.lineWidth = 1.2;
      // rounded rectangle drawing
      const r = Math.min(rx, w/2, Math.abs(h)/2);
      ctx.moveTo(x1 + r, yTop);
      ctx.arcTo(x1 + w, yTop, x1 + w, yTop + h, r);
      ctx.arcTo(x1 + w, yTop + h, x1, yTop + h, r);
      ctx.arcTo(x1, yTop + h, x1, yTop, r);
      ctx.arcTo(x1, yTop, x1 + w, yTop, r);
      ctx.closePath();
      ctx.globalCompositeOperation = 'source-over';
      ctx.fill();
      ctx.stroke();

      // label
      ctx.font = '11px Inter, sans-serif';
      ctx.fillStyle = 'rgba(200,220,230,' + Math.min(0.95, alpha + 0.08) + ')';
      ctx.fillText((z.type || '').toUpperCase(), x1 + 6, yTop + 14);
      ctx.restore();
    }

    // Build visual styles
    const styles = {
      ltf_long: { fill: 'rgba(34,197,94,ALPHA)', border: 'rgba(34,197,94,ALPHA)' },
      ltf_short:{ fill: 'rgba(239,68,68,ALPHA)', border: 'rgba(239,68,68,ALPHA)' },
      htf_long: { fill: 'rgba(16,185,129,ALPHA)', border:'rgba(16,185,129,ALPHA)' },
      htf_short:{ fill: 'rgba(249,115,22,ALPHA)', border:'rgba(249,115,22,ALPHA)' },
      fvg: { fill: 'rgba(255,215,64,ALPHA)', border: 'rgba(255,215,64,ALPHA)' }
    };

    // Draw LTF zones if enabled
    if (layerState.ltf_ob) {
      zones.ltf.forEach(z => {
        const key = (z.type && z.type.toLowerCase().includes('demand')) ? 'ltf_long' : 'ltf_short';
        drawZoneRect(z, styles[key]);
      });
    }
    // Draw HTF zones
    if (layerState.htf_ob) {
      zones.htf.forEach(z => {
        // try extract direction from type
        const short = (String(z.type||'').toLowerCase().includes('supply'));
        const key = short ? 'htf_short' : 'htf_long';
        drawZoneRect(z, styles[key]);
      });
    }
    // Draw FVGs
    if (layerState.fvg) {
      zones.fvgs.forEach(f => {
        drawZoneRect(f, styles.fvg);
      });
    }

    // Draw HVNs as horizontal bands
    if (layerState.hvn) {
      const hvns = data.hvns || data.hvns || [];
      hvns.forEach(h => {
        const y = candleSeries.priceToCoordinate(h);
        if (y==null) return;
        ctx.save();
        ctx.strokeStyle = 'rgba(100,150,255,0.18)';
        ctx.setLineDash([3,4]);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(0, y); ctx.lineTo(overlay.width / devicePixelRatio, y);
        ctx.stroke();
        ctx.restore();
      });
    }

    // Trades: markers are handled by the library (we set markers on series)
  }

  // Markers for trades (entry/exit)
  const markers = [];
  (data.trades || []).forEach(t => {
    const entryTime = parseTimeVal(t.open_time || t.open_time);
    const exitTime = parseTimeVal(t.close_time || t.close_time);
    const entryPrice = +t.entry_price;
    const exitPrice = +t.close_price;
    const sl = +t.sl; const tp = +t.tp;
    const isLong = (t.side === 'long') || (tp > entryPrice);
    if (entryTime) {
      markers.push({ time: entryTime, position: isLong ? 'belowBar' : 'aboveBar', color: isLong? '#2196F3' : '#FF9800', shape: isLong? 'arrowUp' : 'arrowDown', text: 'ENTRY' });
    }
    if (exitTime) {
      const profitable = (t.pnl || 0) > 0;
      markers.push({ time: exitTime, position: isLong ? 'aboveBar' : 'belowBar', color: profitable? '#4CAF50' : '#F44336', shape: isLong? 'arrowDown' : 'arrowUp', text: `EXIT ${t.outcome || ''}`});
    }
  });
  if (layerStateDefault().trades) candleSeries.setMarkers(markers);

  // toolbar state & handlers
  function layerStateDefault(){
    return { ltf_ob:true, htf_ob:true, fvg:true, hvn:false, trades:true };
  }
  let layerState = layerStateDefault();

  // Connect toolbar UI
  function hookButtons(){
    const mapping = [
      {id:'btn_ltf_ob', key:'ltf_ob'},
      {id:'btn_htf_ob', key:'htf_ob'},
      {id:'btn_fvg', key:'fvg'},
      {id:'btn_hvn', key:'hvn'},
      {id:'btn_trades', key:'trades'}
    ];
    mapping.forEach(m=>{
      const btn = document.getElementById(m.id);
      if (!btn) return;
      btn.onclick = () => {
        layerState[m.key] = !layerState[m.key];
        btn.classList.toggle('active', layerState[m.key]);
        if (m.key === 'trades') {
          if (layerState.trades) candleSeries.setMarkers(markers);
          else candleSeries.setMarkers([]);
        }
        drawAll();
      };
    });
  }

  // Expose a refresh that re-normalizes zones and draws
  function refreshZonesAndDraw(){
    zones = normalizeZones();
    // ensure we cap
    if (zones.ltf.length > CONFIG.MAX_LTF_ZONES) zones.ltf = zones.ltf.slice(-CONFIG.MAX_LTF_ZONES);
    if (zones.htf.length > CONFIG.MAX_HTF_ZONES) zones.htf = zones.htf.slice(-CONFIG.MAX_HTF_ZONES);
    drawAll();
  }

  // Initial UI
  function init(){
    hookButtons();
    refreshZonesAndDraw();
    // Summary
    const s = data.summary || {};
    document.getElementById('summaryBox').innerHTML = `
      <div style="font-weight:700; margin-bottom:6px">Backtest</div>
      Trades: ${s.trades || 0} &nbsp; Win: ${((s.win_rate||0)*100).toFixed(1)}% &nbsp; PnL: ${(s.total_pnl||0).toFixed(2)}
    `;
  }

  // redraw on visible range change / scroll / resize
  chart.timeScale().subscribeVisibleTimeRangeChange(() => { drawAll(); });
  chart.subscribeCrosshairMove(() => {}); // keep chart internal updated
  chart.subscribeClick(() => {}); // placeholder

  // also periodically update overlay (handles small model changes)
  setInterval(()=>{ drawAll(); }, 800);

  init();

})();
</script>
</body>
</html>
'''

def main(in_json='backtest_plot_data.json', out_html='backtest_upgraded.html'):
    p = Path(in_json)
    if not p.exists():
        print(f"Error: {in_json} not found.")
        return
    raw = json.loads(p.read_text(encoding='utf-8'))
    html = TEMPLATE.replace("%%PLOT_JSON%%", json.dumps(raw))
    Path(out_html).write_text(html, encoding='utf-8')
    print(f"Successfully generated upgraded chart: {out_html}")
    print(f"  - Candles: {len(raw.get('candles',[]))}")
    print(f"  - Trades: {len(raw.get('trades',[]))}")
    print(f"  - Zones: {len(raw.get('order_blocks',[])) + len(raw.get('fvgs',[]))}")

# if __name__ == "__main__":
#     if len(sys.argv) >= 3:
#         main(sys.argv[1], sys.argv[2])
#     elif len(sys.argv) == 2:
#         main(sys.argv[1], 'backtest_upgraded.html')
#     else:
#         main()


if __name__ == "__main__":
    main('./out/WELSPUNLIV_EMA_Stacked_Pullback_Long_plot_data.json','./out/WELSPUNLIV_EMA_Stacked_Pullback_Long_.html')
