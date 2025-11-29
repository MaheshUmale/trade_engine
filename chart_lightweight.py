import json
from pathlib import Path

INPUT_JSON = "backtest_plot_data.json"
OUTPUT_HTML = "backtest_chart.html"

HTML_TEMPLATE = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Backtest Chart — Fixed</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  html,body { height:100%; margin:0; background:#0b0f14; color:#cfe3eb; font-family:Inter,Arial,sans-serif; }
  #root { position:relative; height:100vh; width:100vw; }
  #chart { position:absolute; inset:0; }
  .toolbar { position:absolute; left:12px; top:12px; z-index:60; background:rgba(8,12,16,0.7); padding:8px; border-radius:8px; display:flex; gap:8px; }
  .tool-btn { background:transparent; border:1px solid rgba(255,255,255,0.04); color:#9fb6c6; padding:6px 8px; cursor:pointer; border-radius:6px; font-weight:600; font-size:13px; }
  .tool-btn.active{ background:rgba(255,255,255,0.03); color:#eaf6ff; }
  .summary { position:absolute; right:12px; top:12px; z-index:60; background:rgba(6,10,14,0.6); padding:10px; border-radius:8px; color:#9fb6c6; font-size:13px; }
  canvas.overlay { position:absolute; left:0; top:0; z-index:50; pointer-events:none; }
</style>
</head>
<body>
<div id="root">
  <div id="chart"></div>
  <div class="toolbar" id="toolbar">
    <button class="tool-btn active" id="btn_ob">OB</button>
    <button class="tool-btn active" id="btn_fvg">FVG</button>
    <button class="tool-btn active" id="btn_hvn">HVN</button>
    <button class="tool-btn active" id="btn_trades">Trades</button>
  </div>
  <div class="summary" id="summary">Loading...</div>
</div>

<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<script>
const raw = %%PLOT_JSON%%;

// small helpers
const parseTime = t => (typeof t === 'number') ? t : Math.floor(new Date(t).getTime()/1000);

// build candles array
const candles = (raw.candles || []).map(c => ({
  time: parseTime(c.time),
  open: +c.open, high: +c.high, low: +c.low, close: +c.close, volume: +(c.volume||0)
})).sort((a,b)=>a.time-b.time);

const chart = LightweightCharts.createChart(document.getElementById('chart'), {
  layout: { backgroundColor:'#0b0f14', textColor:'#cfe3eb' },
  grid: { vertLines:{color:'#0f1720'}, horzLines:{color:'#0f1720'} },
  timeScale: { timeVisible: true, rightOffset: 8 },
  crosshair: { mode: LightweightCharts.CrosshairMode.Normal }
});

const candleSeries = chart.addCandlestickSeries({
  upColor:'#26a69a', downColor:'#ef5350', wickUpColor:'#26a69a', wickDownColor:'#ef5350', borderVisible:false
});
candleSeries.setData(candles);

// overlay canvas
const overlay = document.createElement('canvas');
overlay.className = 'overlay';
document.getElementById('chart').appendChild(overlay);
const ctx = overlay.getContext('2d');

function resizeOverlay(){
  const root = document.getElementById('chart');
  const rect = root.getBoundingClientRect();
  overlay.style.width = rect.width + 'px';
  overlay.style.height = rect.height + 'px';
  overlay.width = Math.max(1, Math.floor(rect.width * devicePixelRatio));
  overlay.height = Math.max(1, Math.floor(rect.height * devicePixelRatio));
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
}
window.addEventListener('resize', ()=>{ chart.resize(document.getElementById('chart').clientWidth, document.getElementById('chart').clientHeight); resizeOverlay(); drawAll(); });
resizeOverlay();

// config
const CONFIG = {
  ZONE_EXTEND_BARS: 20,
  FVG_EXTEND_BARS: 12,
  FADE_AFTER_BARS: 16,
  OPACITY_BASE: 0.14,
  MAX_LTF_ZONES: 12,
  MAX_HTF_ZONES: 8
};

// layer toggles
const layerState = { ob:true, fvg:true, hvn:true, trades:true };
document.getElementById('btn_ob').onclick = () => toggleLayer('ob', 'btn_ob');
document.getElementById('btn_fvg').onclick = () => toggleLayer('fvg', 'btn_fvg');
document.getElementById('btn_hvn').onclick = () => toggleLayer('hvn', 'btn_hvn');
document.getElementById('btn_trades').onclick = () => toggleLayer('trades', 'btn_trades');

function toggleLayer(key, btnId){
  layerState[key] = !layerState[key];
  document.getElementById(btnId).classList.toggle('active', layerState[key]);
  drawAll();
}

// prepare zones with safety checks
function normalizeZone(z){
  if(!z) return null;
  if(typeof z.high === 'undefined' || typeof z.low === 'undefined') return null;
  const ci = (typeof z.created_index !== 'undefined' && z.created_index !== null) ? parseInt(z.created_index) : null;
  const ct = z.created_time || null;
  return { type: String(z.type||''), high: +z.high, low: +z.low, created_index: ci, created_time: ct, strength: + (z.strength||0) };
}

// build arrays
const orderBlocks = (raw.order_blocks||[]).map(normalizeZone).filter(Boolean);
const fvgs = (raw.fvgs||[]).map(normalizeZone).filter(Boolean);
const hvns = (raw.hvns||[]).map(h=>+h).filter(h=>Number.isFinite(h));
const trades = raw.trades || [];
const summary = raw.summary || { trades:0, total_pnl:0, win_rate:0, max_drawdown:0 };

// map created_time -> index if created_index missing
const candle_dt_list = candles.map(c => new Date(c.time*1000));

function mapCreatedIndex(z){
  if(z.created_index !== null && !isNaN(z.created_index)){
    // clamp
    z.created_index = Math.max(0, Math.min(candles.length-1, z.created_index));
    return z;
  }
  if(z.created_time){
    const dt = new Date(z.created_time);
    if(!isNaN(dt.getTime())){
      // find nearest index
      let pos = candle_dt_list.findIndex(d => d >= dt);
      if(pos === -1) pos = candle_dt_list.length - 1;
      if(pos > 0){
        // choose nearer of pos-1 and pos
        const a = Math.abs(dt - candle_dt_list[pos-1]);
        const b = Math.abs(candle_dt_list[pos] - dt);
        pos = (a <= b) ? pos-1 : pos;
      }
      z.created_index = Math.max(0, Math.min(candles.length-1, pos));
      return z;
    }
  }
  // fallback: near end
  z.created_index = Math.max(0, candles.length - 5);
  return z;
}

// apply mapping and filter extreme off-screen price zones
const visible_low = Math.min(...candles.map(c=>c.low));
const visible_high = Math.max(...candles.map(c=>c.high));
function in_visible_price(z, pad_pct=0.5){
  const span = Math.max(visible_high - visible_low, 1e-6);
  const pad = span * pad_pct;
  if(z.high < visible_low - pad) return false;
  if(z.low > visible_high + pad) return false;
  return true;
}

const ltf_zones = orderBlocks.map(mapCreatedIndex).filter(z=> in_visible_price(z)).slice(-CONFIG.MAX_LTF_ZONES);
const fvg_zones = fvgs.map(mapCreatedIndex).filter(z=> in_visible_price(z)).slice(-Math.max(8, CONFIG.FVG_EXTEND_BARS));
const hvn_list = Array.from(new Set(hvns)).filter(h=> h >= visible_low - (visible_high-visible_low)*0.6 && h <= visible_high + (visible_high-visible_low)*0.6);

// trade markers
const markers = [];
trades.forEach(t=>{
  const entryT = (t.open_time) ? Math.floor(new Date(t.open_time).getTime()/1000) : null;
  const exitT = (t.close_time) ? Math.floor(new Date(t.close_time).getTime()/1000) : null;
  if(entryT) markers.push({ time: entryT, position:'belowBar', color:'#2196F3', shape:'arrowUp', text:'ENTRY' });
  if(exitT) markers.push({ time: exitT, position:'aboveBar', color: (t.pnl||0) >= 0 ? '#4CAF50' : '#F44336', shape:'arrowDown', text:'EXIT ' + (t.outcome||'') });
});
candleSeries.setMarkers(layerState.trades ? markers : []);

// coordinate helpers (safely handle API absence)
function idxToX(idx){
  try{
    return chart.timeScale().indexToCoordinate(idx);
  }catch(e){ return null; }
}
function priceToY(price){
  try{
    return candleSeries.priceToCoordinate(price);
  }catch(e){ return null; }
}

// draw single zone rect
function drawZoneRect(z, style){
  const startIdx = z.created_index;
  if(startIdx === null || startIdx < 0 || startIdx >= candles.length) return;
  const endIdx = Math.min(candles.length-1, startIdx + (style.extendBars || CONFIG.ZONE_EXTEND_BARS));
  const x1 = idxToX(startIdx);
  const x2 = idxToX(endIdx);
  const yTop = priceToY(z.high);
  const yBot = priceToY(z.low);
  if([x1,x2,yTop,yBot].some(v=>v===null || typeof v === 'undefined' || isNaN(v))) return;
  const w = x2 - x1;
  const h = yBot - yTop;
  if(w <= 0 || h <= 0) return;

  // fade alpha based on age (bars since creation to visible right)
  const visibleRange = chart.timeScale().getVisibleRange ? chart.timeScale().getVisibleRange() : null;
  let lastVisibleIdx = candles.length - 1;
  if(visibleRange && typeof visibleRange.to === 'number') lastVisibleIdx = Math.min(candles.length-1, Math.ceil(visibleRange.to));
  const barsSince = Math.max(0, lastVisibleIdx - startIdx);
  let alpha = CONFIG.OPACITY_BASE;
  if(barsSince > CONFIG.FADE_AFTER_BARS){
    const extra = barsSince - CONFIG.FADE_AFTER_BARS;
    alpha = Math.max(0.03, CONFIG.OPACITY_BASE * Math.exp(-0.08 * extra));
  }

  ctx.save();
  // fill
  ctx.beginPath();
  const r = Math.min(8, Math.abs(h)/2, w/6);
  // rounded rect path
  const x = x1, y = yTop;
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
  ctx.fillStyle = style.fill.replace('ALPHA', alpha.toFixed(3));
  ctx.strokeStyle = style.border.replace('ALPHA', Math.min(0.9, (alpha*2)).toFixed(3));
  ctx.lineWidth = 1.2;
  ctx.fill();
  ctx.stroke();

  // label
  ctx.font = '11px Inter, sans-serif';
  ctx.fillStyle = 'rgba(220,235,240,' + Math.min(0.95, alpha + 0.06) + ')';
  const label = (z.type || '').toUpperCase();
  ctx.fillText(label, x + 6, y + 14);
  ctx.restore();
}

// draw HVN line
function drawHVN(price){
  const y = priceToY(price);
  if(y === null || isNaN(y)) return;
  ctx.save();
  ctx.beginPath();
  ctx.setLineDash([3,4]);
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = 'rgba(100,150,255,0.18)';
  ctx.moveTo(0, y); ctx.lineTo(overlay.width / devicePixelRatio, y);
  ctx.stroke();
  ctx.restore();
}

// draw everything
function drawAll(){
  // clear
  ctx.clearRect(0,0,overlay.width,overlay.height);

  // guard: ensure API available
  if(!chart.timeScale().indexToCoordinate || !candleSeries.priceToCoordinate){
    ctx.fillStyle = 'rgba(255,140,0,0.6)';
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText('Warning: chart API limited — overlay disabled', 12, 20);
    return;
  }

  // draw HVN first (under zones)
  if(layerState.hvn){
    hvn_list.forEach(h=> drawHVN(h));
  }

  // draw LTF zones
  if(layerState.ob){
    ltf_zones.forEach(z => {
      const isDemand = String(z.type||'').toLowerCase().includes('demand') || String(z.type||'').toLowerCase().includes('long');
      const style = isDemand ? { fill:'rgba(34,197,94,ALPHA)', border:'rgba(34,197,94,ALPHA)', extendBars: CONFIG.ZONE_EXTEND_BARS } : { fill:'rgba(239,68,68,ALPHA)', border:'rgba(239,68,68,ALPHA)', extendBars: CONFIG.ZONE_EXTEND_BARS };
      drawZoneRect(z, style);
    });
  }

  // draw FVGs
  if(layerState.fvg){
    fvg_zones.forEach(f => {
      const style = { fill:'rgba(255,215,64,ALPHA)', border:'rgba(255,215,64,ALPHA)', extendBars: CONFIG.FVG_EXTEND_BARS };
      drawZoneRect(f, style);
    });
  }
}

// redraw on timeScale visible change
chart.timeScale().subscribeVisibleTimeRangeChange(()=> { drawAll(); });
chart.subscribeCrosshairMove(()=>{}); // keep internal state updated

// periodic draw to handle small API updates
setInterval(()=>{ drawAll(); }, 700);

// initial sizing & draw
resizeOverlay();
chart.timeScale().fitContent();
drawAll();

// set summary
document.getElementById('summary').innerHTML = `<b>Backtest</b><br>Trades: ${summary.trades || 0} &nbsp; Win: ${((summary.win_rate||0)*100).toFixed(1)}% &nbsp; PnL: ${(summary.total_pnl||0).toFixed(2)}`;

// trades toggle behavior updates markers
document.getElementById('btn_trades').addEventListener('click', ()=>{
  const active = document.getElementById('btn_trades').classList.toggle('active');
  layerState.trades = active;
  candleSeries.setMarkers(active ? markers : []);
});
</script>
</body>
</html>
'''

def main(in_json=INPUT_JSON, out_html=OUTPUT_HTML):
    p = Path(in_json)
    if not p.exists():
        print(f"Error: {in_json} not found.")
        return
    raw = json.loads(p.read_text(encoding='utf-8'))
    html = HTML_TEMPLATE.replace("%%PLOT_JSON%%", json.dumps(raw))
    Path(out_html).write_text(html, encoding='utf-8')
    print(f"Generated: {out_html}")

if __name__ == "__main__":
    
    filename =  ['63MOONS_9EMA_CONTINUATION',   '63MOONS_EMA_Stacked_Pullback_Long','63MOONS_VWAP_Reclaim_Long','GRSE_9EMA_CONTINUATION','GRSE_EMA_Stacked_Pullback_Long','GRSE_VWAP_Reclaim_Long','M_M_9EMA_CONTINUATION','M_M_EMA_Stacked_Pullback_Long','M_M_VWAP_Reclaim_Long','WELSPUNLIV_9EMA_CONTINUATION','WELSPUNLIV_EMA_Stacked_Pullback_Long','WELSPUNLIV_VWAP_Reclaim_Long']

    for name in filename :
        jsonFile ='./out/'+name+'_plot_data.json'
        htmlName = './out/'+name+'.html'
        main(jsonFile,htmlName)


    
