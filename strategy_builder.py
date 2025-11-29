import http.server
import socketserver
import json
import os
import ast
import re
from pathlib import Path

PORT = 8000
STRATEGIES_FILE = "strategies.json"
RULE_ENGINE_FILE = "rule_engine.py"

class StrategyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_html().encode())
        elif self.path == '/api/strategies':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                with open(STRATEGIES_FILE, 'r') as f:
                    self.wfile.write(f.read().encode())
            except FileNotFoundError:
                self.wfile.write(b'[]')
        elif self.path == '/api/indicators':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            indicators = self.get_available_indicators()
            self.wfile.write(json.dumps(indicators).encode())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/strategies':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                strategies = json.loads(post_data)
                with open(STRATEGIES_FILE, 'w') as f:
                    json.dump(strategies, f, indent=2)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif self.path == '/api/fix_indicator':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            indicator_name = data.get('name')
            code = data.get('code')
            
            if self.inject_indicator(indicator_name, code):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            else:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'{"error": "Failed to inject code"}')

    def get_available_indicators(self):
        """Parse rule_engine.py to find methods in IndicatorEngine"""
        try:
            with open(RULE_ENGINE_FILE, 'r') as f:
                tree = ast.parse(f.read())
            
            indicators = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == 'IndicatorEngine':
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                            args = [a.arg for a in item.args.args if a.arg != 'self']
                            indicators.append({"name": item.name, "args": args})
            return indicators
        except Exception as e:
            print(f"Error parsing rule engine: {e}")
            return []

    def inject_indicator(self, name, code):
        """Inject new indicator method into IndicatorEngine class"""
        try:
            with open(RULE_ENGINE_FILE, 'r') as f:
                content = f.read()
            
            # Find the end of IndicatorEngine class
            class_match = re.search(r'class IndicatorEngine:.*?(?=\nclass|\Z)', content, re.DOTALL)
            if not class_match:
                return False
            
            # Indent the code
            indented_code = "\n    " + code.replace("\n", "\n    ")
            
            # Insert before the next class or end of file
            # We look for the last method in IndicatorEngine and append after it
            # A simpler way is to find the class block and append to it.
            # But regex is tricky with nested classes.
            # Let's assume standard formatting and look for "class PatternEngine" which usually follows
            
            split_marker = "class PatternEngine"
            if split_marker in content:
                parts = content.split(split_marker)
                new_content = parts[0].rstrip() + "\n\n    @staticmethod" + indented_code + "\n\n" + split_marker + parts[1]
            else:
                # Append to end of file if PatternEngine not found (unlikely based on file view)
                # Or find the end of indentation?
                return False

            with open(RULE_ENGINE_FILE, 'w') as f:
                f.write(new_content)
            return True
        except Exception as e:
            print(f"Injection error: {e}")
            return False

    def get_html(self):
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Strategy Builder</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 0; display: flex; height: 100vh; background: #1e1e1e; color: #ccc; }
        #sidebar { width: 250px; background: #252526; padding: 10px; border-right: 1px solid #333; overflow-y: auto; }
        #main { flex: 1; padding: 20px; overflow-y: auto; }
        .strategy-item { padding: 8px; cursor: pointer; border-bottom: 1px solid #333; }
        .strategy-item:hover { background: #37373d; }
        .strategy-item.active { background: #094771; color: white; }
        h2 { margin-top: 0; color: #fff; }
        label { display: block; margin-top: 10px; color: #aaa; }
        input, select, textarea { width: 100%; padding: 6px; background: #3c3c3c; border: 1px solid #555; color: #fff; margin-top: 4px; }
        button { background: #0e639c; color: white; border: none; padding: 8px 16px; cursor: pointer; margin-top: 10px; }
        button:hover { background: #1177bb; }
        .rule-group { border: 1px solid #444; padding: 10px; margin-top: 10px; background: #2d2d2d; }
        .rule { display: flex; gap: 10px; margin-top: 5px; align-items: center; }
        .rule input { width: auto; flex: 1; }
        .btn-sm { padding: 2px 6px; font-size: 12px; margin: 0; }
        .btn-danger { background: #a1260d; }
        .warning { color: #cca700; font-size: 12px; margin-top: 4px; display: flex; align-items: center; gap: 5px; }
        .fix-btn { background: #cca700; color: black; padding: 2px 6px; font-size: 10px; border-radius: 2px; cursor: pointer; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h3 style="color:white">Strategies</h3>
        <div id="strategyList"></div>
        <button onclick="addNewStrategy()">+ New Strategy</button>
        <button onclick="saveStrategies()" style="background:#2da042; width:100%">Save All</button>
    </div>
    <div id="main">
        <div id="editor" style="display:none">
            <h2 id="editTitle">Edit Strategy</h2>
            <label>Name</label>
            <input type="text" id="stratName" onchange="updateCurrentStrategy()">
            <label>Entry Direction</label>
            <select id="stratDir" onchange="updateCurrentStrategy()">
                <option value="LONG">LONG</option>
                <option value="SHORT">SHORT</option>
            </select>
            <label>Timeframes (comma sep)</label>
            <input type="text" id="stratTf" onchange="updateCurrentStrategy()">
            
            <h3>Logic</h3>
            <div id="logicBuilder"></div>
        </div>
        <div id="welcome" style="text-align:center; margin-top:100px;">
            <h1>Select a strategy to edit</h1>
        </div>
    </div>

    <!-- Datalists for auto-complete -->
    <datalist id="indicatorList"></datalist>
    <datalist id="patternList">
        <option value="bullish_engulfing">
        <option value="bearish_engulfing">
        <option value="hammer">
        <option value="pinbar">
        <option value="doji">
        <option value="inside_bar">
    </datalist>

    <script>
        let strategies = [];
        let currentStratIndex = -1;
        let availableIndicators = [];

        const STANDARD_INDICATORS = {
            'rsi': 'def rsi(ohlc, period=14):\\n    closes = [c["close"] for c in ohlc]\\n    if len(closes) < period + 1: return 50.0\\n    deltas = [closes[i]-closes[i-1] for i in range(1, len(closes))]\\n    avg_up = sum([x for x in deltas[-period:] if x > 0]) / period\\n    avg_down = sum([-x for x in deltas[-period:] if x < 0]) / period\\n    if avg_down == 0: return 100.0\\n    rs = avg_up / avg_down\\n    return 100.0 - (100.0 / (1.0 + rs))',
            'macd': 'def macd(ohlc, fast=12, slow=26, signal=9):\\n    # Placeholder implementation\\n    return 0.0',
            'bollinger_upper': 'def bollinger_upper(ohlc, period=20, std_dev=2.0):\\n    closes = [c["close"] for c in ohlc[-period:]]\\n    if not closes: return 0.0\\n    avg = sum(closes)/len(closes)\\n    variance = sum([(x-avg)**2 for x in closes]) / len(closes)\\n    import math\\n    sd = math.sqrt(variance)\\n    return avg + std_dev * sd',
            'bollinger_lower': 'def bollinger_lower(ohlc, period=20, std_dev=2.0):\\n    closes = [c["close"] for c in ohlc[-period:]]\\n    if not closes: return 0.0\\n    avg = sum(closes)/len(closes)\\n    variance = sum([(x-avg)**2 for x in closes]) / len(closes)\\n    import math\\n    sd = math.sqrt(variance)\\n    return avg - std_dev * sd',
            'keltner_upper': 'def keltner_upper(ohlc, period=20, mult=2.0):\\n    # Requires ATR\\n    return 0.0',
            'donchian_high': 'def donchian_high(ohlc, period=20):\\n    highs = [c["high"] for c in ohlc[-period:]]\\n    return max(highs) if highs else 0.0',
            'donchian_low': 'def donchian_low(ohlc, period=20):\\n    lows = [c["low"] for c in ohlc[-period:]]\\n    return min(lows) if lows else 0.0'
        };

        async function init() {
            const [stratRes, indRes] = await Promise.all([
                fetch('/api/strategies'),
                fetch('/api/indicators')
            ]);
            strategies = await stratRes.json();
            availableIndicators = await indRes.json();
            
            // Populate indicator datalist
            const dl = document.getElementById('indicatorList');
            const builtins = ['close', 'open', 'high', 'low', 'volume', 'vwap', 'rvol', 'ema', 'atr'];
            const allInds = new Set([...builtins, ...availableIndicators.map(i => i.name)]);
            allInds.forEach(ind => {
                const opt = document.createElement('option');
                opt.value = ind;
                dl.appendChild(opt);
            });

            renderList();
        }

        function renderList() {
            const list = document.getElementById('strategyList');
            list.innerHTML = '';
            strategies.forEach((s, i) => {
                const div = document.createElement('div');
                div.className = `strategy-item ${i === currentStratIndex ? 'active' : ''}`;
                div.textContent = s.strategy_name;
                div.onclick = () => selectStrategy(i);
                list.appendChild(div);
            });
        }

        function selectStrategy(index) {
            currentStratIndex = index;
            const s = strategies[index];
            document.getElementById('welcome').style.display = 'none';
            document.getElementById('editor').style.display = 'block';
            document.getElementById('stratName').value = s.strategy_name;
            document.getElementById('stratDir').value = s.entry_direction;
            document.getElementById('stratTf').value = (s.timeframes || []).join(', ');
            
            renderLogic(s.logic, document.getElementById('logicBuilder'));
            renderList();
        }

        function renderLogic(node, container) {
            container.innerHTML = '';
            if (!node) return;

            if (node.operator) {
                // Group
                const div = document.createElement('div');
                div.className = 'rule-group';
                // We need to pass 'node' reference correctly. 
                // Since we are re-rendering, we can't easily bind 'node' in HTML string.
                // We'll use DOM elements directly.
                
                const header = document.createElement('div');
                header.style.display = 'flex';
                header.style.justifyContent = 'space-between';
                header.style.marginBottom = '5px';
                
                const opSelect = document.createElement('select');
                opSelect.style.width = '80px';
                opSelect.innerHTML = `<option value="AND">AND</option><option value="OR">OR</option>`;
                opSelect.value = node.operator;
                opSelect.onchange = (e) => { node.operator = e.target.value; };
                
                const addBtn = document.createElement('button');
                addBtn.className = 'btn-sm';
                addBtn.textContent = '+ Rule';
                addBtn.onclick = () => {
                    node.rules = node.rules || [];
                    node.rules.push({ indicator: 'close', condition: '>', value: 'ema20' });
                    renderLogic(node, container);
                };

                const addPatBtn = document.createElement('button');
                addPatBtn.className = 'btn-sm';
                addPatBtn.textContent = '+ Pattern';
                addPatBtn.style.marginLeft = '5px';
                addPatBtn.onclick = () => {
                    node.rules = node.rules || [];
                    node.rules.push({ pattern: 'bullish_engulfing' });
                    renderLogic(node, container);
                };

                header.appendChild(opSelect);
                header.appendChild(addBtn);
                header.appendChild(addPatBtn);
                div.appendChild(header);

                (node.rules || []).forEach((rule, i) => {
                    const ruleDiv = document.createElement('div');
                    renderLogic(rule, ruleDiv);
                    
                    // Add delete button for rule
                    const delBtn = document.createElement('button');
                    delBtn.className = 'btn-sm btn-danger';
                    delBtn.textContent = 'x';
                    delBtn.style.marginLeft = '5px';
                    delBtn.onclick = () => {
                        node.rules.splice(i, 1);
                        renderLogic(node, container);
                    };
                    
                    ruleDiv.style.display = 'flex';
                    ruleDiv.style.alignItems = 'center';
                    ruleDiv.appendChild(delBtn);
                    
                    div.appendChild(ruleDiv);
                });
                container.appendChild(div);
            } else {
                // Leaf Rule
                const div = document.createElement('div');
                div.className = 'rule';
                
                if (node.pattern) {
                    div.innerHTML = `<span>Pattern:</span>`;
                    const patInput = document.createElement('input');
                    patInput.value = node.pattern;
                    patInput.setAttribute('list', 'patternList');
                    patInput.onchange = (e) => { node.pattern = e.target.value; };
                    div.appendChild(patInput);
                } else {
                    // Indicator
                    const indInput = document.createElement('input');
                    indInput.placeholder = 'Indicator (e.g. close)';
                    indInput.value = node.indicator || '';
                    indInput.setAttribute('list', 'indicatorList');
                    indInput.onchange = (e) => { 
                        node.indicator = e.target.value; 
                        checkIndicator(node.indicator, div);
                    };

                    const condSelect = document.createElement('select');
                    condSelect.style.width = '60px';
                    ['>', '<', '>=', '<=', '=='].forEach(op => {
                        const opt = document.createElement('option');
                        opt.value = op;
                        opt.textContent = op;
                        if (node.condition === op) opt.selected = true;
                        condSelect.appendChild(opt);
                    });
                    condSelect.onchange = (e) => { node.condition = e.target.value; };

                    const valInput = document.createElement('input');
                    valInput.placeholder = 'Value (e.g. vwap)';
                    valInput.value = node.value || '';
                    valInput.setAttribute('list', 'indicatorList'); // reuse indicator list for value too
                    valInput.onchange = (e) => { node.value = isNaN(e.target.value) ? e.target.value : Number(e.target.value); };

                    div.appendChild(indInput);
                    div.appendChild(condSelect);
                    div.appendChild(valInput);

                    checkIndicator(node.indicator, div);
                }
                container.appendChild(div);
            }
        }

        function checkIndicator(name, container) {
            // Remove existing warning
            const existing = container.querySelector('.warning');
            if (existing) existing.remove();

            // Check if indicator exists (strip params like ema9 -> ema)
            const baseName = name.replace(/[0-9]+$/, '').split('(')[0];
            const exists = availableIndicators.some(i => i.name === baseName) || 
                           ['close','open','high','low','volume','time'].includes(baseName) ||
                           baseName.startsWith('ema'); // hardcoded support in rule_engine

            if (!exists) {
                const warn = document.createElement('div');
                warn.className = 'warning';
                warn.innerHTML = `⚠️ Missing`;
                
                if (STANDARD_INDICATORS[baseName]) {
                    const fixBtn = document.createElement('span');
                    fixBtn.className = 'fix-btn';
                    fixBtn.textContent = 'Fix';
                    fixBtn.onclick = () => fixIndicator(baseName, STANDARD_INDICATORS[baseName]);
                    warn.appendChild(fixBtn);
                }
                container.appendChild(warn);
            }
        }

        async function fixIndicator(name, code) {
            if (!confirm(`Inject code for ${name}?`)) return;
            const res = await fetch('/api/fix_indicator', {
                method: 'POST',
                body: JSON.stringify({ name, code })
            });
            if (res.ok) {
                alert('Injected! Reloading...');
                location.reload();
            } else {
                alert('Error injecting code');
            }
        }

        function updateCurrentStrategy() {
            const s = strategies[currentStratIndex];
            s.strategy_name = document.getElementById('stratName').value;
            s.entry_direction = document.getElementById('stratDir').value;
            s.timeframes = document.getElementById('stratTf').value.split(',').map(t => t.trim());
            renderList();
        }

        function addNewStrategy() {
            strategies.push({
                strategy_name: "New_Strategy",
                enabled: true,
                entry_direction: "LONG",
                timeframes: ["1m"],
                logic: { operator: "AND", rules: [] }
            });
            selectStrategy(strategies.length - 1);
        }

        async function saveStrategies() {
            const res = await fetch('/api/strategies', {
                method: 'POST',
                body: JSON.stringify(strategies)
            });
            if (res.ok) alert('Saved!');
            else alert('Error saving');
        }

        init();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"Starting Strategy Builder on http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), StrategyHandler) as httpd:
        httpd.serve_forever()
