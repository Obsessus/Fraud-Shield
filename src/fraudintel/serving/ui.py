"""Frontend UI for the Fraud Intelligence service (Stage 19).

A single self-contained HTML page (no external dependencies) that lets a human
load a real transaction, edit its key feature values, and ask the model to score
it. Served by the FastAPI app at ``GET /``. The page talks to ``/sample`` (to load
a real transaction) and ``/predict`` (to score it).
"""

from __future__ import annotations

UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Fraud Intelligence — Check a Transaction</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --fg:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8; --bad:#f87171; --good:#4ade80; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg); color:var(--fg); }
  header { padding:24px 32px; border-bottom:1px solid #334155; }
  header h1 { margin:0; font-size:20px; }
  header p { margin:6px 0 0; color:var(--muted); font-size:13px; }
  main { max-width:920px; margin:24px auto; padding:0 20px; }
  .card { background:var(--card); border:1px solid #334155; border-radius:12px; padding:20px; margin-bottom:20px; }
  .row { display:flex; flex-wrap:wrap; gap:16px; align-items:center; }
  .badge { font-size:12px; padding:4px 10px; border-radius:999px; background:#334155; color:var(--muted); }
  .badge.actual-fraud { background:rgba(248,113,113,.18); color:var(--bad); }
  .badge.actual-clean { background:rgba(74,222,128,.18); color:var(--good); }
  button { background:var(--accent); color:#06283d; border:0; border-radius:8px; padding:10px 16px; font-weight:600; cursor:pointer; }
  button.secondary { background:#334155; color:var(--fg); }
  button:hover { opacity:.9; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:12px; margin-top:14px; }
  label.field { display:flex; flex-direction:column; font-size:12px; color:var(--muted); gap:4px; }
  label.field input { background:#0b1220; border:1px solid #334155; color:var(--fg); border-radius:6px; padding:8px; font-size:14px; }
  .verdict { font-size:28px; font-weight:800; padding:18px; border-radius:12px; text-align:center; }
  .verdict.fraud { background:rgba(248,113,113,.15); color:var(--bad); border:1px solid var(--bad); }
  .verdict.clean { background:rgba(74,222,128,.15); color:var(--good); border:1px solid var(--good); }
  .meta { color:var(--muted); font-size:13px; margin-top:10px; }
  ul.factors { margin:10px 0 0; padding-left:18px; }
  ul.factors li { font-size:13px; margin:4px 0; }
  .hint { color:var(--muted); font-size:12px; margin-top:10px; }
  code { background:#0b1220; padding:2px 6px; border-radius:5px; }
</style>
</head>
<body>
<header>
  <h1>Fraud Intelligence Platform</h1>
  <p>Load a real transaction, tweak its values, and ask the model to score it for fraud.</p>
</header>
<main>
  <div class="card">
    <div class="row">
      <strong id="tid">—</strong>
      <span id="actual" class="badge" style="display:none"></span>
      <span style="flex:1"></span>
      <button id="newBtn">New random transaction</button>
      <button id="clearBtn" class="secondary">Clear to blank</button>
      <button id="checkBtn">Check fraud</button>
    </div>
    <div class="hint">The form shows the model's most influential features. Loaded transactions use their
      <em>real</em> values for all 416 engineered features; when you Clear, missing features default to 0.</div>
    <div id="fields" class="grid"></div>
  </div>

  <div id="resultCard" class="card" style="display:none">
    <div id="verdict" class="verdict">—</div>
    <div id="verdictMeta" class="meta"></div>
    <div id="agreement" class="meta"></div>
    <div class="hint">Top risk factors (SHAP contributions):</div>
    <ul id="factors" class="factors"></ul>
  </div>

  <div id="demoCard" class="card" style="display:none">
    <div class="row">
      <strong>Real-time feature demo</strong>
      <span style="flex:1"></span>
      <button id="demoBtn">Recompute history &rarr; re-predict</button>
    </div>
    <div class="hint">Edit the raw <code>card1</code> / <code>addr1</code> / <code>TransactionAmt</code>
      fields and watch the model's historical-aggregate features get recomputed <em>live</em> from the
      training history (exactly how the offline feature store would), then re-score the transaction.</div>
    <div class="grid">
      <label class="field"><span>card1 (raw identity)</span><input id="dCard1" type="number" step="any" /></label>
      <label class="field"><span>addr1 (raw identity)</span><input id="dAddr1" type="number" step="any" /></label>
      <label class="field"><span>TransactionAmt (raw amount)</span><input id="dAmt" type="number" step="any" /></label>
    </div>
    <div id="demoResult" class="card" style="display:none; margin-top:14px">
      <div id="demoVerdict" class="verdict">—</div>
      <div id="demoMeta" class="meta"></div>
      <div class="hint">How the history features were derived:</div>
      <ul id="demoDerivation" class="factors"></ul>
    </div>
  </div>
</main>

<script>
const state = { features: {}, label: null, tid: null };

function setText(id, t) { document.getElementById(id).textContent = t; }

function renderFields(names, values) {
  const box = document.getElementById('fields');
  box.innerHTML = '';
  names.forEach(name => {
    const lab = document.createElement('label');
    lab.className = 'field';
    const span = document.createElement('span');
    span.textContent = name;
    const inp = document.createElement('input');
    inp.type = 'number'; inp.step = 'any';
    inp.value = (values[name] !== undefined && values[name] !== null) ? values[name] : 0;
    inp.dataset.name = name;
    inp.addEventListener('input', () => {
      state.features[name] = parseFloat(inp.value) || 0;
    });
    lab.appendChild(span); lab.appendChild(inp);
    box.appendChild(lab);
  });
}

function applySample(resp) {
  state.features = resp.features || {};
  state.label = (resp.label !== undefined && resp.label !== null) ? resp.label : null;
  state.tid = resp.transaction_id || 'manual';
  setText('tid', 'Transaction: ' + state.tid);
  const act = document.getElementById('actual');
  if (state.label !== null) {
    act.style.display = 'inline-block';
    act.textContent = 'Actual: ' + (state.label === 1 ? 'FRAUD' : 'CLEAN');
    act.className = 'badge ' + (state.label === 1 ? 'actual-fraud' : 'actual-clean');
  } else {
    act.style.display = 'none';
  }
  renderFields(resp.top_features || [], state.features);
  document.getElementById('resultCard').style.display = 'none';
  showDemoIfAvailable(resp);
}

function showDemoIfAvailable(resp) {
  const demo = document.getElementById('demoCard');
  if (resp.feature_demo) {
    demo.style.display = 'block';
    document.getElementById('dCard1').value = (resp.features['card1'] !== undefined) ? resp.features['card1'] : 0;
    document.getElementById('dAddr1').value = (resp.features['addr1'] !== undefined) ? resp.features['addr1'] : 0;
    document.getElementById('dAmt').value = (resp.features['TransactionAmt'] !== undefined) ? resp.features['TransactionAmt'] : 0;
  } else {
    demo.style.display = 'none';
  }
}

async function loadSample() {
  const r = await fetch('/sample');
  if (!r.ok) { alert('Could not load a sample (is the UI sample data present?).'); return; }
  applySample(await r.json());
}

async function checkFraud() {
  const body = { transaction_id: state.tid || 'manual', features: state.features };
  const r = await fetch('/predict', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)
  });
  if (!r.ok) { alert('Prediction failed: ' + (await r.text())); return; }
  const j = await r.json();
  state.lastScore = j.score;
  const isFraud = j.decision === 'review';
  const v = document.getElementById('verdict');
  v.textContent = isFraud ? 'FRAUD RISK' : 'CLEAN';
  v.className = 'verdict ' + (isFraud ? 'fraud' : 'clean');
  setText('verdictMeta', 'Score ' + (j.score).toFixed(4) + '  vs  threshold ' + (j.threshold).toFixed(4));
  const ag = document.getElementById('agreement');
  if (state.label !== null) {
    const correct = (state.label === 1) === isFraud;
    ag.textContent = 'Model said ' + (isFraud ? 'FRAUD' : 'CLEAN') + ', actual was ' +
      (state.label === 1 ? 'FRAUD' : 'CLEAN') + '  →  ' + (correct ? 'correct' : 'missed');
  } else { ag.textContent = ''; }
  const ul = document.getElementById('factors');
  ul.innerHTML = '';
  (j.risk_factors || []).forEach(f => {
    const li = document.createElement('li');
    li.textContent = f.feature + '  (' + (f.contribution >= 0 ? '+' : '') + f.contribution.toFixed(4) + ')';
    ul.appendChild(li);
  });
  document.getElementById('resultCard').style.display = 'block';
}

function clearBlank() {
  state.features = {};
  state.label = null;
  state.tid = 'manual';
  setText('tid', 'Transaction: manual');
  document.getElementById('actual').style.display = 'none';
  const box = document.getElementById('fields');
  box.querySelectorAll('input').forEach(inp => { inp.value = 0; state.features[inp.dataset.name] = 0; });
  document.getElementById('resultCard').style.display = 'none';
}

async function runDemo() {
  const c = parseFloat(document.getElementById('dCard1').value);
  const a = parseFloat(document.getElementById('dAddr1').value);
  const m = parseFloat(document.getElementById('dAmt').value);
  const demoReq = {
    features: state.features,
    card1: isNaN(c) ? null : c,
    addr1: isNaN(a) ? null : a,
    amount: isNaN(m) ? null : m,
  };
  const r1 = await fetch('/demo/features', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(demoReq)
  });
  if (!r1.ok) { alert('Feature derivation failed: ' + (await r1.text())); return; }
  const dj = await r1.json();
  const r2 = await fetch('/predict', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ transaction_id: state.tid || 'manual', features: dj.features })
  });
  if (!r2.ok) { alert('Prediction failed: ' + (await r2.text())); return; }
  const j = await r2.json();
  const isFraud = j.decision === 'review';
  const v = document.getElementById('demoVerdict');
  v.textContent = isFraud ? 'FRAUD RISK' : 'CLEAN';
  v.className = 'verdict ' + (isFraud ? 'fraud' : 'clean');
  const base = (state.lastScore !== undefined) ? state.lastScore : null;
  setText('demoMeta', 'Score ' + j.score.toFixed(4) + '  vs  threshold ' + j.threshold.toFixed(4)
    + (base !== null ? '   (base score was ' + base.toFixed(4) + ')' : ''));
  const ul = document.getElementById('demoDerivation');
  ul.innerHTML = '';
  (dj.derivation || []).forEach(d => {
    const li = document.createElement('li');
    li.textContent = d.note;
    ul.appendChild(li);
  });
  document.getElementById('demoResult').style.display = 'block';
}

document.getElementById('newBtn').addEventListener('click', loadSample);
document.getElementById('clearBtn').addEventListener('click', clearBlank);
document.getElementById('checkBtn').addEventListener('click', checkFraud);
document.getElementById('demoBtn').addEventListener('click', runDemo);
loadSample();
</script>
</body>
</html>
"""
