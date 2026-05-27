/**
 * upload.js — FraudShield AI
 * Handles: CSV drag-drop, batch predict, single predict, results table
 *
 * API_BASE auto-detection:
 *   - When served by FastAPI (same origin) → uses relative URLs
 *   - When served by Live Server (dev)     → uses localhost:8000
 */

// ── Auto-detect API base URL ──────────────────────────────────────────
const API_BASE = (() => {
  const port = window.location.port;
  // Production same-origin (Render, Vercel, Docker) → relative paths
  if (!port || (port !== '5500' && port !== '5501' && port !== '3000')) {
    return '';
  }
  // Dev mode: Live Server on 5500/5501, API on 8000
  return 'http://localhost:8000';
})();


// ── Helpers ────────────────────────────────────────────────────────────────
function show(el) { if (el) el.style.display = el._displayType || 'flex'; }
function hide(el) { if (el) { el._displayType = el.style.display || 'flex'; el.style.display = 'none'; } }
function showBlock(el) { if (el) el.style.display = 'block'; }

function riskBadge(label, conf) {
  const fraud = label === 'FRAUD';
  const high  = conf === 'HIGH';
  const cls   = fraud ? (high ? 'badge-fraud' : 'badge-medium') : 'badge-safe';
  const icon  = fraud ? (high ? '🚨' : '⚠️') : '✅';
  return `<span class="risk-badge ${cls}">${icon} ${label}</span>`;
}

function showErr(msg) {
  const el = document.getElementById('upload-error');
  if (el) { el.textContent = msg; el.style.display = 'block'; }
}
function hideErr() {
  const el = document.getElementById('upload-error');
  if (el) el.style.display = 'none';
}

// ── DOM refs ───────────────────────────────────────────────────────────────
const dropZone      = document.getElementById('drop-zone');
const csvInput      = document.getElementById('csv-input');
const dropIcon      = document.getElementById('drop-icon');
const dropText      = document.getElementById('drop-text');
const fileInfo      = document.getElementById('file-info');
const fileNameEl    = document.getElementById('file-name');
const fileSizeEl    = document.getElementById('file-size');
const clearFileBtn  = document.getElementById('clear-file');
const progressWrap  = document.getElementById('progress-wrapper');
const progressBar   = document.getElementById('progress-bar');
const progressPct   = document.getElementById('progress-pct');
const predictBtn    = document.getElementById('predict-btn');
const predictBtnTxt = document.getElementById('predict-btn-text');
const downloadBtn   = document.getElementById('download-btn');
const resultsSection= document.getElementById('results-section');
const resultsTbody  = document.getElementById('results-tbody');
const filterSelect  = document.getElementById('filter-select');
const goDashboard   = document.getElementById('go-dashboard');
const singleForm    = document.getElementById('single-form');
const singleResult  = document.getElementById('single-result');
const singleError   = document.getElementById('single-error');
const vGrid         = document.getElementById('v-features-grid');
const vExtraGrid    = document.getElementById('v-extra-grid');
const showMoreBtn   = document.getElementById('show-more-btn');

let selectedFile = null;
let batchResults = null;

// ── Build V-input grids ────────────────────────────────────────────────────
if (vGrid) {
  for (let i = 1; i <= 10; i++) {
    const inp = document.createElement('input');
    inp.id = `f-v${i}`; inp.type = 'number'; inp.step = 'any';
    inp.className = 'form-input'; inp.style.fontSize = '0.78rem'; inp.style.padding = '7px 8px';
    inp.placeholder = `V${i}`;
    inp.setAttribute('aria-label', `PCA Feature V${i}`);
    vGrid.appendChild(inp);
  }
}
if (vExtraGrid) {
  for (let i = 11; i <= 28; i++) {
    const inp = document.createElement('input');
    inp.id = `f-v${i}`; inp.type = 'number'; inp.step = 'any';
    inp.className = 'form-input'; inp.style.fontSize = '0.78rem'; inp.style.padding = '7px 8px';
    inp.placeholder = `V${i}`;
    inp.setAttribute('aria-label', `PCA Feature V${i}`);
    vExtraGrid.appendChild(inp);
  }
}
if (showMoreBtn) {
  showMoreBtn.addEventListener('click', () => {
    const hidden = vExtraGrid.style.display === 'none' || !vExtraGrid.style.display;
    vExtraGrid.style.display = hidden ? 'grid' : 'none';
    showMoreBtn.textContent = hidden ? '- Hide V11-V28' : '+ Show V11-V28';
  });
  vExtraGrid.style.display = 'none';
}

// ── Enable/disable predict button ──────────────────────────────────────────
function setBtn(enabled) {
  if (!predictBtn) return;
  predictBtn.disabled = !enabled;
  predictBtn.style.opacity = enabled ? '1' : '0.45';
  predictBtn.style.cursor  = enabled ? 'pointer' : 'not-allowed';
}
setBtn(false);

// ── Drag & Drop ────────────────────────────────────────────────────────────
if (dropZone) {
  ['dragenter','dragover'].forEach(e =>
    dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('drag-over'); })
  );
  ['dragleave','drop'].forEach(e =>
    dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove('drag-over'); })
  );
  dropZone.addEventListener('drop', ev => {
    const f = ev.dataTransfer.files[0];
    if (f) handleFile(f);
  });
  // Keyboard accessibility — Enter/Space to trigger file input
  dropZone.addEventListener('keydown', ev => {
    if (ev.key === 'Enter' || ev.key === ' ') {
      ev.preventDefault();
      csvInput?.click();
    }
  });
}
if (csvInput) csvInput.addEventListener('change', () => { if (csvInput.files[0]) handleFile(csvInput.files[0]); });

// expose globally for sample buttons in HTML
window.handleFile = function handleFile(file) {
  if (!file.name.toLowerCase().endsWith('.csv')) { showErr('Please upload a .csv file.'); return; }
  selectedFile = file;
  if (dropIcon) dropIcon.textContent = '✓';
  if (dropText) dropText.textContent = 'File ready — click "Analyze CSV" to run detection';
  if (dropZone) { dropZone.classList.add('file-loaded'); dropZone.classList.remove('drag-over'); }
  if (fileNameEl) fileNameEl.textContent = file.name;
  if (fileSizeEl) fileSizeEl.textContent = (file.size / 1024).toFixed(1) + ' KB';
  if (fileInfo) fileInfo.style.display = 'flex';
  setBtn(true);
  hideErr();
};

if (clearFileBtn) {
  clearFileBtn.addEventListener('click', () => {
    selectedFile = null;
    if (csvInput) csvInput.value = '';
    if (dropIcon) dropIcon.textContent = '☁️';
    if (dropText) dropText.textContent = 'Drag & drop your CSV here';
    if (dropZone) { dropZone.classList.remove('file-loaded','drag-over'); }
    if (fileInfo) fileInfo.style.display = 'none';
    if (resultsSection) resultsSection.classList.remove('visible');
    if (downloadBtn) downloadBtn.style.display = 'none';
    if (progressWrap) progressWrap.style.display = 'none';
    setBtn(false);
  });
}

// ── Batch Predict ──────────────────────────────────────────────────────────
if (predictBtn) {
  predictBtn.addEventListener('click', async () => {
    if (!selectedFile) { showErr('Please select a CSV file first.'); return; }
    hideErr();

    // Loading state
    setBtn(false);
    if (predictBtnTxt) predictBtnTxt.textContent = 'Analyzing...';
    if (progressWrap) progressWrap.style.display = 'block';
    if (resultsSection) resultsSection.classList.remove('visible');

    let pct = 0;
    const timer = setInterval(() => {
      pct = Math.min(pct + Math.random() * 14, 88);
      if (progressBar) progressBar.style.width = pct + '%';
      if (progressPct) progressPct.textContent = Math.round(pct) + '%';
    }, 280);

    try {
      const form = new FormData();
      form.append('file', selectedFile);

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 120000); // 2 min timeout

      const res = await fetch(`${API_BASE}/batch_predict`, {
        method: 'POST',
        body: form,
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (!res.ok) {
        const j = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(j.detail || `Server error ${res.status}`);
      }
      const data = await res.json();
      batchResults = data;
      try { localStorage.setItem('fs-batch', JSON.stringify(data)); } catch(_) {}

      // Complete progress
      clearInterval(timer);
      if (progressBar) progressBar.style.width = '100%';
      if (progressPct) progressPct.textContent = '100%';

      // Summary
      const sTotal = document.getElementById('s-total');
      const sFraud = document.getElementById('s-fraud');
      const sLegit = document.getElementById('s-legit');
      const sRate  = document.getElementById('s-rate');
      if (sTotal) sTotal.textContent = data.total_rows.toLocaleString();
      if (sFraud) sFraud.textContent = data.fraud_count.toLocaleString();
      if (sLegit) sLegit.textContent = data.legit_count.toLocaleString();
      if (sRate)  sRate.textContent  = data.fraud_rate.toFixed(2) + '%';

      renderTable(data.predictions);
      if (resultsSection) resultsSection.classList.add('visible');
      if (downloadBtn) downloadBtn.style.display = 'flex';
      if (goDashboard) goDashboard.style.display = 'inline-flex';

      setTimeout(() => { if (resultsSection) resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 300);

    } catch (err) {
      clearInterval(timer);
      if (err.name === 'AbortError') {
        showErr('Request timed out. Try a smaller CSV or check if the backend is running.');
      } else {
        showErr(`${err.message} — Make sure backend is running at ${API_BASE || window.location.origin}`);
      }
    } finally {
      setBtn(true);
      if (predictBtnTxt) predictBtnTxt.textContent = 'Analyze CSV';
      setTimeout(() => { if (progressWrap) progressWrap.style.display = 'none'; }, 1600);
    }
  });
}

// ── Render table (virtualized for large datasets) ─────────────────────────
function renderTable(predictions, filter = 'all') {
  if (!resultsTbody) return;
  let rows = predictions;
  if (filter === 'fraud') rows = rows.filter(r => r.label === 'FRAUD');
  if (filter === 'legit') rows = rows.filter(r => r.label === 'LEGIT');
  if (!rows.length) {
    resultsTbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:32px;color:#64748b">No rows match this filter.</td></tr>';
    return;
  }

  // For large datasets, limit initial render to 500 rows for performance
  const MAX_RENDER = 500;
  const visible = rows.slice(0, MAX_RENDER);
  const hasMore = rows.length > MAX_RENDER;

  const barColor = r => r.label === 'FRAUD' ? '#ef4444' : '#10b981';
  let html = visible.map(r => `
    <tr class="${r.label === 'FRAUD' ? 'fraud-row' : ''}">
      <td style="font-weight:600;color:#94a3b8">#${r.row}</td>
      <td style="font-weight:600">$${Number(r.amount).toFixed(2)}</td>
      <td>
        <div style="display:flex;align-items:center;gap:8px">
          <div style="width:60px;height:4px;background:rgba(255,255,255,0.08);border-radius:99px;overflow:hidden">
            <div style="width:${Math.min(r.risk_score,100)}%;height:100%;background:${barColor(r)};border-radius:99px"></div>
          </div>
          <span style="font-size:0.8rem;font-family:monospace;font-weight:700;color:${barColor(r)}">${r.risk_score}%</span>
        </div>
      </td>
      <td style="font-family:monospace;font-size:0.8rem;color:#64748b">${Number(r.fraud_probability).toFixed(4)}</td>
      <td>${riskBadge(r.label, r.confidence)}</td>
      <td style="font-size:0.78rem;color:#64748b">${r.confidence}</td>
    </tr>`).join('');

  if (hasMore) {
    html += `<tr><td colspan="6" style="text-align:center;padding:16px;color:#14B8A6;font-size:0.85rem;font-weight:600">
      Showing ${MAX_RENDER.toLocaleString()} of ${rows.length.toLocaleString()} rows. Download CSV for full results.
    </td></tr>`;
  }

  resultsTbody.innerHTML = html;
}

if (filterSelect) filterSelect.addEventListener('change', () => {
  if (batchResults) renderTable(batchResults.predictions, filterSelect.value);
});

// ── Download ───────────────────────────────────────────────────────────────
if (downloadBtn) {
  downloadBtn.addEventListener('click', () => {
    if (!batchResults) return;
    const csv = 'row,amount,risk_score,fraud_probability,label,confidence\n'
      + batchResults.predictions.map(r =>
          `${r.row},${r.amount},${r.risk_score},${r.fraud_probability},${r.label},${r.confidence}`
        ).join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = 'fraud_predictions.csv'; a.click();
    URL.revokeObjectURL(a.href);
  });
}

// ── Single Predict ─────────────────────────────────────────────────────────
if (singleForm) {
  singleForm.addEventListener('submit', async e => {
    e.preventDefault();
    if (singleResult) singleResult.style.display = 'none';
    if (singleError)  singleError.style.display  = 'none';

    const amountVal = document.getElementById('f-amount')?.value;
    if (!amountVal || parseFloat(amountVal) < 0) {
      if (singleError) {
        singleError.textContent = 'Please enter a valid Amount (≥ 0).';
        singleError.style.display = 'block';
      }
      return;
    }

    const body = {
      Amount: parseFloat(amountVal) || 0,
      Time:   parseFloat(document.getElementById('f-time')?.value)   || 0,
    };
    for (let i = 1; i <= 28; i++) {
      body[`V${i}`] = parseFloat(document.getElementById(`f-v${i}`)?.value) || 0;
    }

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(j.detail || 'Prediction failed');
      }
      const data    = await res.json();

      // ── Update LocalStorage / Dashboard Pool ──────────────────────────────
      try {
        let batch = localStorage.getItem('fs-batch');
        let batchData;
        if (batch) {
          batchData = JSON.parse(batch);
        } else {
          batchData = {
            total_rows: 0,
            fraud_count: 0,
            legit_count: 0,
            fraud_rate: 0.0,
            predictions: []
          };
        }

        // Add index to match batch row format perfectly and avoid dashboard JS crashes
        const newPred = {
          row_index: batchData.predictions.length,
          row: `Single #${batchData.predictions.length + 1}`,
          time: body.Time || 0.0,
          amount: body.Amount || 0.0,
          fraud_probability: data.fraud_probability,
          risk_score: data.risk_score,
          label: data.label,
          confidence: data.confidence
        };

        batchData.predictions.push(newPred);
        batchData.total_rows = batchData.predictions.length;
        batchData.fraud_count = batchData.predictions.filter(p => p.label === 'FRAUD').length;
        batchData.legit_count = batchData.total_rows - batchData.fraud_count;
        batchData.fraud_rate = parseFloat(((batchData.fraud_count / batchData.total_rows) * 100).toFixed(4));

        localStorage.setItem('fs-batch', JSON.stringify(batchData));
      } catch(e) {
        console.error('Failed to update local storage with single transaction prediction:', e);
      }

      const isFraud = data.label === 'FRAUD';
      const isHigh  = data.confidence === 'HIGH';
      const color   = isFraud ? '#f87171' : '#34d399';
      const bgColor = isFraud ? 'rgba(239,68,68,0.08)' : 'rgba(16,185,129,0.08)';
      const border  = isFraud ? 'rgba(239,68,68,0.2)' : 'rgba(16,185,129,0.2)';

      if (singleResult) {
        singleResult.style.cssText = `display:block;border-radius:16px;padding:24px;text-align:center;background:${bgColor};border:1px solid ${border};position:relative;overflow:hidden`;
        document.getElementById('result-emoji').textContent = isFraud ? (isHigh ? '🚨' : '⚠️') : '✅';
        const lbl = document.getElementById('result-label');
        if (lbl) { lbl.textContent = data.label; lbl.style.color = color; }
        const conf = document.getElementById('result-confidence');
        if (conf) conf.textContent = `Confidence: ${data.confidence} · Score: ${data.risk_score}%`;
        const score = document.getElementById('result-score');
        if (score) { score.textContent = data.risk_score + '%'; score.style.color = color; }
        const prob = document.getElementById('result-prob');
        if (prob) prob.textContent = `Raw probability: ${data.fraud_probability.toFixed(4)}`;
        const bar = document.getElementById('result-bar');
        if (bar) {
          bar.style.background = isFraud
            ? 'linear-gradient(90deg,#f87171,#ef4444)'
            : 'linear-gradient(90deg,#34d399,#10b981)';
          bar.style.width = '0%';
          setTimeout(() => { bar.style.width = data.risk_score + '%'; }, 60);
        }

        // Scan line animation
        const scanLine = document.createElement('div');
        scanLine.className = 'scan-line';
        singleResult.appendChild(scanLine);
        setTimeout(() => scanLine.remove(), 2200);
      }
    } catch (err) {
      if (singleError) {
        singleError.textContent = err.message + ' — Backend at ' + (API_BASE || window.location.origin);
        singleError.style.display = 'block';
      }
    }
  });
}
