/**
 * dashboard.js — Chart.js visualizations + top-fraud table
 * Reads batch results from localStorage (saved by upload.js).
 */

const emptyState   = document.getElementById('empty-state');
const dashMain     = document.getElementById('dashboard-main');
const refreshBtn   = document.getElementById('refresh-btn');
const dashDownload = document.getElementById('dash-download');

let dashData = null;

function loadData() {
  const raw = localStorage.getItem('fs-batch');
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return null; }
}

function init() {
  dashData = loadData();
  if (!dashData || !dashData.predictions || dashData.predictions.length === 0) {
    if (emptyState) emptyState.classList.remove('hidden');
    if (dashMain)   dashMain.classList.add('hidden');
    return;
  }
  if (emptyState) emptyState.classList.add('hidden');
  if (dashMain)   dashMain.classList.remove('hidden');

  populateKPIs();
  drawPieChart();
  drawBarChart();
  drawLineChart();
  populateTopFraud();
}

// ── KPIs ──────────────────────────────────────────────────────────────
function populateKPIs() {
  const el = (id) => document.getElementById(id);
  const kpiTotal = el('kpi-total');
  const kpiFraud = el('kpi-fraud');
  const kpiLegit = el('kpi-legit');
  const kpiRate  = el('kpi-rate');
  const subtitle = el('dash-subtitle');

  if (kpiTotal) kpiTotal.textContent = fmtNum(dashData.total_rows);
  if (kpiFraud) kpiFraud.textContent = fmtNum(dashData.fraud_count);
  if (kpiLegit) kpiLegit.textContent = fmtNum(dashData.legit_count);
  if (kpiRate)  kpiRate.textContent  = dashData.fraud_rate.toFixed(2) + '%';
  if (subtitle) subtitle.textContent =
    `Showing ${fmtNum(dashData.total_rows)} transactions · ${dashData.fraud_count} flagged`;
}

// ── Chart colors ──────────────────────────────────────────────────────
const COLORS = {
  fraud:   '#ef4444',
  legit:   '#10b981',
  teal:    '#14B8A6',
  sky:     '#0EA5E9',
  grid:    'rgba(255,255,255,0.06)',
  textMut: '#94a3b8',
};

function isDark() {
  return document.documentElement.getAttribute('data-theme') === 'dark';
}

// ── Pie chart ─────────────────────────────────────────────────────────
let pieChart = null;
function drawPieChart() {
  const canvas = document.getElementById('pie-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (pieChart) pieChart.destroy();
  pieChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Fraud', 'Legitimate'],
      datasets: [{
        data: [dashData.fraud_count, dashData.legit_count],
        backgroundColor: [COLORS.fraud, COLORS.legit],
        borderWidth: 0,
        spacing: 4,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      cutout: '65%',
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1e293b',
          titleColor: '#f1f5f9',
          bodyColor: '#94a3b8',
          padding: 12,
          cornerRadius: 10,
        },
      },
    },
  });
}

// ── Bar chart (risk distribution) ─────────────────────────────────────
let barChart = null;
function drawBarChart() {
  const canvas = document.getElementById('bar-chart');
  if (!canvas) return;
  const buckets = [
    { label: '0–10%',   min: 0,  max: 10 },
    { label: '10–20%',  min: 10, max: 20 },
    { label: '20–30%',  min: 20, max: 30 },
    { label: '30–50%',  min: 30, max: 50 },
    { label: '50–70%',  min: 50, max: 70 },
    { label: '70–90%',  min: 70, max: 90 },
    { label: '90–100%', min: 90, max: 101 },
  ];
  const counts = buckets.map(b =>
    dashData.predictions.filter(p => p.risk_score >= b.min && p.risk_score < b.max).length
  );

  const ctx = canvas.getContext('2d');
  if (barChart) barChart.destroy();
  barChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: buckets.map(b => b.label),
      datasets: [{
        label: 'Transactions',
        data: counts,
        backgroundColor: counts.map((_, i) => {
          const ratio = i / (buckets.length - 1);
          return `rgba(${Math.round(20 + 219 * ratio)}, ${Math.round(184 - 116 * ratio)}, ${Math.round(166 - 98 * ratio)}, 0.7)`;
        }),
        borderRadius: 6,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#1e293b', titleColor: '#f1f5f9', bodyColor: '#94a3b8', padding: 12, cornerRadius: 10 },
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: COLORS.textMut, font: { size: 10 } } },
        y: { grid: { color: COLORS.grid }, ticks: { color: COLORS.textMut, font: { size: 10 } } },
      },
    },
  });
}

// ── Line chart (risk score over transactions) ─────────────────────────
let lineChart = null;
function drawLineChart() {
  const canvas = document.getElementById('line-chart');
  if (!canvas) return;
  // Sample up to 200 points for performance
  const preds = dashData.predictions;
  const step  = Math.max(1, Math.floor(preds.length / 200));
  const sampled = preds.filter((_, i) => i % step === 0);

  const ctx = canvas.getContext('2d');
  if (lineChart) lineChart.destroy();

  const gradient = ctx.createLinearGradient(0, 0, 0, 250);
  gradient.addColorStop(0, 'rgba(14,165,233,0.3)');
  gradient.addColorStop(1, 'rgba(14,165,233,0.01)');

  lineChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: sampled.map(p => p.row),
      datasets: [{
        label: 'Risk Score %',
        data: sampled.map(p => p.risk_score),
        borderColor: COLORS.sky,
        backgroundColor: gradient,
        fill: true,
        tension: 0.35,
        pointRadius: 0,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#1e293b', titleColor: '#f1f5f9', bodyColor: '#94a3b8', padding: 12, cornerRadius: 10 },
      },
      scales: {
        x: { display: false },
        y: {
          min: 0, max: 100,
          grid: { color: COLORS.grid },
          ticks: { color: COLORS.textMut, callback: v => v + '%', font: { size: 10 } },
        },
      },
    },
  });
}

// ── Top fraud table ───────────────────────────────────────────────────
function populateTopFraud() {
  const sorted = [...dashData.predictions].sort((a, b) => b.risk_score - a.risk_score);
  const top20  = sorted.slice(0, 20);

  const tbody = document.getElementById('top-fraud-tbody');
  if (!tbody) return;

  tbody.innerHTML = top20.map((r, i) => {
    const isFraud = r.label === 'FRAUD';
    return `<tr class="${isFraud ? 'fraud-row' : ''}">
      <td class="font-bold text-xs">#${i + 1}</td>
      <td>${r.row}</td>
      <td>$${r.amount.toFixed(2)}</td>
      <td><div class="flex items-center gap-2">
        <div class="w-16 h-1.5 rounded bg-white/10 overflow-hidden">
          <div class="h-full rounded" style="width:${r.risk_score}%;background:${isFraud ? '#ef4444' : '#fb923c'}"></div>
        </div>
        <span class="text-xs font-mono">${r.risk_score}%</span>
      </div></td>
      <td class="font-mono text-xs">${r.fraud_probability.toFixed(4)}</td>
      <td>${riskBadgeHtml(r.label, r.confidence)}</td>
    </tr>`;
  }).join('');
}

// ── Download all as CSV ───────────────────────────────────────────────
if (dashDownload) {
  dashDownload.addEventListener('click', () => {
    if (!dashData) return;
    const header = 'row,amount,risk_score,fraud_probability,label,confidence\n';
    const csv = header + dashData.predictions.map(r =>
      `${r.row},${r.amount},${r.risk_score},${r.fraud_probability},${r.label},${r.confidence}`
    ).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'fraud_predictions_full.csv'; a.click();
    URL.revokeObjectURL(url);
  });
}

// ── Refresh ───────────────────────────────────────────────────────────
if (refreshBtn) refreshBtn.addEventListener('click', init);

// ── Boot ──────────────────────────────────────────────────────────────
init();
