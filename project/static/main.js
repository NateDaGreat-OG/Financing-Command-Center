const styleSelect = document.getElementById("styleSelect");
const strategySelect = document.getElementById("strategySelect");
const symbolsInput = document.getElementById("symbolsInput");
const rlEpisodesInput = document.getElementById("rlEpisodesInput");
const backtestBtn = document.getElementById("backtestBtn");
const optimizeBtn = document.getElementById("optimizeBtn");
const runAllocationBtn = document.getElementById("runAllocationBtn");
const runCycleAnalysisBtn = document.getElementById("runCycleAnalysisBtn");
const liveBtn = document.getElementById("liveBtn");
const trainRlBtn = document.getElementById("trainRlBtn");
const backtestResults = document.getElementById("backtestResults");
const optimizeResults = document.getElementById("optimizeResults");
const capitalAllocationResults = document.getElementById("capitalAllocationResults");
const cycleAnalysisResults = document.getElementById("cycleAnalysisResults");
const rlTrainingResults = document.getElementById("rlTrainingResults");
const rlRewardChartWrap = document.getElementById("rlRewardChartWrap");
const rlTrainingLog = document.getElementById("rlTrainingLog");
const rlTrainingLogBody = document.getElementById("rlTrainingLogBody");
const rlModelRegistry = document.getElementById("rlModelRegistry");
const rlRunResults = document.getElementById("rlRunResults");
const signalsResults = document.getElementById("signalsResults");
const tradeLogResults = document.getElementById("tradeLogResults");

let rewardChart = null;

/** Escape user/API content before inserting into innerHTML. */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

async function fetchStyles() {
  const response = await fetch("/api/styles");
  const styles = await response.json();
  styleSelect.innerHTML = styles
    .map(style => `<option value="${style}">${style.replace(/_/g, " ")}</option>`)
    .join("");
  if (styles.length) {
    await fetchStrategies(styles[0]);
  }
}

async function fetchStrategies(style) {
  const response = await fetch(`/api/strategies/${style}`);
  if (!response.ok) {
    strategySelect.innerHTML = "<option>Not found</option>";
    return;
  }
  const strategies = await response.json();
  strategySelect.innerHTML = strategies
    .map(name => `<option value="${name}">${name.replace(/_/g, " ")}</option>`)
    .join("");
}

async function runBacktest() {
  const strategy = strategySelect.value;
  const symbols = symbolsInput.value.split(",").map(s => s.trim()).filter(Boolean);
  if (!strategy || !symbols.length) {
    alert("Please select a strategy and provide symbols.");
    return;
  }

  const response = await fetch("/api/backtest", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({strategy, symbols}),
  });

  const data = await response.json();
  if (!response.ok) {
    backtestResults.innerHTML = `<div class="text-danger">${data.error || "Backtest failed"}</div>`;
    return;
  }

  backtestResults.innerHTML = `
    <div class="row">
      <div class="col-6"><strong>Trade Count:</strong> ${data.metrics.trade_count}</div>
      <div class="col-6"><strong>Win Rate:</strong> ${data.metrics.win_rate}%</div>
    </div>
    <div class="row mt-2">
      <div class="col-6"><strong>CAGR:</strong> ${data.metrics.cagr}%</div>
      <div class="col-6"><strong>Max Drawdown:</strong> ${data.metrics.max_drawdown}</div>
    </div>
    <div class="row mt-2">
      <div class="col-12"><strong>Sharpe:</strong> ${data.metrics.sharpe}</div>
    </div>
  `;

  tradeLogResults.innerHTML = `
    <div class="table-responsive">
      <table class="table table-sm table-striped">
        <thead>
          <tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>Net PnL</th></tr>
        </thead>
        <tbody>
          ${data.trade_log.map(trade => `
            <tr>
              <td>${trade.symbol}</td>
              <td>${trade.side}</td>
              <td>${trade.entry_price}</td>
              <td>${trade.exit_price}</td>
              <td>${trade.net_pnl.toFixed(2)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
  signalsResults.innerHTML = "Backtest complete.";
}

async function runOptimization() {
  const strategy = strategySelect.value;
  const symbols = symbolsInput.value.split(",").map(s => s.trim()).filter(Boolean);
  if (!strategy || !symbols.length) {
    alert("Please select a strategy and provide symbols.");
    return;
  }

  const response = await fetch("/api/optimize", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({strategy, symbols, objective: "max_sharpe", method: "grid"}),
  });

  const data = await response.json();
  if (!response.ok) {
    optimizeResults.innerHTML = `<div class="text-danger">${data.error || "Optimization failed"}</div>`;
    return;
  }

  optimizeResults.innerHTML = `
    <div><strong>Best Params:</strong> <pre>${JSON.stringify(data.best_params, null, 2)}</pre></div>
    <div><strong>Best Metrics:</strong> <pre>${JSON.stringify(data.best_metrics, null, 2)}</pre></div>
  `;
}

function renderRewardChart(symbolResults) {
  // Collect all (episode, reward) pairs across symbols, keyed by symbol
  const datasets = [];
  const palette = ["#1f78ff", "#56d364", "#f78166", "#e3b341", "#79c0ff", "#d2a8ff"];
  let idx = 0;
  for (const [symbol, result] of Object.entries(symbolResults)) {
    if (!Array.isArray(result.history) || !result.history.length) continue;
    datasets.push({
      label: symbol,  // Chart.js handles its own escaping for display
      data: result.history
        .filter(h => h && typeof h.episode === "number" && typeof h.reward === "number")
        .map(h => ({x: h.episode, y: parseFloat(h.reward.toFixed(4))})),
      borderColor: palette[idx % palette.length],
      backgroundColor: "transparent",
      tension: 0.3,
      pointRadius: 2,
    });
    idx++;
  }
  if (!datasets.length) return;

  rlRewardChartWrap.style.display = "block";
  const ctx = document.getElementById("rlRewardChart").getContext("2d");
  if (rewardChart) {
    rewardChart.destroy();
  }
  rewardChart = new Chart(ctx, {
    type: "line",
    data: {datasets},
    options: {
      responsive: true,
      parsing: false,
      plugins: {
        legend: {position: "top"},
        title: {display: true, text: "Episode Reward Curve"},
      },
      scales: {
        x: {type: "linear", title: {display: true, text: "Episode"}},
        y: {title: {display: true, text: "Reward"}},
      },
    },
  });
}

function renderTrainingLog(symbolResults) {
  const rows = [];
  for (const [symbol, result] of Object.entries(symbolResults)) {
    if (!Array.isArray(result.history) || !result.history.length) continue;
    for (const h of result.history) {
      if (!h || typeof h.episode !== "number") continue;
      rows.push(`
        <div class="log-row">
          <span class="log-ep">${escapeHtml(symbol)} ep ${h.episode}</span>
          <span class="log-reward">r=${Number(h.reward).toFixed(3)}</span>
          <span class="log-loss">loss=${Number(h.avg_loss).toFixed(4)}</span>
          <span class="log-eps">ε=${Number(h.epsilon).toFixed(3)}</span>
        </div>`);
    }
  }
  if (!rows.length) return;
  rlTrainingLog.style.display = "block";
  rlTrainingLogBody.innerHTML = rows.join("");
  // scroll to bottom
  rlTrainingLogBody.scrollTop = rlTrainingLogBody.scrollHeight;
}

async function trainRLAgent() {
  const symbols = symbolsInput.value.split(",").map(s => s.trim()).filter(Boolean);
  const episodes = parseInt(rlEpisodesInput.value, 10) || 50;
  if (!symbols.length) {
    alert("Please provide symbols.");
    return;
  }

  const safeSymbols = symbols.map(escapeHtml).join(", ");
  rlTrainingResults.innerHTML = `<div class="text-muted">Training DQN agent for ${safeSymbols} over ${episodes} episodes…</div>`;
  rlRewardChartWrap.style.display = "none";
  rlTrainingLog.style.display = "none";

  const response = await fetch("/api/rl/train", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({agent: "dqn", symbols, episodes, timeframe: "1D"}),
  });

  const data = await response.json();
  if (!response.ok) {
    rlTrainingResults.innerHTML = `<div class="text-danger">${data.error || "RL training failed"}</div>`;
    return;
  }

  // Summary
  const summaryRows = Object.entries(data).map(([sym, res]) => {
    const safeSym = escapeHtml(sym);
    if (res.error) return `<tr><td>${safeSym}</td><td colspan="3" class="text-danger">${escapeHtml(res.error)}</td></tr>`;
    return `<tr>
      <td>${safeSym}</td>
      <td>${escapeHtml(String(res.episodes_trained))}</td>
      <td>${res.best_reward !== undefined ? Number(res.best_reward).toFixed(4) : "—"}</td>
      <td><span class="badge bg-success">Saved</span></td>
    </tr>`;
  }).join("");

  rlTrainingResults.innerHTML = `
    <table class="table table-sm table-striped mb-0">
      <thead><tr><th>Symbol</th><th>Episodes</th><th>Best Reward</th><th>Model</th></tr></thead>
      <tbody>${summaryRows}</tbody>
    </table>`;

  renderRewardChart(data);
  renderTrainingLog(data);
  await fetchRLModels();
}

async function fetchRLModels() {
  const response = await fetch("/api/rl/models");
  if (!response.ok) {
    rlModelRegistry.innerHTML = `<div class="text-danger">Could not load model registry</div>`;
    return;
  }
  const models = await response.json();
  const symbols = Object.keys(models);
  if (!symbols.length) {
    rlModelRegistry.innerHTML = `<p class="text-muted">No trained models yet. Train an agent to see models here.</p>`;
    return;
  }

  const rows = symbols.map(sym => {
    const m = models[sym];
    const kb = (m.size_bytes / 1024).toFixed(1);
    const modified = new Date(m.modified * 1000).toLocaleString();
    return `<tr>
      <td>${escapeHtml(sym)}</td>
      <td>${escapeHtml(kb)} KB</td>
      <td>${escapeHtml(modified)}</td>
      <td>
        <button class="btn btn-sm btn-outline-danger rl-model-delete" data-symbol="${escapeHtml(sym)}">Delete</button>
      </td>
    </tr>`;
  }).join("");

  rlModelRegistry.innerHTML = `
    <table class="table table-sm table-striped mb-0">
      <thead><tr><th>Symbol</th><th>Size</th><th>Last Modified</th><th></th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function deleteRLModel(symbol) {
  if (!confirm(`Delete RL model for ${symbol}?`)) return;
  const response = await fetch(`/api/rl/models/${encodeURIComponent(symbol)}`, {method: "DELETE"});
  if (!response.ok) {
    const data = await response.json();
    alert(data.error || "Delete failed");
    return;
  }
  await fetchRLModels();
}

async function runRLAgent() {
  const symbol = symbolsInput.value.split(",").map(s => s.trim()).filter(Boolean)[0];
  if (!symbol) {
    alert("Please provide a symbol.");
    return;
  }

  const response = await fetch("/api/rl/run", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({agent: "dqn", symbol}),
  });

  const data = await response.json();
  if (!response.ok) {
    rlRunResults.innerHTML = `<div class="text-danger">${data.error || "RL run failed"}</div>`;
    return;
  }

  rlRunResults.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
}

async function runCycleAnalysis() {
  const symbols = symbolsInput.value.split(",").map(s => s.trim()).filter(Boolean);
  if (!symbols.length) {
    alert("Please provide symbols for cycle analysis.");
    return;
  }

  cycleAnalysisResults.innerHTML = "Analyzing market cycles...";
  const response = await fetch("/api/cycles/analyze", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({symbols, timeframe: "1D"}),
  });
  const data = await response.json();
  if (!response.ok) {
    cycleAnalysisResults.innerHTML = `<div class="text-danger">${data.error || "Cycle analysis failed"}</div>`;
    return;
  }

  cycleAnalysisResults.innerHTML = `
    <div><strong>Aggregate Regime:</strong> ${JSON.stringify(data.cycle_state, null, 2)}</div>
    <div><strong>Symbol States:</strong> <pre>${JSON.stringify(data.symbol_states, null, 2)}</pre></div>
  `;
}

async function runCapitalAllocation() {
  const strategies = [strategySelect.value].filter(Boolean);
  const symbols = symbolsInput.value.split(",").map(s => s.trim()).filter(Boolean);
  if (!strategies.length || !symbols.length) {
    alert("Please select a strategy and provide symbols for capital allocation.");
    return;
  }

  capitalAllocationResults.innerHTML = "Allocating capital...";
  const response = await fetch("/api/capital/allocate", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({strategies, symbols, timeframe: "1D"}),
  });

  const data = await response.json();
  if (!response.ok) {
    capitalAllocationResults.innerHTML = `<div class="text-danger">${data.error || "Capital allocation failed"}</div>`;
    return;
  }

  capitalAllocationResults.innerHTML = `
    <div><strong>Allocation Map:</strong></div>
    <pre>${JSON.stringify(data.allocation, null, 2)}</pre>
    <div><strong>Cycle:</strong> ${JSON.stringify(data.cycle_state, null, 2)}</div>
  `;
}

async function runLive() {
  const strategy = strategySelect.value;
  const symbols = symbolsInput.value.split(",").map(s => s.trim()).filter(Boolean);
  if (!strategy || !symbols.length) {
    alert("Please select a strategy and provide symbols.");
    return;
  }

  const response = await fetch("/api/live", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({strategy, symbols}),
  });

  const data = await response.json();
  if (!response.ok) {
    signalsResults.innerHTML = `<div class="text-danger">${data.error || "Live execution failed"}</div>`;
    return;
  }

  signalsResults.innerHTML = `<pre>${JSON.stringify(data.results, null, 2)}</pre>`;
  tradeLogResults.innerHTML = `<pre>Account: ${JSON.stringify(data.account, null, 2)}</pre>`;
}

styleSelect.addEventListener("change", event => fetchStrategies(event.target.value));
backtestBtn.addEventListener("click", runBacktest);
optimizeBtn.addEventListener("click", runOptimization);
runAllocationBtn.addEventListener("click", runCapitalAllocation);
runCycleAnalysisBtn.addEventListener("click", runCycleAnalysis);
trainRlBtn.addEventListener("click", trainRLAgent);
liveBtn.addEventListener("click", runLive);

// Event delegation for dynamically rendered model-registry delete buttons.
rlModelRegistry.addEventListener("click", event => {
  const btn = event.target.closest(".rl-model-delete");
  if (btn) {
    deleteRLModel(btn.dataset.symbol);
  }
});

// ============================================================
// Intelligence Layer UI elements
// ============================================================

const intelBacktestBtn = document.getElementById("intelBacktestBtn");
const intelOptimizeBtn = document.getElementById("intelOptimizeBtn");
const intelLiveBtn = document.getElementById("intelLiveBtn");
const intelDiagnosticsBtn = document.getElementById("intelDiagnosticsBtn");

const intelligenceResults = document.getElementById("intelligenceResults");
const cycleResults = document.getElementById("cycleResults");
const rlIntelResults = document.getElementById("rlIntelResults");
const riskResults = document.getElementById("riskResults");
const capitalIntelResults = document.getElementById("capitalIntelResults");
const executionResults = document.getElementById("executionResults");
const governanceResults = document.getElementById("governanceResults");
const diagnosticsResults = document.getElementById("diagnosticsResults");

// ============================================================
// Intelligence Layer functions
// ============================================================

async function runIntelligenceBacktest() {
  const strategy = strategySelect.value;
  const symbols = symbolsInput.value.split(",").map(s => s.trim()).filter(Boolean);
  if (!strategy || !symbols.length) {
    alert("Please select a strategy and provide symbols.");
    return;
  }

  intelligenceResults.innerHTML = `<div class="text-muted">Running intelligence-enriched backtest…</div>`;

  const response = await fetch("/api/intel/backtest", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({strategy, symbols}),
  });

  const data = await response.json();
  if (!response.ok) {
    intelligenceResults.innerHTML = `<div class="text-danger">${escapeHtml(data.error || "Intelligence backtest failed")}</div>`;
    return;
  }

  const m = data.metrics || {};
  intelligenceResults.innerHTML = `
    <div class="row">
      <div class="col-4"><strong>Sharpe (ann.):</strong> ${escapeHtml(String(m.annualized_sharpe ?? m.sharpe ?? "–"))}</div>
      <div class="col-4"><strong>Sortino:</strong> ${escapeHtml(String(m.sortino ?? "–"))}</div>
      <div class="col-4"><strong>Profit Factor:</strong> ${escapeHtml(String(m.profit_factor ?? "–"))}</div>
    </div>
    <div class="row mt-2">
      <div class="col-4"><strong>Expectancy:</strong> ${escapeHtml(String(m.expectancy ?? "–"))}</div>
      <div class="col-4"><strong>Max Drawdown:</strong> ${escapeHtml(String(m.max_drawdown ?? "–"))}</div>
      <div class="col-4"><strong>Win Rate:</strong> ${escapeHtml(String(m.win_rate ?? "–"))}%</div>
    </div>
    <div class="row mt-2">
      <div class="col-4"><strong>CAGR:</strong> ${escapeHtml(String(m.cagr ?? "–"))}%</div>
      <div class="col-4"><strong>Trades:</strong> ${escapeHtml(String(m.trade_count ?? "–"))}</div>
      <div class="col-4"><strong>Final Equity:</strong> $${escapeHtml(String(m.final_equity ?? "–"))}</div>
    </div>`;

  if (data.intelligence_diagnostics) {
    displayDiagnostics(data.intelligence_diagnostics);
  }
}

async function runIntelligenceOptimizer() {
  const strategy = strategySelect.value;
  const symbols = symbolsInput.value.split(",").map(s => s.trim()).filter(Boolean);
  if (!strategy || !symbols.length) {
    alert("Please select a strategy and provide symbols.");
    return;
  }

  intelligenceResults.innerHTML = `<div class="text-muted">Running intelligence optimizer…</div>`;

  const response = await fetch("/api/intel/optimize", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({strategy, symbols, objective: "max_sharpe", method: "grid"}),
  });

  const data = await response.json();
  if (!response.ok) {
    intelligenceResults.innerHTML = `<div class="text-danger">${escapeHtml(data.error || "Intelligence optimizer failed")}</div>`;
    return;
  }

  intelligenceResults.innerHTML = `
    <div><strong>Method:</strong> ${escapeHtml(data.method || "grid")}</div>
    <div class="mt-2"><strong>Best Score:</strong> ${escapeHtml(String(data.best_score ?? "–"))}</div>
    <div class="mt-2"><strong>Best Params:</strong> <pre>${escapeHtml(JSON.stringify(data.best_params, null, 2))}</pre></div>
    <div><strong>Best Metrics:</strong> <pre>${escapeHtml(JSON.stringify(data.best_metrics, null, 2))}</pre></div>`;

  if (data.best_intel_diagnostics) {
    displayDiagnostics(data.best_intel_diagnostics);
  }
}

async function runIntelligenceLive() {
  const strategy = strategySelect.value;
  const symbols = symbolsInput.value.split(",").map(s => s.trim()).filter(Boolean);
  if (!strategy || !symbols.length) {
    alert("Please select a strategy and provide symbols.");
    return;
  }

  intelligenceResults.innerHTML = `<div class="text-muted">Running intelligence live engine (dry run)…</div>`;

  const response = await fetch("/api/intel/live", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({strategy, symbols, dry_run: true}),
  });

  const data = await response.json();
  if (!response.ok) {
    intelligenceResults.innerHTML = `<div class="text-danger">${escapeHtml(data.error || "Intelligence live failed")}</div>`;
    return;
  }

  intelligenceResults.innerHTML = `
    <div><strong>Status:</strong> ${escapeHtml(data.status || "")}</div>
    <div><strong>Total Signals:</strong> ${escapeHtml(String(data.total_signals ?? 0))}</div>
    <div><strong>Orders Submitted:</strong> ${escapeHtml(String((data.orders_submitted || []).length))}</div>`;

  if (data.diagnostics) {
    displayDiagnostics(data.diagnostics);
  }
}

async function fetchAndShowDiagnostics() {
  const strategy = strategySelect.value;
  const symbols = symbolsInput.value.split(",").map(s => s.trim()).filter(Boolean);
  if (!strategy || !symbols.length) {
    alert("Please select a strategy and provide symbols.");
    return;
  }

  diagnosticsResults.innerHTML = `<div class="text-muted">Loading diagnostics…</div>`;

  const response = await fetch("/api/intel/diagnostics", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({strategy, symbols}),
  });

  const data = await response.json();
  if (!response.ok) {
    diagnosticsResults.innerHTML = `<div class="text-danger">${escapeHtml(data.error || "Diagnostics failed")}</div>`;
    return;
  }

  displayDiagnostics(data);
}

// ============================================================
// Display helpers for each intelligence panel
// ============================================================

function displayDiagnostics(diag) {
  if (!diag) return;
  diagnosticsResults.innerHTML = `<pre class="small">${escapeHtml(JSON.stringify(diag, null, 2))}</pre>`;
  if (diag.ai) displayIntelligence(diag.ai);
  if (diag.cycle) displayCycle(diag.cycle);
  if (diag.rl) displayRL(diag.rl);
  if (diag.risk) displayRisk(diag.risk);
  if (diag.capital) displayCapital(diag.capital);
  if (diag.execution) displayExecution(diag.execution);
  if (diag.governance) displayGovernance(diag.governance);
}

function displayIntelligence(ai) {
  if (!ai) return;
  const topSignals = (ai.top_signals || []).map(s => `
    <tr>
      <td>${escapeHtml(s.symbol || "")}</td>
      <td>${escapeHtml(s.side || "")}</td>
      <td>${escapeHtml(String(s.ai_score ?? "–"))}</td>
      <td>${escapeHtml(String(s.entry_price ?? "–"))}</td>
      <td>${escapeHtml(String(s.size ?? "–"))}</td>
    </tr>`).join("");
  intelligenceResults.innerHTML = `
    <div class="row">
      <div class="col-4"><strong>Signals:</strong> ${escapeHtml(String(ai.signal_count ?? 0))}</div>
      <div class="col-4"><strong>Avg AI Score:</strong> ${escapeHtml(String(ai.avg_ai_score ?? "–"))}</div>
      <div class="col-4"><strong>Strategy:</strong> ${escapeHtml(ai.strategy || "–")}</div>
    </div>
    ${topSignals ? `<table class="table table-sm mt-2"><thead><tr><th>Symbol</th><th>Side</th><th>Score</th><th>Entry</th><th>Size</th></tr></thead><tbody>${topSignals}</tbody></table>` : ""}`;
}

function displayCycle(cycle) {
  if (!cycle) return;
  cycleResults.innerHTML = `
    <div class="row">
      <div class="col-4"><strong>Trend:</strong> <span class="badge bg-secondary">${escapeHtml(cycle.trend || "–")}</span></div>
      <div class="col-4"><strong>Volatility:</strong> <span class="badge bg-secondary">${escapeHtml(cycle.volatility || "–")}</span></div>
      <div class="col-4"><strong>Liquidity:</strong> <span class="badge bg-secondary">${escapeHtml(cycle.liquidity || "–")}</span></div>
    </div>
    <div class="row mt-2">
      <div class="col-4"><strong>Macro:</strong> <span class="badge bg-secondary">${escapeHtml(cycle.macro || "–")}</span></div>
      <div class="col-4"><strong>Intraday:</strong> <span class="badge bg-secondary">${escapeHtml(cycle.intraday || "–")}</span></div>
      <div class="col-4"><strong>Sector:</strong> ${escapeHtml(JSON.stringify(cycle.sector_rotation || {}))}</div>
    </div>
    ${Object.keys(cycle.cycle_params || {}).length ? `<div class="mt-2"><strong>Cycle Params:</strong> <pre class="small">${escapeHtml(JSON.stringify(cycle.cycle_params, null, 2))}</pre></div>` : ""}`;
}

function displayRL(rl) {
  if (!rl) return;
  rlIntelResults.innerHTML = `
    <div class="row">
      <div class="col-4"><strong>Action:</strong> ${escapeHtml(String(rl.rl_action ?? "–"))}</div>
      <div class="col-4"><strong>Confidence:</strong> ${escapeHtml(String(rl.rl_confidence ?? "–"))}</div>
      <div class="col-4"><strong>AI Score:</strong> ${escapeHtml(String(rl.rl_ai_score ?? "–"))}</div>
    </div>
    <div class="mt-2"><strong>Recent Actions:</strong> ${escapeHtml(JSON.stringify(rl.recent_actions || []))}</div>`;
}

function displayRisk(risk) {
  if (!risk) return;
  riskResults.innerHTML = `
    <div class="row">
      <div class="col-4"><strong>Leverage:</strong> ${escapeHtml(String(risk.dynamic_leverage ?? "–"))}</div>
      <div class="col-4"><strong>Drawdown:</strong> ${escapeHtml(String(risk.current_drawdown ?? "–"))}</div>
      <div class="col-4"><strong>Breached:</strong> <span class="badge ${risk.drawdown_breached ? "bg-danger" : "bg-success"}">${risk.drawdown_breached ? "YES" : "NO"}</span></div>
    </div>
    <div class="row mt-2">
      <div class="col-4"><strong>Target Vol:</strong> ${escapeHtml(String(risk.target_vol ?? "–"))}</div>
      <div class="col-4"><strong>Max DD:</strong> ${escapeHtml(String(risk.max_portfolio_drawdown ?? "–"))}</div>
      <div class="col-4"><strong>Signals:</strong> ${escapeHtml(String(risk.signal_count ?? "–"))}</div>
    </div>`;
}

function displayCapital(capital) {
  if (!capital) return;
  const summaryRows = Object.entries(capital.allocation_summary || {}).map(([strat, syms]) =>
    `<tr><td>${escapeHtml(strat)}</td><td><pre class="small mb-0">${escapeHtml(JSON.stringify(syms, null, 2))}</pre></td></tr>`
  ).join("");
  capitalIntelResults.innerHTML = `
    <div class="row">
      <div class="col-6"><strong>Total Allocated:</strong> $${escapeHtml(String(capital.total_allocated ?? 0))}</div>
      <div class="col-6"><strong>Strategies:</strong> ${escapeHtml((capital.strategies || []).join(", ") || "–")}</div>
    </div>
    ${summaryRows ? `<table class="table table-sm mt-2"><thead><tr><th>Strategy</th><th>Allocation</th></tr></thead><tbody>${summaryRows}</tbody></table>` : ""}`;
}

function displayExecution(exec) {
  if (!exec) return;
  const win = exec.execution_window || {};
  executionResults.innerHTML = `
    <div class="row">
      <div class="col-4"><strong>Signals:</strong> ${escapeHtml(String(exec.signal_count ?? 0))}</div>
      <div class="col-4"><strong>Est. Slippage:</strong> $${escapeHtml(String(exec.total_estimated_slippage ?? 0))}</div>
      <div class="col-4"><strong>Slippage bps:</strong> ${escapeHtml(String(exec.slippage_bps ?? "–"))}</div>
    </div>
    <div class="row mt-2">
      <div class="col-4"><strong>Execute Now:</strong> <span class="badge ${win.execute_now ? "bg-success" : "bg-secondary"}">${win.execute_now ? "YES" : "NO"}</span></div>
      <div class="col-4"><strong>Urgency:</strong> ${escapeHtml(win.urgency || "–")}</div>
      <div class="col-4"><strong>Session:</strong> ${escapeHtml(win.intraday_session || "–")}</div>
    </div>
    <div class="mt-2 small text-muted">${escapeHtml(win.reason || "")}</div>`;
}

function displayGovernance(gov) {
  if (!gov) return;
  const disabled = (gov.disabled_strategies || []).map(s => `<span class="badge bg-danger me-1">${escapeHtml(s)}</span>`).join("");
  const reduced = (gov.capital_reduced_strategies || []).map(s => `<span class="badge bg-warning text-dark me-1">${escapeHtml(s)}</span>`).join("");
  governanceResults.innerHTML = `
    <div class="row">
      <div class="col-4"><strong>Monitored:</strong> ${escapeHtml(String(gov.strategy_count ?? 0))}</div>
      <div class="col-8"><strong>Disabled:</strong> ${disabled || '<span class="text-success">None</span>'}</div>
    </div>
    <div class="row mt-2">
      <div class="col-4"><strong>Capital Reduced:</strong></div>
      <div class="col-8">${reduced || '<span class="text-success">None</span>'}</div>
    </div>`;
}

// ============================================================
// Wire intelligence layer buttons
// ============================================================

intelBacktestBtn.addEventListener("click", runIntelligenceBacktest);
intelOptimizeBtn.addEventListener("click", runIntelligenceOptimizer);
intelLiveBtn.addEventListener("click", runIntelligenceLive);
intelDiagnosticsBtn.addEventListener("click", fetchAndShowDiagnostics);

fetchStyles();
fetchRLModels();
