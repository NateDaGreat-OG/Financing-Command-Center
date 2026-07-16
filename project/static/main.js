const styleSelect = document.getElementById("styleSelect");
const strategySelect = document.getElementById("strategySelect");
const symbolsInput = document.getElementById("symbolsInput");
const backtestBtn = document.getElementById("backtestBtn");
const optimizeBtn = document.getElementById("optimizeBtn");
const liveBtn = document.getElementById("liveBtn");
const trainRlBtn = document.getElementById("trainRlBtn");
const backtestResults = document.getElementById("backtestResults");
const optimizeResults = document.getElementById("optimizeResults");
const rlTrainingResults = document.getElementById("rlTrainingResults");
const rlRunResults = document.getElementById("rlRunResults");
const signalsResults = document.getElementById("signalsResults");
const tradeLogResults = document.getElementById("tradeLogResults");

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

async function trainRLAgent() {
  const symbols = symbolsInput.value.split(",").map(s => s.trim()).filter(Boolean);
  if (!symbols.length) {
    alert("Please provide symbols.");
    return;
  }

  const response = await fetch("/api/rl/train", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({agent: "dqn", symbols, episodes: 50, timeframe: "1D"}),
  });

  const data = await response.json();
  if (!response.ok) {
    rlTrainingResults.innerHTML = `<div class="text-danger">${data.error || "RL training failed"}</div>`;
    return;
  }

  rlTrainingResults.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
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
trainRlBtn.addEventListener("click", trainRLAgent);
liveBtn.addEventListener("click", runLive);

fetchStyles();
