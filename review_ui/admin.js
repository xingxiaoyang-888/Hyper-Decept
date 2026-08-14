let adminToken = "";
const byId = (id) => document.getElementById(id);

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
}

let toastTimer;
function toast(message, isError = false) {
  clearTimeout(toastTimer);
  const element = byId("toast");
  element.textContent = message;
  element.classList.toggle("is-error", isError);
  element.classList.add("is-visible");
  toastTimer = setTimeout(() => element.classList.remove("is-visible"), 3000);
}

async function authorizedFetch(path) {
  const response = await fetch(path, { headers: { Authorization: `Bearer ${adminToken}` } });
  if (!response.ok) {
    let detail = "Admin request failed";
    try { detail = (await response.json()).detail || detail; } catch (_) { /* no body */ }
    throw new Error(detail);
  }
  return response;
}

function formatRate(value) {
  return value == null ? "No data" : `${(Number(value) * 100).toFixed(1)}%`;
}

async function loadSummary() {
  try {
    const response = await authorizedFetch("/api/admin/summary");
    const payload = await response.json();
    byId("sessionTotal").textContent = payload.sessions;
    byId("completedTotal").textContent = payload.completed_sessions;
    byId("completionRate").textContent = payload.sessions
      ? `${(payload.completed_sessions / payload.sessions * 100).toFixed(1)}%`
      : "0%";
    const labels = {
      risk_only: "Risk only",
      standard_signals: "Standard signals",
      hypertrace_evidence: "HyperTrace evidence",
    };
    byId("conditionRows").innerHTML = Object.entries(payload.conditions).map(([condition, values]) => `
      <tr><td>${labels[condition] || condition}</td><td>${values.responses}</td><td>${formatRate(values.accuracy)}</td><td>${formatRate(values.appropriate_reliance)}</td></tr>`).join("");
    byId("adminLogin").hidden = true;
    byId("adminDashboard").hidden = false;
    refreshIcons();
  } catch (error) {
    toast(error.message, true);
  }
}

async function exportResponses() {
  try {
    const response = await authorizedFetch("/api/admin/export");
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `hypertrace_responses_${new Date().toISOString().slice(0, 10)}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    toast(error.message, true);
  }
}

byId("adminLoginForm").addEventListener("submit", (event) => {
  event.preventDefault();
  adminToken = byId("adminToken").value;
  loadSummary();
});
byId("refreshAdmin").addEventListener("click", loadSummary);
byId("exportAdmin").addEventListener("click", exportResponses);
refreshIcons();
