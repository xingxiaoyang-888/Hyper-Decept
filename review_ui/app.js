const state = {
  sessionId: sessionStorage.getItem("hypertraceSession") || null,
  currentCase: null,
  started: false,
};

const byId = (id) => document.getElementById(id);
const views = ["consentView", "briefingView", "trialView", "questionnaireView", "completeView"];

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[character]));
}

function percent(value, digits = 1) {
  return `${(Number(value || 0) * 100).toFixed(digits)}%`;
}

function dateLabel(value) {
  if (!value) return "Not available";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function setView(id) {
  views.forEach((viewId) => byId(viewId).classList.toggle("is-active", viewId === id));
  const phase = {
    consentView: "Study entry",
    briefingView: "Task briefing",
    trialView: "Account review",
    questionnaireView: "Post-task questionnaire",
    completeView: "Study complete",
  }[id];
  byId("phaseLabel").textContent = phase;
  byId("progressWrap").hidden = id !== "trialView";
  window.scrollTo({ top: 0, behavior: "instant" });
  refreshIcons();
}

function setLoading(active, text = "Loading study data") {
  byId("loading").hidden = !active;
  const label = byId("loading").querySelector("p");
  label.textContent = text;
}

let toastTimer;
function toast(message, isError = false) {
  clearTimeout(toastTimer);
  const element = byId("toast");
  element.textContent = message;
  element.classList.toggle("is-error", isError);
  element.classList.add("is-visible");
  toastTimer = setTimeout(() => element.classList.remove("is-visible"), 3200);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  let payload = {};
  try {
    payload = await response.json();
  } catch (_) {
    payload = {};
  }
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

async function createSession(event) {
  event.preventDefault();
  setLoading(true, "Creating private session");
  try {
    const payload = await api("/api/session", {
      method: "POST",
      body: JSON.stringify({
        participant_code: byId("participantCode").value,
        consent: byId("consentConfirmed").checked,
        age_confirmed: byId("ageConfirmed").checked,
      }),
    });
    state.sessionId = payload.session_id;
    sessionStorage.setItem("hypertraceSession", state.sessionId);
    setView("briefingView");
    if (payload.resumed) toast("Existing incomplete session resumed");
  } catch (error) {
    toast(error.message, true);
  } finally {
    setLoading(false);
  }
}

function renderRecommendation(caseData) {
  const coordinated = caseData.model_recommendation === "coordinated";
  byId("recommendationTitle").textContent = coordinated
    ? "Escalate as coordinated"
    : "Do not escalate";
  byId("recommendationTitle").className = coordinated ? "recommend-positive" : "recommend-negative";
  byId("riskPercentile").textContent = percent(caseData.risk_percentile, 1);
  byId("riskStratum").textContent = `${caseData.risk_stratum} priority`;
}

function renderSignals(caseData) {
  const visible = ["standard_signals", "hypertrace_evidence"].includes(caseData.view_mode);
  byId("signalsSection").hidden = !visible;
  if (!visible) return;

  const relations = caseData.relation_summary || {};
  const maximum = Math.max(1, ...Object.values(relations).map(Number));
  byId("relationBars").innerHTML = Object.entries(relations).length
    ? Object.entries(relations).map(([name, count]) => `
      <div class="relation-row">
        <span>${escapeHtml(name.replaceAll("_", " "))}</span>
        <div><i style="width:${Math.max(6, Number(count) / maximum * 100)}%"></i></div>
        <strong>${escapeHtml(count)}</strong>
      </div>`).join("")
    : '<p class="empty-state">No relation summary available.</p>';

  const temporal = caseData.temporal_summary || {};
  byId("firstEvent").textContent = dateLabel(temporal.first_event_at);
  byId("lastEvent").textContent = dateLabel(temporal.last_event_at);
  const events = temporal.event_type_counts || {};
  byId("eventSummary").innerHTML = Object.entries(events).length
    ? Object.entries(events).map(([name, count]) => `<span><strong>${escapeHtml(count)}</strong>${escapeHtml(name)}</span>`).join("")
    : '<span class="empty-state">No event summary available.</span>';
}

function renderTrace(caseData) {
  const visible = caseData.view_mode === "hypertrace_evidence";
  byId("traceSection").hidden = !visible;
  if (!visible) return;

  const explanation = caseData.explanation || {};
  const audit = caseData.audit || {};
  byId("sparsityValue").textContent = percent(explanation.sparsity, 1);
  byId("evidenceUnits").textContent = `${explanation.selected_units || 0} of ${explanation.candidate_units || 0} units`;
  byId("sufficiencyValue").textContent = Number(explanation.sufficiency_error || 0).toFixed(4);
  byId("geometryValue").textContent = Number(explanation.geometry_fidelity || 0).toFixed(3);
  byId("agreementValue").textContent = percent(explanation.checkpoint_agreement, 1);
  byId("fullRisk").textContent = percent(explanation.full_percentile, 1);
  byId("keptRisk").textContent = percent(explanation.keep_only_percentile, 1);
  byId("keptRiskBar").style.width = `${Math.max(2, Number(explanation.keep_only_percentile || 0) * 100)}%`;
  byId("provenanceValue").textContent = percent(audit.provenance_coverage, 0);
  byId("timestampValue").textContent = percent(audit.timestamp_coverage, 0);
  byId("voteValue").textContent = percent(explanation.prototype_vote_agreement, 0);

  const evidence = caseData.evidence || [];
  byId("evidenceCount").textContent = `${evidence.length} records`;
  byId("evidenceRows").innerHTML = evidence.length
    ? evidence.map((record) => `
      <tr>
        <td><time>${escapeHtml(dateLabel(record.timestamp))}</time></td>
        <td><span class="event-type">${escapeHtml(record.event_type || "event")}</span></td>
        <td>${escapeHtml(record.text || "Source record available")}</td>
        <td><code>${escapeHtml(record.evidence_id || "verified")}</code></td>
      </tr>`).join("")
    : '<tr><td colspan="4" class="empty-cell">No text excerpt is available for this evidence unit.</td></tr>';
}

function resetDecision() {
  document.querySelectorAll('input[name="decision"]').forEach((input) => { input.checked = false; });
  byId("confidence").value = "50";
  byId("confidenceValue").textContent = "50";
  byId("rationale").value = "";
  byId("submitDecision").disabled = true;
}

function renderCase(caseData) {
  state.currentCase = caseData;
  byId("operationLabel").textContent = caseData.operation_label;
  byId("caseTitle").textContent = `Case ${String(caseData.trial_index + 1).padStart(2, "0")}`;
  const completed = caseData.trial_index;
  const total = caseData.trial_count;
  byId("progressText").textContent = `Case ${completed + 1} of ${total}`;
  const progress = completed / total * 100;
  byId("progressBar").style.width = `${progress}%`;
  byId("progressWrap").querySelector("[role=progressbar]").setAttribute("aria-valuenow", String(Math.round(progress)));
  renderRecommendation(caseData);
  renderSignals(caseData);
  renderTrace(caseData);
  resetDecision();
  setView("trialView");
  refreshIcons();
}

async function loadNextTrial() {
  setLoading(true, "Loading assigned case");
  try {
    const payload = await api(`/api/session/${state.sessionId}/trial`);
    if (payload.complete) {
      buildQuestionnaire();
      setView("questionnaireView");
    } else {
      renderCase(payload.case);
    }
  } catch (error) {
    toast(error.message, true);
  } finally {
    setLoading(false);
  }
}

async function submitDecision() {
  const selected = document.querySelector('input[name="decision"]:checked');
  if (!selected || !state.currentCase) return;
  byId("submitDecision").disabled = true;
  setLoading(true, "Saving decision");
  try {
    await api(`/api/session/${state.sessionId}/response`, {
      method: "POST",
      body: JSON.stringify({
        trial_index: state.currentCase.trial_index,
        case_id: state.currentCase.case_id,
        decision: selected.value,
        confidence: Number(byId("confidence").value),
        rationale: byId("rationale").value,
      }),
    });
    await loadNextTrial();
  } catch (error) {
    toast(error.message, true);
    byId("submitDecision").disabled = false;
  } finally {
    setLoading(false);
  }
}

const ratings = [
  ["trust", "I could decide when to rely on the model recommendation."],
  ["clarity", "The information presented for each case was clear."],
  ["workload", "The review task required substantial mental effort."],
  ["evidence_usefulness", "The available case information supported my final decision."],
];

function buildQuestionnaire() {
  byId("ratingFields").innerHTML = ratings.map(([name, label]) => `
    <fieldset class="rating-row">
      <legend>${escapeHtml(label)}</legend>
      <div class="rating-scale">
        ${[1, 2, 3, 4, 5, 6, 7].map((value) => `<label><input type="radio" name="${name}" value="${value}" required><span>${value}</span></label>`).join("")}
      </div>
      <div class="rating-ends"><span>Strongly disagree</span><span>Strongly agree</span></div>
    </fieldset>`).join("");
}

async function submitQuestionnaire(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  setLoading(true, "Completing study");
  try {
    const payload = await api(`/api/session/${state.sessionId}/questionnaire`, {
      method: "POST",
      body: JSON.stringify({
        trust: Number(form.get("trust")),
        clarity: Number(form.get("clarity")),
        workload: Number(form.get("workload")),
        evidence_usefulness: Number(form.get("evidence_usefulness")),
        feedback: byId("feedback").value,
      }),
    });
    byId("completionCode").textContent = payload.completion_code;
    sessionStorage.removeItem("hypertraceSession");
    setView("completeView");
  } catch (error) {
    toast(error.message, true);
  } finally {
    setLoading(false);
  }
}

function bindEvents() {
  byId("consentForm").addEventListener("submit", createSession);
  byId("startTrials").addEventListener("click", loadNextTrial);
  byId("confidence").addEventListener("input", (event) => {
    byId("confidenceValue").textContent = event.target.value;
  });
  document.querySelectorAll('input[name="decision"]').forEach((input) => {
    input.addEventListener("change", () => { byId("submitDecision").disabled = false; });
  });
  byId("submitDecision").addEventListener("click", submitDecision);
  byId("questionnaireForm").addEventListener("submit", submitQuestionnaire);
}

async function initialize() {
  bindEvents();
  refreshIcons();
  try {
    const health = await api("/api/health");
    if (health.demo_data) byId("phaseLabel").textContent = "Demonstration study";
  } catch (error) {
    toast("Study service is unavailable", true);
  }
  if (state.sessionId) setView("briefingView");
}

initialize();
