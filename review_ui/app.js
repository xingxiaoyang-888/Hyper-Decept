const state = {
  sessionId: sessionStorage.getItem("hypertraceSession") || null,
  currentCase: null,
  started: false,
  preview: false,
  previewIndex: 0,
  previewCount: 0,
  currentPhase: null,
  demoMode: false,
  updateTimer: null,
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
  return date.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
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

function renderCommonContext(caseData) {
  const context = caseData.case_context || {};
  byId("contextScope").textContent = caseData.snapshot_scope === "full_release_snapshot"
    ? "Full release snapshot"
    : String(caseData.snapshot_scope || "Declared data range").replaceAll("_", " ");
  byId("contextRecords").textContent = context.record_count == null
    ? "Not available"
    : `${context.record_count} available; ${context.displayed_record_count || 0} shown`;
  const start = dateLabel(context.observation_start);
  const end = dateLabel(context.observation_end);
  byId("contextWindow").textContent = start === "Not available" && end === "Not available"
    ? "Not available"
    : `${start} to ${end}`;
  const eventCounts = context.event_type_counts || {};
  byId("contextEvents").innerHTML = Object.entries(eventCounts).length
    ? Object.entries(eventCounts).map(([name, count]) => `<span><strong>${escapeHtml(count)}</strong>${escapeHtml(name.replaceAll("_", " "))}</span>`).join("")
    : '<span class="empty-state">Activity composition is unavailable.</span>';
  const commonEvidence = caseData.common_case_evidence || [];
  byId("commonEvidenceCount").textContent = `${commonEvidence.length} records`;
  byId("commonEvidenceRows").innerHTML = commonEvidence.length
    ? commonEvidence.map((record) => `
      <tr>
        <td><time>${escapeHtml(dateLabel(record.timestamp))}</time></td>
        <td><span class="event-type">${escapeHtml(record.event_type || "event")}</span></td>
        <td>${escapeHtml(record.text || "Activity record available")}</td>
      </tr>`).join("")
    : '<tr><td colspan="3" class="empty-cell">No shared activity record is available.</td></tr>';
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
  renderVersionHistory(caseData);
}

function evidenceKey(record) {
  return record.evidence_id || `${record.timestamp || ""}|${record.event_type || ""}|${record.text || ""}`;
}

function versionSnapshot(caseData) {
  const supplied = Array.isArray(caseData.evidence_versions) ? caseData.evidence_versions : [];
  if (supplied.length > 1) return supplied;
  if (supplied.length === 1) {
    const only = supplied[0];
    return [only, {
      ...only,
      version_id: only.version_id === "v1" ? "v2" : "current",
      parent_version_id: only.version_id,
      reason: "Current constrained evidence reconstruction",
      evidence: caseData.evidence || only.evidence || [],
    }];
  }
  const evidence = caseData.evidence || [];
  const current = {
    version_id: "v2",
    created_at: caseData.explanation?.created_at || caseData.generated_at || null,
    reason: "Current constrained evidence reconstruction",
    evidence,
  };
  // Older bundles predate version metadata; expose a deterministic baseline
  // snapshot so the offline prototype can still demonstrate comparison.
  const baselineCount = Math.max(0, Math.floor(evidence.length / 2));
  return [
    {
      version_id: "v1",
      created_at: null,
      reason: "Initial evidence snapshot",
      evidence: evidence.slice(0, baselineCount),
    },
    current,
  ];
}

function renderVersionHistory(caseData) {
  const panel = byId("versionPanel");
  if (!panel) return;
  const versions = versionSnapshot(caseData);
  panel.hidden = caseData.view_mode !== "hypertrace_evidence";
  if (panel.hidden) return;
  const select = byId("versionSelect");
  select.innerHTML = versions.map((version, index) => `<option value="${index}">${escapeHtml(version.version_id || `v${index + 1}`)}${index === versions.length - 1 ? " (current)" : ""}</option>`).join("");
  select.value = String(Math.max(0, versions.length - 2));
  updateVersionDiff(caseData, versions);
  select.onchange = () => updateVersionDiff(caseData, versions);
  byId("rollbackVersion").onclick = async () => {
    const index = Number(select.value);
    if (index >= versions.length - 1) return toast("The selected version is already current");
    const chosen = versions[index];
    const current = versions[versions.length - 1];
    const restored = {
      ...chosen,
      version_id: `v${versions.length + 1}`,
      parent_version_id: current.version_id || `v${versions.length}`,
      created_at: new Date().toISOString(),
      reason: `Restored from ${chosen.version_id || `v${index + 1}`}`,
    };
    try {
      const path = state.preview ? "/api/preview/rollback" : `/api/session/${state.sessionId}/evidence/rollback`;
      const payload = await api(path, { method: "POST", body: JSON.stringify({
        trial_index: caseData.trial_index, case_id: caseData.case_id,
        version_id: chosen.version_id, reason: `Restored from ${chosen.version_id || `v${index + 1}`}`,
      }) });
      caseData.evidence_versions = payload.versions;
      caseData.evidence = payload.versions[payload.versions.length - 1].evidence;
      caseData.current_version_id = payload.current_version_id;
      renderTrace(caseData);
      toast(`${payload.current_version_id} created from ${chosen.version_id || `v${index + 1}`}`);
    } catch (error) {
      toast(error.message, true);
    }
  };
  byId("checkUpdates").onclick = () => checkForUpdates(caseData, false);
  byId("simulateUpdate").hidden = !state.preview || !state.demoMode;
  byId("simulateUpdate").onclick = async () => {
    try {
      const payload = await api("/api/preview/simulate-update", { method: "POST", body: JSON.stringify({ trial_index: caseData.trial_index, case_id: caseData.case_id }) });
      caseData.evidence_versions = payload.versions;
      caseData.evidence = payload.versions[payload.versions.length - 1].evidence;
      renderTrace(caseData);
      showUpdateAlert("New evidence detected. Review the version difference before continuing.", true);
      await checkForUpdates(caseData, true);
    } catch (error) { toast(error.message, true); }
  };
}

function showUpdateAlert(message, warning = false) {
  const alert = byId("updateAlert");
  if (!alert) return;
  alert.hidden = false;
  alert.textContent = message;
  alert.classList.toggle("is-warning", warning);
}

async function checkForUpdates(caseData, silent = false) {
  if (!caseData || caseData.view_mode !== "hypertrace_evidence") return;
  try {
    const query = new URLSearchParams({ case_id: caseData.case_id, since: caseData.current_version_id || "" });
    const path = state.preview ? `/api/preview/updates?${query}` : `/api/session/${state.sessionId}/updates?${query}`;
    const update = await api(path);
    if (!update.changed) { if (!silent) showUpdateAlert("No new evidence version is available."); return; }
    const versionsPath = state.preview ? `/api/preview/versions?case_id=${encodeURIComponent(caseData.case_id)}` : `/api/session/${state.sessionId}/versions?case_id=${encodeURIComponent(caseData.case_id)}`;
    const latest = await api(versionsPath);
    caseData.evidence_versions = latest.versions;
    caseData.evidence = latest.versions[latest.versions.length - 1].evidence;
    caseData.current_version_id = latest.current_version_id;
    renderTrace(caseData);
    showUpdateAlert(update.requires_rollback ? "The latest evidence update invalidates the current packet." : "A new evidence version is available.", update.requires_rollback);
    if (byId("autoRollback").checked && update.requires_rollback) {
      const versions = caseData.evidence_versions || [];
      const previous = versions.length > 1 ? versions[versions.length - 2] : null;
      if (previous) {
        const path = state.preview ? "/api/preview/rollback" : `/api/session/${state.sessionId}/evidence/rollback`;
        const payload = await api(path, { method: "POST", body: JSON.stringify({ trial_index: caseData.trial_index, case_id: caseData.case_id, version_id: previous.version_id, reason: "Automatic rollback after invalidating evidence update" }) });
        caseData.evidence_versions = payload.versions;
        caseData.evidence = payload.versions[payload.versions.length - 1].evidence;
        caseData.current_version_id = payload.current_version_id;
        renderTrace(caseData);
        showUpdateAlert(`Automatically restored ${payload.current_version_id} after the invalidating update.`, false);
      }
    }
  } catch (error) { if (!silent) toast(error.message, true); }
}

function stopUpdatePolling() {
  if (state.updateTimer) window.clearInterval(state.updateTimer);
  state.updateTimer = null;
}

function startUpdatePolling(caseData) {
  stopUpdatePolling();
  if (caseData.view_mode !== "hypertrace_evidence" || !state.demoMode) return;
  state.updateTimer = window.setInterval(() => checkForUpdates(state.currentCase, true), 15000);
}

function updateVersionDiff(caseData, versions) {
  const index = Number(byId("versionSelect").value);
  const current = versions[versions.length - 1] || { evidence: [] };
  const compared = versions[index] || current;
  const currentKeys = new Set((current.evidence || []).map(evidenceKey));
  const comparedKeys = new Set((compared.evidence || []).map(evidenceKey));
  const added = (current.evidence || []).filter(record => !comparedKeys.has(evidenceKey(record)));
  const removed = (compared.evidence || []).filter(record => !currentKeys.has(evidenceKey(record)));
  const unchanged = (current.evidence || []).filter(record => comparedKeys.has(evidenceKey(record))).length;
  byId("versionStatus").textContent = index === versions.length - 1 ? "Current version" : `Comparing ${compared.version_id || `v${index + 1}`} → ${current.version_id || `v${versions.length}`}`;
  byId("versionDiff").innerHTML = `<span><strong>${added.length}</strong> added</span><span><strong>${removed.length}</strong> removed</span><span><strong>${unchanged}</strong> retained</span><small>${escapeHtml(compared.reason || "Evidence snapshot")}</small>`;
}

function decisionLabel(value) {
  return value === "coordinated" ? "Coordinated" : "Not coordinated";
}

function resetDecision(caseData) {
  document.querySelectorAll('input[name="decision"]').forEach((input) => { input.checked = false; });
  byId("confidence").value = "50";
  byId("confidenceValue").textContent = "50";
  byId("rationale").value = "";
  const phase = caseData.phase || "assisted";
  const initial = caseData.initial_response || null;
  const isInitial = phase === "initial";
  const isReady = phase === "ready_to_reveal";
  const isAssisted = phase === "assisted";
  const isSubmitted = phase === "submitted";

  byId("decisionFields").hidden = isReady;
  byId("initialLockedSummary").hidden = !initial;
  byId("revealAssistance").hidden = !isReady || state.preview;
  byId("revealAssistance").disabled = false;
  byId("submitDecision").hidden = isReady || isSubmitted;
  byId("rationaleField").hidden = !(isAssisted || isSubmitted);
  byId("reviseDecision").hidden = !isSubmitted || state.preview;
  byId("continueTrial").hidden = !isSubmitted || state.preview;

  if (initial) {
    byId("lockedDecision").textContent = decisionLabel(initial.decision);
    byId("lockedConfidence").textContent = `${initial.confidence}% confidence`;
  }

  if (isInitial) {
    byId("decisionStage").textContent = "Stage 1 of 3";
    byId("decisionTitle").textContent = "Initial decision";
    byId("submitDecision").disabled = true;
    byId("submitDecision").innerHTML = 'Lock initial decision <i data-lucide="lock" aria-hidden="true"></i>';
    byId("decisionNote").textContent = "No model assistance is visible. This response is locked before the reveal.";
  } else if (isReady) {
    byId("decisionStage").textContent = "Stage 2 of 3";
    byId("decisionTitle").textContent = "Reveal assistance";
    byId("decisionNote").textContent = "Opening assistance starts the assisted-review timer.";
  } else if (isSubmitted) {
    const finalResponse = caseData.final_response || {};
    byId("decisionStage").textContent = "Submitted judgment";
    byId("decisionTitle").textContent = "Judgment recorded";
    byId("decisionNote").textContent = "This demonstration keeps the submitted judgment editable and records each revision.";
    const finalInput = document.querySelector(`input[name="decision"][value="${finalResponse.decision}"]`);
    if (finalInput) finalInput.checked = true;
    byId("confidence").value = String(finalResponse.confidence ?? 50);
    byId("confidenceValue").textContent = String(finalResponse.confidence ?? 50);
    byId("rationale").value = finalResponse.rationale || "";
    byId("submitDecision").disabled = true;
  } else {
    byId("decisionStage").textContent = "Stage 3 of 3";
    byId("decisionTitle").textContent = "Final decision";
    byId("decisionNote").textContent = "Submit a final judgment after inspecting the assigned assistance.";
    if (initial) {
      const initialInput = document.querySelector(`input[name="decision"][value="${initial.decision}"]`);
      if (initialInput) initialInput.checked = true;
      byId("confidence").value = String(initial.confidence);
      byId("confidenceValue").textContent = String(initial.confidence);
    }
    byId("submitDecision").disabled = state.preview || !initial;
    byId("submitDecision").innerHTML = state.preview
      ? 'Preview only <i data-lucide="eye" aria-hidden="true"></i>'
      : 'Submit final decision <i data-lucide="arrow-right" aria-hidden="true"></i>';
  }
}

function renderCase(caseData) {
  state.currentCase = caseData;
  state.currentPhase = caseData.phase || "assisted";
  byId("operationLabel").textContent = caseData.operation_label;
  byId("caseTitle").textContent = `Case ${String(caseData.trial_index + 1).padStart(2, "0")}`;
  const completed = caseData.trial_index;
  const total = caseData.trial_count;
  byId("progressText").textContent = `Case ${completed + 1} of ${total}`;
  const progress = completed / total * 100;
  byId("progressBar").style.width = `${progress}%`;
  byId("progressWrap").querySelector("[role=progressbar]").setAttribute("aria-valuenow", String(Math.round(progress)));
  renderCommonContext(caseData);
  const assisted = ["assisted", "submitted"].includes(state.currentPhase);
  byId("recommendationSection").hidden = !assisted;
  byId("initialStageNotice").hidden = assisted;
  byId("stageStatus").textContent = assisted ? "Model assistance revealed" : "Initial judgment";
  if (assisted) renderRecommendation(caseData);
  byId("riskOnlyNotice").hidden = !assisted || caseData.view_mode !== "risk_only";
  renderSignals(caseData);
  renderTrace(caseData);
  resetDecision(caseData);
  setView("trialView");
  byId("phaseLabel").textContent = assisted ? "AI-assisted judgment" : "Initial judgment";
  refreshIcons();
  startUpdatePolling(caseData);
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

async function loadPreview(mode = null, index = null) {
  if (mode) byId("previewMode").value = mode;
  if (index != null) state.previewIndex = index;
  setLoading(true, "Loading preview case");
  try {
    const selectedMode = byId("previewMode").value;
    const payload = await api(`/api/preview?mode=${encodeURIComponent(selectedMode)}&case_index=${state.previewIndex}`);
    state.preview = true;
    state.previewCount = payload.available_cases;
    renderCase(payload.case);
    byId("previewControls").hidden = false;
    byId("progressWrap").hidden = true;
    byId("previewCaseLabel").textContent = `${state.previewIndex + 1} / ${state.previewCount}`;
    refreshIcons();
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
  const initialPhase = state.currentPhase === "initial";
  setLoading(true, initialPhase ? "Locking initial judgment" : "Saving final judgment");
  try {
    const payload = await api(`/api/session/${state.sessionId}/${initialPhase ? "initial-response" : "response"}`, {
      method: "POST",
      body: JSON.stringify({
        trial_index: state.currentCase.trial_index,
        case_id: state.currentCase.case_id,
        decision: selected.value,
        confidence: Number(byId("confidence").value),
        ...(initialPhase ? {} : { rationale: byId("rationale").value }),
      }),
    });
    if (!initialPhase && state.demoMode && payload.case) {
      renderCase(payload.case);
    } else {
      await loadNextTrial();
    }
  } catch (error) {
    toast(error.message, true);
    byId("submitDecision").disabled = false;
  } finally {
    setLoading(false);
  }
}

async function reviseDecision() {
  if (!state.currentCase || state.preview || state.currentPhase !== "submitted") return;
  const selected = document.querySelector('input[name="decision"]:checked');
  if (!selected) return;
  try {
    const payload = await api(`/api/session/${state.sessionId}/response/revise`, { method: "POST", body: JSON.stringify({
      trial_index: state.currentCase.trial_index, case_id: state.currentCase.case_id,
      decision: selected.value, confidence: Number(byId("confidence").value),
      rationale: byId("rationale").value, reason: "Reviewer revised the submitted judgment",
    }) });
    state.currentCase.final_response = { decision: selected.value, confidence: Number(byId("confidence").value), rationale: byId("rationale").value };
    toast(`Judgment revision ${payload.revision_no} saved`);
  } catch (error) { toast(error.message, true); }
}

async function revealAssistance() {
  if (!state.currentCase || state.currentPhase !== "ready_to_reveal") return;
  byId("revealAssistance").disabled = true;
  setLoading(true, "Opening assigned model assistance");
  try {
    const payload = await api(`/api/session/${state.sessionId}/reveal`, {
      method: "POST",
      body: JSON.stringify({
        trial_index: state.currentCase.trial_index,
        case_id: state.currentCase.case_id,
      }),
    });
    renderCase(payload.case);
  } catch (error) {
    toast(error.message, true);
    byId("revealAssistance").disabled = false;
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
    input.addEventListener("change", () => {
      if (!state.preview) byId("submitDecision").disabled = false;
    });
  });
  byId("submitDecision").addEventListener("click", submitDecision);
  byId("reviseDecision").addEventListener("click", reviseDecision);
  byId("continueTrial").addEventListener("click", loadNextTrial);
  byId("revealAssistance").addEventListener("click", revealAssistance);
  byId("questionnaireForm").addEventListener("submit", submitQuestionnaire);
  byId("previewMode").addEventListener("change", () => loadPreview());
  byId("previousPreview").addEventListener("click", () => {
    loadPreview(null, (state.previewIndex - 1 + state.previewCount) % state.previewCount);
  });
  byId("nextPreview").addEventListener("click", () => {
    loadPreview(null, (state.previewIndex + 1) % state.previewCount);
  });
}

async function initialize() {
  bindEvents();
  refreshIcons();
  try {
    const health = await api("/api/health");
    state.demoMode = Boolean(health.demo_data);
    if (health.demo_data) byId("phaseLabel").textContent = "Demonstration study";
  } catch (error) {
    toast("Study service is unavailable", true);
  }
  const parameters = new URLSearchParams(window.location.search);
  const requestedPreview = parameters.get("preview");
  if (requestedPreview) {
    state.preview = true;
    await loadPreview(requestedPreview, Number(parameters.get("case") || 0));
  } else if (state.sessionId) {
    setView("briefingView");
  }
}

initialize();
