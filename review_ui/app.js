const demoPackets = [
  {
    schema_version: "hypertrace.consensus-evidence-packet.v1",
    case_id: "user:demo-1047", operation: "honduras", labels_consumed: false,
    prediction: { consensus_percentile: 0.994, consensus_rank: 1, checkpoint_rank_std: 0.018, recomputed_baseline_percentile: 0.994, baseline_percentile_match_error: 0, decision: "review_candidate", probability_calibrated: false },
    geometry: { checkpoint_count: 15, poincare_radius_mean: 0.438, poincare_radius_std: 0.111, geodesic_margin_mean: 0.246, geodesic_margin_std: 1.964, coordination_vote_fraction: 0.733, coordinate_averaging_disabled: true },
    counterfactual: { intervention: "remove all incident learned-relation edges for this account", consensus_percentile_before: 0.994, consensus_percentile_after: 0.612, consensus_percentile_change: -0.382, mean_probability_change: -0.34, checkpoint_count: 15 },
    temporal_summary: { event_type_counts: { retweet: 31, reply: 12, mention: 7, post: 5 }, first_event_at: "2019-01-18T08:12:00+00:00", last_event_at: "2019-04-03T21:44:00+00:00", decision_time_scope: "full_release_snapshot" },
    relation_summary: { retweets: 42, replies_to: 18, mentions: 9 },
    evidence: [
      { evidence_id: "evt:demo:0192", actor_id: "demo-1047", timestamp: "2019-04-03T21:44:00+00:00", event_type: "retweet", text: "Repeated amplification of a shared campaign URL." },
      { evidence_id: "evt:demo:0177", actor_id: "demo-1047", timestamp: "2019-04-03T20:51:00+00:00", event_type: "mention", text: "Mentioned three accounts in the same coordination cluster." },
      { evidence_id: "evt:demo:0128", actor_id: "demo-1047", timestamp: "2019-04-02T14:13:00+00:00", event_type: "reply", text: "Reply occurred inside a dense synchronized window." }
    ],
    critical_edges: [
      { evidence_id: "edge:demo:001", source_type: "user", source_id: "demo-1047", target_type: "user", target_id: "demo-2218", edge_type: "retweets", timestamp: "2019-04-03T21:44:00+00:00" },
      { evidence_id: "edge:demo:002", source_type: "user", source_id: "demo-1047", target_type: "user", target_id: "demo-6630", edge_type: "mentions", timestamp: "2019-04-03T20:51:00+00:00" },
      { evidence_id: "edge:demo:003", source_type: "user", source_id: "demo-9184", target_type: "user", target_id: "demo-1047", edge_type: "replies_to", timestamp: "2019-04-02T14:13:00+00:00" },
      { evidence_id: "edge:demo:004", source_type: "user", source_id: "demo-1047", target_type: "tweet", target_id: "tweet-493", edge_type: "posts", timestamp: "2019-04-01T08:09:00+00:00" }
    ],
    reference_universe: { source: "consensus_scores.user_id", account_count: 186420, target_label_values_consumed: false, eligibility_membership_consumed: true },
    provenance: { bundle_manifest: "/runtime/external/honduras/manifest.json", bundle_manifest_sha256: "28e6695133872a4070120064a95aa0710ad3b9ba0ec0c410f94c74c24c05ab57", freeze_manifest: "/runtime/frozen/base_formal_v5_uae/manifest.json", freeze_manifest_sha256: "4d08f71d49edb3870929367e77b56496f09c3ad3add103682e33d6a7b62fa7ac", consensus_scores: "/runtime/audits/v5_rank_consensus/honduras/rank_consensus_scores.csv.gz", consensus_scores_sha256: "f6959c0251d7f10e4bbc1462cc530071763c402058fbf9a657dc5d78185785cb" },
    warnings: ["Demonstration packet. No target labels are displayed.", "Evidence covers the full release snapshot, not a historical online decision cutoff."]
  },
  {
    case_id: "user:demo-2781", operation: "honduras", labels_consumed: false,
    prediction: { consensus_percentile: 0.988, consensus_rank: 2, checkpoint_rank_std: 0.026, recomputed_baseline_percentile: 0.988, baseline_percentile_match_error: 0, probability_calibrated: false },
    geometry: { checkpoint_count: 15, poincare_radius_mean: 0.402, poincare_radius_std: 0.089, geodesic_margin_mean: 0.174, geodesic_margin_std: 1.13, coordination_vote_fraction: 0.667, coordinate_averaging_disabled: true },
    counterfactual: { consensus_percentile_before: 0.988, consensus_percentile_after: 0.704, consensus_percentile_change: -0.284, mean_probability_change: -0.21, checkpoint_count: 15 },
    temporal_summary: { event_type_counts: { retweet: 18, reply: 9, mention: 4 }, first_event_at: "2019-02-02T10:22:00+00:00", last_event_at: "2019-04-01T17:18:00+00:00", decision_time_scope: "full_release_snapshot" },
    relation_summary: { retweets: 24, replies_to: 11, mentions: 5 },
    evidence: [], critical_edges: [], reference_universe: { source: "consensus_scores.user_id", account_count: 186420, target_label_values_consumed: false, eligibility_membership_consumed: true }, provenance: {}, warnings: []
  },
  {
    case_id: "user:demo-8356", operation: "uae", labels_consumed: false,
    prediction: { consensus_percentile: 0.982, consensus_rank: 3, checkpoint_rank_std: 0.041, recomputed_baseline_percentile: 0.982, baseline_percentile_match_error: 0, probability_calibrated: false },
    geometry: { checkpoint_count: 15, poincare_radius_mean: 0.371, poincare_radius_std: 0.104, geodesic_margin_mean: 0.091, geodesic_margin_std: 1.46, coordination_vote_fraction: 0.60, coordinate_averaging_disabled: true },
    counterfactual: { consensus_percentile_before: 0.982, consensus_percentile_after: 0.811, consensus_percentile_change: -0.171, mean_probability_change: -0.12, checkpoint_count: 15 },
    temporal_summary: { event_type_counts: { retweet: 15, quote: 8, post: 4 }, first_event_at: "2018-11-21T04:01:00+00:00", last_event_at: "2019-03-19T23:11:00+00:00", decision_time_scope: "full_release_snapshot" },
    relation_summary: { retweets: 17, quotes: 8 },
    evidence: [], critical_edges: [], reference_universe: { source: "consensus_scores.user_id", account_count: 93080, target_label_values_consumed: false, eligibility_membership_consumed: true }, provenance: {}, warnings: []
  }
];

let packets = demoPackets;
let selectedIndex = 0;
const reviewEvents = new Map();

const byId = (id) => document.getElementById(id);
const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, Number(value) || 0));
const percent = (value, digits = 1) => `${(Number(value) * 100).toFixed(digits)}%`;
const shortId = (value) => String(value || "—").replace(/^user:/, "");
const dateLabel = (value) => {
  if (!value) return "Not available";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
};
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character]));

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
}

function renderQueue(filter = "") {
  const query = filter.trim().toLowerCase();
  const rows = packets.map((packet, index) => ({ packet, index })).filter(({ packet }) => String(packet.case_id).toLowerCase().includes(query));
  byId("caseCount").textContent = String(rows.length);
  byId("caseList").innerHTML = rows.map(({ packet, index }) => {
    const prediction = packet.prediction || {};
    const operation = packet.operation || "unknown";
    return `<button class="case-item ${index === selectedIndex ? "is-selected" : ""}" type="button" data-case-index="${index}">
      <span class="case-item-top"><strong>${escapeHtml(shortId(packet.case_id))}</strong><span class="case-item-priority">${percent(prediction.consensus_percentile || 0)}</span></span>
      <span class="case-item-meta"><span>${escapeHtml(operation.toUpperCase())}</span><span>Rank ${escapeHtml(prediction.consensus_rank || "—")}</span></span>
    </button>`;
  }).join("");
  document.querySelectorAll("[data-case-index]").forEach((button) => button.addEventListener("click", () => {
    selectedIndex = Number(button.dataset.caseIndex);
    renderAll();
  }));
}

function renderNetwork(packet) {
  const svg = byId("networkGraph");
  const caseId = shortId(packet.case_id);
  const edges = (packet.critical_edges || []).slice(0, 16);
  const nodes = new Map([[caseId, { id: caseId, focal: true }]]);
  edges.forEach((edge) => {
    const source = shortId(edge.source_id);
    const target = shortId(edge.target_id);
    if (source) nodes.set(source, { id: source, focal: source === caseId });
    if (target) nodes.set(target, { id: target, focal: target === caseId });
  });
  if (nodes.size === 1) {
    ["related-01", "related-02", "content-03", "related-04"].forEach((id) => nodes.set(id, { id, focal: false }));
  }
  const width = 720, height = 350, center = { x: 360, y: 175 };
  const others = [...nodes.values()].filter((node) => !node.focal).slice(0, 14);
  const positions = new Map([[caseId, center]]);
  others.forEach((node, index) => {
    const angle = (Math.PI * 2 * index / Math.max(others.length, 1)) - Math.PI / 2;
    const radius = 120 + (index % 2) * 35;
    positions.set(node.id, { x: center.x + Math.cos(angle) * radius * 1.6, y: center.y + Math.sin(angle) * radius });
  });
  const effectiveEdges = edges.length ? edges : others.map((node, index) => ({ source_id: caseId, target_id: node.id, edge_type: ["retweets", "mentions", "replies_to"][index % 3] }));
  const edgeMarkup = effectiveEdges.map((edge) => {
    const source = positions.get(shortId(edge.source_id)) || center;
    const target = positions.get(shortId(edge.target_id)) || center;
    const mx = (source.x + target.x) / 2, my = (source.y + target.y) / 2;
    return `<g><line class="network-edge" x1="${source.x}" y1="${source.y}" x2="${target.x}" y2="${target.y}"></line><text class="network-edge-label" x="${mx}" y="${my - 4}" text-anchor="middle">${escapeHtml(edge.edge_type || "related")}</text></g>`;
  }).join("");
  const nodeMarkup = [...nodes.values()].slice(0, 15).map((node) => {
    const point = positions.get(node.id) || center;
    const radius = node.focal ? 30 : 19;
    const label = node.focal ? "Reviewed" : node.id.length > 12 ? `${node.id.slice(0, 9)}…` : node.id;
    return `<g><circle class="network-node ${node.focal ? "focal" : ""}" cx="${point.x}" cy="${point.y}" r="${radius}"></circle><text class="network-node-label ${node.focal ? "focal" : ""}" x="${point.x}" y="${point.y + 4}" text-anchor="middle">${escapeHtml(label)}</text></g>`;
  }).join("");
  svg.innerHTML = `<desc id="networkDescription">Relationship neighborhood for ${escapeHtml(caseId)}.</desc>${edgeMarkup}${nodeMarkup}`;
}

function renderEvidence(packet) {
  const events = (packet.evidence || []).map((row) => ({ ...row, recordKind: "Event" }));
  const edges = (packet.critical_edges || []).map((row) => ({ ...row, recordKind: "Edge" }));
  const rows = [...events, ...edges];
  byId("evidenceCount").textContent = `${rows.length} records`;
  byId("evidenceRows").innerHTML = rows.length ? rows.map((row) => {
    const source = row.actor_id || row.source_id || "—";
    const target = row.target_id || "—";
    const detail = row.text || row.edge_type || row.event_type || "—";
    return `<tr><td><code>${escapeHtml(row.evidence_id || row.event_id || "unassigned")}</code></td><td>${escapeHtml(dateLabel(row.timestamp))}</td><td>${escapeHtml(row.recordKind)}</td><td>${escapeHtml(source)}</td><td>${escapeHtml(target)}</td><td>${escapeHtml(detail)}</td></tr>`;
  }).join("") : `<tr><td colspan="6">No source rows are present in this demonstration packet.</td></tr>`;
}

function renderProvenance(packet) {
  const entries = Object.entries(packet.provenance || {}).filter(([key]) => key.endsWith("sha256"));
  byId("provenanceList").innerHTML = entries.length ? entries.map(([key, value]) => `<div><dt>${escapeHtml(key.replaceAll("_", " "))}</dt><dd><code>${escapeHtml(value)}</code><button class="copy-hash" type="button" data-copy="${escapeHtml(value)}" aria-label="Copy hash"><i data-lucide="copy" aria-hidden="true"></i></button></dd></div>`).join("") : `<div><dt>Artifact hashes</dt><dd><code>Not present in this demonstration packet</code></dd></div>`;
  document.querySelectorAll("[data-copy]").forEach((button) => button.addEventListener("click", async () => {
    await navigator.clipboard.writeText(button.dataset.copy);
    showToast("Hash copied");
  }));
}

function renderAudit(packet) {
  const stored = reviewEvents.get(packet.case_id) || [];
  const base = [
    { title: "Packet ingested", detail: "Evidence packet hash registered" },
    { title: "Evidence linked", detail: `${(packet.evidence || []).length + (packet.critical_edges || []).length} source records resolved` },
    { title: "Ready for review", detail: "Target labels remain concealed" }
  ];
  byId("auditEvents").innerHTML = [...base, ...stored].map((event) => `<li><strong>${escapeHtml(event.title)}</strong><span>${escapeHtml(event.detail)}</span></li>`).join("");
}

function renderCase() {
  const packet = packets[selectedIndex] || packets[0];
  if (!packet) return;
  const prediction = packet.prediction || {};
  const geometry = packet.geometry || {};
  const counterfactual = packet.counterfactual || {};
  const temporal = packet.temporal_summary || {};
  const relations = packet.relation_summary || {};
  const priority = clamp(prediction.consensus_percentile);
  const vote = clamp(geometry.coordination_vote_fraction);
  const before = clamp(counterfactual.consensus_percentile_before);
  const after = clamp(counterfactual.consensus_percentile_after);
  const change = Number(counterfactual.consensus_percentile_change || 0);
  const margin = Number(geometry.geodesic_margin_mean || 0);
  const radius = clamp(geometry.poincare_radius_mean);
  const agreement = clamp(1 - Number(prediction.checkpoint_rank_std || 0));

  byId("datasetName").textContent = `${String(packet.operation || "External").toUpperCase()} operation`;
  byId("operationName").textContent = String(packet.operation || "External operation").toUpperCase();
  byId("caseTitle").textContent = packet.case_id || "Unknown case";
  byId("priorityValue").textContent = percent(priority);
  byId("rankValue").textContent = `#${prediction.consensus_rank || "—"}`;
  byId("rankSpread").textContent = `Rank spread ${percent(prediction.checkpoint_rank_std || 0, 2)}`;
  byId("voteValue").textContent = percent(vote, 0);
  byId("deletionValue").textContent = `${change >= 0 ? "+" : ""}${(change * 100).toFixed(1)} pp`;
  byId("marginValue").textContent = `${margin >= 0 ? "+" : ""}${margin.toFixed(3)} ± ${Number(geometry.geodesic_margin_std || 0).toFixed(3)}`;
  byId("radiusValue").textContent = `${radius.toFixed(3)} ± ${Number(geometry.poincare_radius_std || 0).toFixed(3)}`;
  byId("agreementValue").textContent = percent(agreement, 1);
  byId("marginBar").style.width = `${clamp((margin + 2) / 4) * 100}%`;
  byId("radiusBar").style.width = `${radius * 100}%`;
  byId("agreementBar").style.width = `${agreement * 100}%`;
  byId("beforeValue").textContent = percent(before);
  byId("afterValue").textContent = percent(after);
  byId("beforeBar").style.width = `${before * 100}%`;
  byId("afterBar").style.width = `${after * 100}%`;
  byId("counterfactualDirection").textContent = change < 0 ? `${Math.abs(change * 100).toFixed(1)} pp decrease` : `${(change * 100).toFixed(1)} pp increase`;
  byId("counterfactualText").textContent = change < 0 ? "The account's review priority falls when all learned incident relations are removed." : "The deletion intervention does not reduce review priority; inspect source evidence before escalation.";
  byId("firstEvent").textContent = dateLabel(temporal.first_event_at);
  byId("lastEvent").textContent = dateLabel(temporal.last_event_at);
  const counts = temporal.event_type_counts || {};
  const eventTotal = Object.values(counts).reduce((sum, value) => sum + Number(value || 0), 0);
  byId("eventTotal").textContent = `${eventTotal} events`;
  byId("eventCounts").innerHTML = Object.entries(counts).map(([key, value]) => `<span>${escapeHtml(key)} <strong>${escapeHtml(value)}</strong></span>`).join("") || "<span>No event counts</span>";
  byId("relationCounts").innerHTML = Object.entries(relations).map(([key, value]) => `<span>${escapeHtml(key)} <strong>${escapeHtml(value)}</strong></span>`).join("") || "<span>No relation counts</span>";
  byId("integrityStatus").innerHTML = prediction.baseline_percentile_match_error <= 0.0001 ? '<i data-lucide="shield-check" aria-hidden="true"></i> Integrity verified' : '<i data-lucide="shield-alert" aria-hidden="true"></i> Review integrity warning';
  renderNetwork(packet);
  renderEvidence(packet);
  renderProvenance(packet);
  renderAudit(packet);
  refreshIcons();
}

function renderAll() {
  renderQueue(byId("caseSearch").value);
  renderCase();
}

function showToast(message) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.classList.add("is-visible");
  clearTimeout(showToast.timeout);
  showToast.timeout = setTimeout(() => toast.classList.remove("is-visible"), 2200);
}

function parsePacketText(text) {
  const trimmed = text.trim();
  if (!trimmed) return [];
  if (trimmed.startsWith("[")) return JSON.parse(trimmed);
  if (trimmed.startsWith("{") && !trimmed.includes("\n")) return [JSON.parse(trimmed)];
  return trimmed.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

byId("openPacket").addEventListener("click", () => byId("packetFile").click());
byId("packetFile").addEventListener("change", async (event) => {
  try {
    const loaded = [];
    for (const file of event.target.files) loaded.push(...parsePacketText(await file.text()));
    if (!loaded.length) throw new Error("No packets found");
    packets = loaded;
    selectedIndex = 0;
    renderAll();
    showToast(`${loaded.length} packet${loaded.length === 1 ? "" : "s"} loaded`);
  } catch (error) {
    showToast(`Could not open packets: ${error.message}`);
  }
});
byId("caseSearch").addEventListener("input", (event) => renderQueue(event.target.value));
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((candidate) => {
    const active = candidate === tab;
    candidate.classList.toggle("is-active", active);
    candidate.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("is-active", panel.dataset.panel === tab.dataset.tab));
}));
document.querySelectorAll("[data-decision]").forEach((button) => button.addEventListener("click", () => {
  const packet = packets[selectedIndex];
  const rationale = byId("reviewRationale").value.trim();
  const labels = { confirmed: "Escalation confirmed", rejected: "Recommendation rejected", more_evidence: "Additional evidence requested" };
  const events = reviewEvents.get(packet.case_id) || [];
  events.push({ title: labels[button.dataset.decision], detail: rationale || "No review note recorded" });
  reviewEvents.set(packet.case_id, events);
  byId("reviewRationale").value = "";
  renderAudit(packet);
  showToast(labels[button.dataset.decision]);
}));

window.addEventListener("DOMContentLoaded", () => {
  renderAll();
  refreshIcons();
});
