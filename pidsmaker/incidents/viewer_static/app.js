/* Read-only, dependency-local incident explorer. All evidence enters as text. */
"use strict";

const state = {
  summary: null,
  active: -1,
  bounds: null,
  focus: null,
  selectedEdge: null,
  selectedNode: null,
  views: {},
  simulations: {},
  requestId: 0,
  memoryRequestId: 0,
  selectedMemory: null,
  timer: null,
};

function byId(id) { return document.getElementById(id); }

async function getJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Request failed (" + response.status + "): " +
      (await response.text()).replace(/<[^>]*>/g, " ").trim().slice(0, 140));
  }
  return response.json();
}

function setStatus(message, kind) {
  const element = byId("status");
  element.textContent = message;
  element.className = "status" + (kind ? " " + kind : "");
}

function formatTime(raw) {
  if (raw === null || raw === undefined) return "—";
  try {
    return new Date(Number(BigInt(String(raw)) / 1000000n)).toISOString();
  } catch (_) {
    return String(raw) + " ns";
  }
}

function formatShort(raw) {
  const full = formatTime(raw);
  return full.length > 19 ? full.slice(0, 19).replace("T", " ") + " UTC" : full;
}

function timeAt(percent) {
  const start = BigInt(state.bounds[0]);
  const span = BigInt(state.bounds[1]) - start;
  return (start + span * BigInt(percent) / 100n).toString();
}

function selectedRange() {
  return [timeAt(byId("time-from").value), timeAt(byId("time-to").value)];
}

function updateTimeLabel() {
  if (!state.bounds) return;
  const range = selectedRange();
  byId("time-label").textContent = formatShort(range[0]) + "  →  " + formatShort(range[1]);
}

function detailItem(label, value) {
  const item = document.createElement("div");
  item.className = "detail-item";
  const heading = document.createElement("span");
  heading.textContent = label;
  const content = document.createElement("strong");
  content.textContent = value === null || value === undefined ? "—" : String(value);
  item.append(heading, content);
  return item;
}

function showEdge(edge) {
  state.selectedEdge = edge.id;
  state.selectedNode = null;
  byId("detail-title").textContent = "Event " + edge.source.id + " → " + edge.target.id;
  byId("detail-role").textContent = edge.role.replaceAll("_", " ");
  const grid = document.createElement("div");
  grid.className = "detail-grid";
  [
    ["Anomaly loss", edge.score === null ? "Unscored" : edge.score],
    ["Operation", edge.operation],
    ["Timestamp (UTC)", formatTime(edge.time)],
    ["Exact time (ns)", edge.time],
    ["Event UUID", edge.event_uuid],
    ["Relation ID", edge.edge_type],
    ["Multiedge key", edge.key],
    ["Evidence role", edge.role.replaceAll("_", " ")],
  ].forEach(([label, value]) => grid.appendChild(detailItem(label, value)));
  byId("detail-body").replaceChildren(grid);
  updateHighlights();
}

function showNode(node) {
  state.selectedNode = node.id;
  state.selectedEdge = null;
  byId("focus-node").value = node.id;
  byId("detail-title").textContent = "Node " + node.id;
  byId("detail-role").textContent = "entity";
  const grid = document.createElement("div");
  grid.className = "detail-grid";
  grid.append(detailItem("Node ID", node.id), detailItem("Node type", node.type),
    detailItem("Provenance label", node.label));
  byId("detail-body").replaceChildren(grid);
  updateHighlights();
}

function updateHighlights() {
  for (const view of Object.values(state.views)) {
    view.links.classed("is-selected", d => d.id === state.selectedEdge);
    view.nodes.classed("is-selected", d => d.id === state.selectedNode);
  }
}

function nodeKind(node) {
  const kind = String(node.type).toLowerCase();
  if (kind.includes("process") || kind.includes("subject")) return "type-process";
  if (kind.includes("file")) return "type-file";
  if (kind.includes("netflow") || kind.includes("socket")) return "type-netflow";
  return "type-other";
}

function edgePath(edge) {
  const sx = edge.source.x, sy = edge.source.y;
  const tx = edge.target.x, ty = edge.target.y;
  const dx = tx - sx, dy = ty - sy;
  const distance = Math.sqrt(dx * dx + dy * dy);
  if (distance < 1 || edge.source.id === edge.target.id) {
    return "M" + sx + "," + sy + " C" + (sx + 35) + "," + (sy - 48) +
      " " + (sx - 35) + "," + (sy - 48) + " " + (sx + 10) + "," + (sy - 4);
  }
  const nx = -dy / distance, ny = dx / distance;
  const mx = (sx + tx) / 2 + nx * edge.curve;
  const my = (sy + ty) / 2 + ny * edge.curve;
  return "M" + sx + "," + sy + " Q" + mx + "," + my + " " + tx + "," + ty;
}

function drawGraph(elementId, graph, paneName) {
  if (state.simulations[paneName]) state.simulations[paneName].stop();
  const host = byId(elementId);
  host.replaceChildren();
  delete state.views[paneName];
  const width = Math.max(host.clientWidth || 450, 300);
  const height = Math.max(host.clientHeight || 390, 280);
  const nodes = graph.nodes.map((node, index) => ({
    ...node,
    x: width / 2 + 110 * Math.cos(index * 2.399963),
    y: height / 2 + 110 * Math.sin(index * 2.399963),
  }));
  if (!nodes.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No events in this time window.";
    host.appendChild(empty);
    return;
  }
  const edges = graph.edges.map(edge => ({ ...edge }));
  const buckets = new Map();
  edges.forEach(edge => {
    const pair = [edge.source, edge.target].sort().join("|");
    if (!buckets.has(pair)) buckets.set(pair, []);
    buckets.get(pair).push(edge);
  });
  buckets.forEach(bucket => bucket.forEach((edge, index) => {
    edge.curve = (index - (bucket.length - 1) / 2) * 27;
  }));

  const svg = d3.select(host).append("svg")
    .attr("viewBox", "0 0 " + width + " " + height)
    .attr("role", "img")
    .attr("aria-label", paneName + " provenance network");
  const arrowId = "arrow-" + paneName;
  svg.append("defs").append("marker").attr("id", arrowId)
    .attr("viewBox", "0 -5 10 10").attr("refX", 17).attr("refY", 0)
    .attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto")
    .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#9facc5");
  const layer = svg.append("g");
  svg.call(d3.zoom().scaleExtent([0.3, 5]).on("zoom", () => {
    layer.attr("transform", d3.event.transform);
  }));
  const links = layer.append("g").selectAll("path").data(edges).enter().append("path")
    .attr("class", edge => "graph-edge role-" + edge.role.replaceAll("_", "-"))
    .attr("marker-end", "url(#" + arrowId + ")")
    .on("click", edge => { d3.event.stopPropagation(); showEdge(edge); });
  links.append("title").text(edge => edge.role.replaceAll("_", " ") + " · " +
    edge.operation + " · loss " + (edge.score === null ? "unscored" : edge.score));

  const nodeGroups = layer.append("g").selectAll("g").data(nodes).enter().append("g")
    .attr("class", node => "graph-node " + nodeKind(node))
    .on("click", node => { d3.event.stopPropagation(); showNode(node); });
  nodeGroups.append("circle").attr("r", 10);
  if (nodes.length <= 24) {
    nodeGroups.append("text").attr("x", 14).attr("y", 4).text(node => node.id);
  }
  nodeGroups.append("title").text(node => node.label + " (" + node.id + ")");

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(edges).id(node => node.id).distance(88).strength(0.6))
    .force("charge", d3.forceManyBody().strength(-210))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide(24));
  const tick = () => {
    links.attr("d", edgePath);
    nodeGroups.attr("transform", node => "translate(" + node.x + "," + node.y + ")");
  };
  simulation.on("tick", tick);
  nodeGroups.call(d3.drag()
    .on("start", node => { if (!d3.event.active) simulation.alphaTarget(0.3).restart(); node.fx = node.x; node.fy = node.y; })
    .on("drag", node => { node.fx = d3.event.x; node.fy = d3.event.y; })
    .on("end", node => { if (!d3.event.active) simulation.alphaTarget(0); node.fx = null; node.fy = null; }));
  for (let index = 0; index < 50; index++) simulation.tick();
  tick();
  simulation.alpha(0.1).restart();
  state.simulations[paneName] = simulation;
  state.views[paneName] = { links, nodes: nodeGroups };
  updateHighlights();
}

function renderIncidentList() {
  const host = byId("incident-list");
  host.replaceChildren();
  state.summary.incidents.forEach(incident => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "incident-button" + (incident.id === state.active ? " active" : "");
    const title = document.createElement("strong");
    title.textContent = "Incident " + String(incident.id + 1).padStart(2, "0");
    const stats = document.createElement("small");
    stats.className = "mini-stats";
    stats.textContent = incident.seeds + " seeds  ·  " + incident.connectors + " connectors";
    const time = document.createElement("small");
    time.textContent = formatShort(incident.start_time);
    button.append(title, stats, time);
    button.addEventListener("click", () => selectIncident(incident.id));
    host.appendChild(button);
  });
}

async function selectMemoryMatch(match, requestId) {
  state.selectedMemory = match.entry_id;
  for (const button of byId("memory-matches").querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.entryId === match.entry_id);
  }
  try {
    const params = new URLSearchParams({ entry_id: match.entry_id });
    const graph = await getJson("/api/memory/" + state.active + "/graph?" + params);
    if (requestId !== state.memoryRequestId || state.selectedMemory !== match.entry_id) return;
    byId("memory-graph-title").textContent = match.entry_id + " · verified " + match.label;
    byId("memory-graph-size").textContent = graph.nodes.length + " nodes · " + graph.edges.length + " events";
    drawGraph("memory-graph", graph, "memory");
    if (graph.truncated) byId("memory-status").textContent += " Reference display is truncated.";
  } catch (error) {
    if (requestId === state.memoryRequestId) byId("memory-status").textContent = error.message;
  }
}

async function loadMemory(index) {
  if (!state.summary.memory_enabled) return;
  const requestId = ++state.memoryRequestId;
  byId("memory-status").textContent = "Loading verified examples…";
  byId("memory-matches").replaceChildren();
  byId("memory-graph").replaceChildren();
  byId("memory-graph-title").textContent = "Choose an example";
  byId("memory-graph-size").textContent = "—";
  try {
    const result = await getJson("/api/memory/" + index);
    if (requestId !== state.memoryRequestId) return;
    byId("memory-threshold").textContent = result.threshold === null ?
      "Threshold not calibrated" : "Similarity threshold ≥ " + result.threshold;
    byId("memory-status").textContent = !result.matches.length ?
      "No verified references are available." : result.no_strong_match ?
      "No reliable memory match. Showing the closest example below threshold for context." :
      result.threshold === null ? "Closest verified examples; no threshold decision." :
      "Verified examples above the similarity threshold.";
    const host = byId("memory-matches");
    result.matches.forEach(match => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.entryId = match.entry_id;
      button.className = "memory-match";
      const title = document.createElement("strong");
      title.textContent = match.entry_id + " · " + match.label;
      const detail = document.createElement("small");
      const status = match.above_threshold === false ? "below threshold · context only" :
        match.above_threshold === null ? "threshold not calibrated" : "above threshold";
      detail.textContent = "Similarity " + match.similarity.toFixed(3) + " · " + status;
      button.append(title, detail);
      button.addEventListener("click", () => selectMemoryMatch(match, requestId));
      host.appendChild(button);
    });
    if (result.matches.length) selectMemoryMatch(result.matches[0], requestId);
  } catch (error) {
    if (requestId === state.memoryRequestId) byId("memory-status").textContent = error.message;
  }
}

function filteredIncidentGraph(graph) {
  const [start, end] = selectedRange().map(BigInt);
  const edges = graph.edges.filter(edge => {
    const t = BigInt(edge.time);
    return start <= t && t <= end;
  });
  const ids = new Set(edges.flatMap(edge => [edge.source, edge.target]));
  return { nodes: graph.nodes.filter(node => ids.has(node.id)), edges, truncated: graph.truncated };
}

async function refreshViews() {
  if (state.active < 0) return;
  const requestId = ++state.requestId;
  setStatus("Loading bounded provenance…");
  const [start, end] = selectedRange();
  const params = new URLSearchParams({ hops: byId("hops").value, start, end });
  if (state.focus !== null) params.set("focus", state.focus);
  try {
    const [incident, original] = await Promise.all([
      getJson("/api/incident/" + state.active),
      getJson("/api/original/" + state.active + "?" + params.toString()),
    ]);
    if (requestId !== state.requestId) return;
    const filtered = filteredIncidentGraph(incident);
    drawGraph("incident-graph", filtered, "incident");
    drawGraph("original-graph", original, "original");
    byId("incident-graph-size").textContent = filtered.nodes.length + " nodes · " + filtered.edges.length + " events";
    byId("original-graph-size").textContent = original.nodes.length + " nodes · " + original.edges.length + " events";
    if (incident.truncated || original.truncated) {
      setStatus("Display limit reached. Narrow the time window or focus a node to see more.", "warning");
    } else {
      setStatus("Click an event to compare its role and original evidence. Loss is an anomaly score, not a probability.");
    }
  } catch (error) {
    if (requestId === state.requestId) setStatus(error.message, "error");
  }
}

function selectIncident(index) {
  state.active = index;
  state.focus = null;
  state.selectedEdge = null;
  state.selectedNode = null;
  byId("focus-node").value = "";
  byId("time-from").value = "0";
  byId("time-to").value = "100";
  const incident = state.summary.incidents[index];
  state.bounds = [incident.view_start_time, incident.view_end_time];
  byId("case-title").textContent = "Incident " + String(index + 1).padStart(2, "0");
  byId("case-subtitle").textContent = formatShort(incident.start_time) + " to " + formatShort(incident.end_time) + " · unverified candidate";
  byId("seed-count").textContent = incident.seeds;
  byId("connector-count").textContent = incident.connectors;
  byId("context-count").textContent = incident.context;
  byId("detail-title").textContent = "Select an event or node";
  byId("detail-role").textContent = "—";
  byId("detail-body").textContent = "Click an edge in either graph to inspect its score and provenance identity. Drag nodes to rearrange; scroll to zoom.";
  updateTimeLabel();
  renderIncidentList();
  refreshViews();
  loadMemory(index);
}

function scheduleRefresh() {
  clearTimeout(state.timer);
  updateTimeLabel();
  state.timer = setTimeout(refreshViews, 150);
}

async function start() {
  try {
    state.summary = await getJson("/api/summary");
    byId("threshold-badge").textContent = "Seed threshold ≥ " + state.summary.threshold;
    byId("incident-count").textContent = state.summary.incidents.length;
    byId("memory-panel").hidden = !state.summary.memory_enabled;
    if (!state.summary.incidents.length) {
      setStatus("No seed incidents were produced for this region.");
      return;
    }
    selectIncident(0);
  } catch (error) {
    setStatus(error.message, "error");
  }
}

byId("time-from").addEventListener("input", () => {
  if (+byId("time-from").value > +byId("time-to").value) byId("time-to").value = byId("time-from").value;
  scheduleRefresh();
});
byId("time-to").addEventListener("input", () => {
  if (+byId("time-to").value < +byId("time-from").value) byId("time-from").value = byId("time-to").value;
  scheduleRefresh();
});
byId("hops").addEventListener("change", refreshViews);
byId("focus-button").addEventListener("click", () => {
  const value = byId("focus-node").value.trim();
  if (!/^-?\d+$/.test(value)) { setStatus("Enter a numeric node ID to focus.", "error"); return; }
  state.focus = value;
  refreshViews();
});
byId("clear-focus").addEventListener("click", () => {
  state.focus = null;
  byId("focus-node").value = "";
  refreshViews();
});

start();
