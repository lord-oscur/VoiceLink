// Left panel: stored conversations, labels, project folders.
import { icon } from "./icons.js";
import { $, esc, toast } from "./ui.js";

let items = [];
let projects = [];
let activeId = null;
let activeLabels = [];
let activeProject = null;
let hooks = {};
let labelFilter = new Set();

export function initHistory(h) {
  hooks = h;
  $("convSearch").oninput = render;
  $("newChatSide").onclick = () => hooks.newChat();
  $("convList").onclick = onListClick;
  $("projectFilter").onchange = render;
  $("projectNew").onclick = newProject;
  $("labelFilters").onclick = onLabelFilterClick;
  $("incognito").onchange = () => {
    hooks.incognito($("incognito").checked);
    renderChatChrome();
  };
  $("chatProject").onchange = () => {
    activeProject = $("chatProject").value || null;
    sendMeta();
    const it = items.find((c) => c.id === activeId);
    if (it) it.project_id = activeProject;
    render();
  };
  $("chatLabelAdd").onclick = () => addLabel();
  $("chatLabels").onclick = (e) => {
    const chip = e.target.closest("[data-label]");
    if (!chip) return;
    activeLabels = activeLabels.filter((x) => x !== chip.dataset.label);
    sendMeta();
    const it = items.find((c) => c.id === activeId);
    if (it) it.labels = activeLabels.slice();
    render();
    renderChatChrome();
  };
  refresh();
}

export async function refresh() {
  try {
    [items, projects] = await Promise.all([
      (await fetch("/api/conversations")).json(),
      (await fetch("/api/projects")).json(),
    ]);
  } catch (_) {
    return;
  }
  render();
  renderChatChrome();
}

export function setActive(id, meta) {
  activeId = id;
  if (meta) {
    activeLabels = (meta.labels || []).slice();
    activeProject = meta.project_id || null;
  } else if (!id) {
    activeLabels = [];
    activeProject = null;
  }
  if (id) {
    $("incognito").checked = false;
    hooks.incognito(false);
  }
  render();
  renderChatChrome();
}

let timer = null;
export function saved(id, title, meta) {
  activeId = id;
  if (meta) {
    if (meta.labels) activeLabels = meta.labels.slice();
    if ("project_id" in meta) activeProject = meta.project_id || null;
  }
  const it = items.find((c) => c.id === id);
  if (it) {
    if (title) it.title = title;
    it.labels = activeLabels.slice();
    it.project_id = activeProject;
  }
  render();
  renderChatChrome();
  clearTimeout(timer);
  timer = setTimeout(refresh, 700);
}

function projectName(pid) {
  return (projects.find((p) => p.id === pid) || {}).name || "";
}

function allLabels() {
  const set = new Set();
  for (const c of items) for (const l of c.labels || []) set.add(l);
  for (const l of activeLabels) set.add(l);
  return [...set].sort((a, b) => a.localeCompare(b));
}

function fillProjectSelect(sel, value, emptyLabel) {
  sel.innerHTML = "";
  sel.append(new Option(emptyLabel, ""));
  for (const p of projects) sel.append(new Option(p.name, p.id));
  sel.value = value || "";
}

function render() {
  fillProjectSelect($("projectFilter"), $("projectFilter").value, "All projects");
  const labels = allLabels();
  $("labelFilters").innerHTML = labels.map((l) =>
    `<button type="button" class="label-chip${labelFilter.has(l) ? " active" : ""}" data-filter="${esc(l)}">${esc(l)}</button>`
  ).join("");

  const q = $("convSearch").value.trim().toLowerCase();
  const proj = $("projectFilter").value;
  let list = items.filter((c) => {
    if (q && !(c.title + " " + c.preview + " " + (c.labels || []).join(" ")).toLowerCase().includes(q)) return false;
    if (proj && c.project_id !== proj) return false;
    if (labelFilter.size && !(c.labels || []).some((l) => labelFilter.has(l))) return false;
    return true;
  });

  const box = $("convList");
  if (!list.length) {
    box.innerHTML = `<p class="empty">${q || proj || labelFilter.size
      ? "No conversation matches."
      : "No conversations yet. Talk or type to start one."}</p>`;
    return;
  }
  let html = "", last = null;
  for (const c of list) {
    const b = bucket(c.updated || c.created);
    if (b !== last) { html += `<div class="conv-group">${esc(b)}</div>`; last = b; }
    const time = new Date(c.updated).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
    const pname = projectName(c.project_id);
    html += `<div class="conv${c.id === activeId ? " active" : ""}" data-id="${esc(c.id)}" title="${esc(c.preview)}">
      <div class="conv-main"><span class="conv-title">${esc(c.title)}</span>
      <span class="conv-meta">${pname ? `<span class="conv-proj">${esc(pname)}</span> · ` : ""}${time} · ${c.count} msg</span>
      ${(c.labels || []).length ? `<span class="conv-labels">${c.labels.map((l) => `<span>${esc(l)}</span>`).join("")}</span>` : ""}</div>
      <button class="icon-btn small" data-act="label" title="Add label">${icon("tag")}</button>
      <button class="icon-btn small" data-act="move" title="Move to project">${icon("folder")}</button>
      <button class="icon-btn small" data-act="rename" title="Rename">${icon("pencil")}</button>
      <button class="icon-btn small" data-act="delete" title="Delete">${icon("trash")}</button></div>`;
  }
  box.innerHTML = html;
}

function renderChatChrome() {
  const incog = $("incognito").checked;
  $("incognitoBanner").hidden = !incog;
  $("chatMeta").hidden = incog;
  if (incog) return;
  fillProjectSelect($("chatProject"), activeProject, "No project");
  $("chatLabels").innerHTML = activeLabels.map((l) =>
    `<button type="button" class="label-chip" data-label="${esc(l)}">${esc(l)} ${icon("x")}</button>`
  ).join("");
}

function sendMeta() {
  hooks.meta({ labels: activeLabels, project_id: activeProject });
}

function onLabelFilterClick(e) {
  const b = e.target.closest("[data-filter]");
  if (!b) return;
  const l = b.dataset.filter;
  if (labelFilter.has(l)) labelFilter.delete(l);
  else labelFilter.add(l);
  render();
}

async function newProject() {
  const name = prompt("New project folder name:", "");
  if (!name || !name.trim()) return;
  const r = await fetch("/api/projects", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!r.ok) { toast(await r.text() || "Could not create the project", "error"); return; }
  const p = await r.json();
  projects.push(p);
  $("projectFilter").value = p.id;
  toast(`Project “${p.name}” created`);
  render();
  renderChatChrome();
}

function bucket(iso) {
  const d = new Date(iso);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const days = (today - new Date(d.getFullYear(), d.getMonth(), d.getDate())) / 86400000;
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return "Previous 7 days";
  if (days < 30) return "Previous 30 days";
  return d.toLocaleDateString(undefined, { year: "numeric", month: "long" });
}

async function patchConv(id, body) {
  const r = await fetch(`/api/conversations/${id}`, {
    method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) { toast("Could not update the conversation", "error"); return null; }
  return r.json();
}

async function onListClick(e) {
  const row = e.target.closest(".conv");
  if (!row) return;
  const id = row.dataset.id;
  const act = e.target.closest("button")?.dataset.act;
  const conv = items.find((c) => c.id === id);
  if (act === "rename") {
    const title = prompt("Rename conversation:", conv?.title || "");
    if (!title || !title.trim()) return;
    const d = await patchConv(id, { title });
    if (d && conv) { conv.title = d.title; render(); }
  } else if (act === "delete") {
    if (!confirm(`Delete "${conv?.title || "this conversation"}"? This cannot be undone.`)) return;
    const r = await fetch(`/api/conversations/${id}`, { method: "DELETE" });
    if (!r.ok) { toast("Could not delete the conversation", "error"); return; }
    items = items.filter((c) => c.id !== id);
    if (id === activeId) hooks.newChat();
    render();
  } else if (act === "label") {
    const name = prompt("Add a label:", "");
    if (!name || !name.trim()) return;
    const labels = [...new Set([...(conv.labels || []), name.trim()])];
    const d = await patchConv(id, { labels });
    if (d) {
      conv.labels = d.labels;
      if (id === activeId) activeLabels = d.labels.slice();
      render();
      renderChatChrome();
    }
  } else if (act === "move") {
    const names = projects.map((p) => p.name);
    const choice = prompt("Move to project (blank = none):\n" + (names.join(", ") || "(no projects yet — create one first)"),
                          projectName(conv.project_id));
    if (choice == null) return;
    const p = projects.find((x) => x.name.toLowerCase() === choice.trim().toLowerCase());
    const project_id = choice.trim() ? (p ? p.id : null) : null;
    if (choice.trim() && !p) { toast("No project with that name", "error"); return; }
    const d = await patchConv(id, { project_id });
    if (d) {
      conv.project_id = d.project_id;
      if (id === activeId) activeProject = d.project_id;
      render();
      renderChatChrome();
    }
  } else if (id !== activeId) {
    hooks.open(id);
  }
}

async function addLabel() {
  const name = prompt("Add a label to this chat:", "");
  if (!name || !name.trim()) return;
  if (!activeLabels.some((x) => x.toLowerCase() === name.trim().toLowerCase())) {
    activeLabels = [...activeLabels, name.trim()];
  }
  sendMeta();
  const it = items.find((c) => c.id === activeId);
  if (it) it.labels = activeLabels.slice();
  render();
  renderChatChrome();
}
