// Left panel: stored conversations (data/conversations on the server).
import { icon } from "./icons.js";
import { $, esc, toast } from "./ui.js";

let items = [];
let activeId = null;
let hooks = {};

export function initHistory(h) {
  hooks = h;
  $("convSearch").oninput = render;
  $("newChatSide").onclick = () => hooks.newChat();
  $("convList").onclick = onListClick;
  refresh();
}

export async function refresh() {
  try {
    items = await (await fetch("/api/conversations")).json();
  } catch (_) {
    return;
  }
  render();
}

export function setActive(id) {
  activeId = id;
  render();
}

// The server saved the current conversation: show it and refresh the list shortly.
let timer = null;
export function saved(id, title) {
  activeId = id;
  const it = items.find((c) => c.id === id);
  if (it && title) it.title = title;
  render();
  clearTimeout(timer);
  timer = setTimeout(refresh, 700);
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

function render() {
  const q = $("convSearch").value.trim().toLowerCase();
  const list = q ? items.filter((c) => (c.title + " " + c.preview).toLowerCase().includes(q)) : items;
  const box = $("convList");
  if (!list.length) {
    box.innerHTML = `<p class="empty">${q ? "No conversation matches." : "No conversations yet. Talk or type to start one."}</p>`;
    return;
  }
  let html = "", last = null;
  for (const c of list) {
    const b = bucket(c.updated || c.created);
    if (b !== last) { html += `<div class="conv-group">${esc(b)}</div>`; last = b; }
    const time = new Date(c.updated).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
    html += `<div class="conv${c.id === activeId ? " active" : ""}" data-id="${esc(c.id)}" title="${esc(c.preview)}">
      <div class="conv-main"><span class="conv-title">${esc(c.title)}</span>
      <span class="conv-meta">${time} · ${c.count} msg</span></div>
      <button class="icon-btn small" data-act="rename" title="Rename">${icon("pencil")}</button>
      <button class="icon-btn small" data-act="delete" title="Delete">${icon("trash")}</button></div>`;
  }
  box.innerHTML = html;
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
    const r = await fetch(`/api/conversations/${id}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title }),
    });
    if (r.ok && conv) { conv.title = (await r.json()).title; render(); }
  } else if (act === "delete") {
    if (!confirm(`Delete "${conv?.title || "this conversation"}"? This cannot be undone.`)) return;
    const r = await fetch(`/api/conversations/${id}`, { method: "DELETE" });
    if (!r.ok) { toast("Could not delete the conversation", "error"); return; }
    items = items.filter((c) => c.id !== id);
    if (id === activeId) hooks.newChat();
    render();
  } else if (id !== activeId) {
    hooks.open(id);
  }
}
