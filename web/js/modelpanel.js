// Right panel: the LM Studio model (switch / reload) and per-request generation settings.
import { settings } from "./settings.js";
import { $, capsHTML, esc, fmtBytes, fmtK, toast } from "./ui.js";

// [settings key, server default key, label, min, max, step, hint]
const GEN_FIELDS = [
  ["temperature", "temperature", "Temperature", 0, 2, 0.05, "Randomness. Lower = more focused."],
  ["top_p", "top_p", "Top P", 0, 1, 0.01, "Nucleus sampling cut-off."],
  ["top_k", "top_k", "Top K", 0, 500, 1, "Sample from the K most likely tokens."],
  ["repeat_penalty", "repeat_penalty", "Repeat penalty", 0.5, 2, 0.01, "Above 1 discourages repetition."],
  ["max_tokens", "max_tokens", "Max reply tokens", 16, 32768, 1, "Longest reply, in tokens."],
  ["history", "history_messages", "Messages remembered", 2, 400, 1, "How many past messages are sent to the model."],
];

let defaults = {};
let models = [];
let status = null;
let selected = null;   // model key chosen in the list
let loading = false;
let hooks = {};

export function initModelPanel(h) {
  hooks = h;
  buildGenFields();
  $("modelSearch").oninput = renderList;
  $("modelRefresh").onclick = loadModels;
  $("modelList").onclick = (e) => {
    const row = e.target.closest(".model-row");
    if (!row) return;
    selected = row.dataset.key;
    renderList();
    renderLoad();
  };
  $("loadBtn").onclick = doLoad;
  $("sysPrompt").onchange = () => settings.set("model.systemPrompt", $("sysPrompt").value.trim());
  $("sysDefault").onclick = () => { settings.set("model.systemPrompt", ""); renderGen(); };
  $("thinking").onchange = () => settings.set("model.thinking", $("thinking").checked);
  $("genReset").onclick = () => {
    for (const [key] of GEN_FIELDS) settings.set("model." + key, null);
    settings.set("model.thinking", false);
  };
  settings.onChange((paths) => { if (paths.includes("*") || paths.some((p) => p.startsWith("model"))) renderGen(); });
  fetch("/api/llm_defaults").then((r) => r.json()).then((d) => { defaults = d; renderGen(); }).catch(() => {});
  renderGen();
}

export function setStatus(llm) {
  status = llm;
  renderCurrent();
  if (!models.length && llm.ok) loadModels();
}

export function panelOpened() {
  if (!models.length) loadModels();
}

function currentInfo() {
  return status && status.ok ? status.info : null;
}

function renderCurrent() {
  const info = currentInfo();
  const box = $("modelCard");
  if (!status) { box.innerHTML = `<p class="hint">Connecting…</p>`; return; }
  if (!status.ok) {
    box.innerHTML = `<p class="bad-text">LM Studio is not reachable at ${esc(status.base_url)}.</p>
      <p class="hint">Start LM Studio's server (Developer tab). ${esc(status.error || "")}</p>`;
    return;
  }
  const inst = info && info.instances && info.instances[0];
  box.innerHTML = `<div class="model-name">${esc(info ? info.name : status.model)}</div>
    <div class="model-key">${esc(status.model)}</div>
    <div class="caps">${capsHTML(info)}</div>
    ${info ? `<div class="model-meta">${[info.params, info.quant, fmtBytes(info.size),
      inst && inst.flash_attention != null ? "flash attention " + (inst.flash_attention ? "on" : "off") : ""]
      .filter(Boolean).map(esc).join(" · ")}</div>` : ""}`;
  $("thinking").disabled = !(info && info.caps && info.caps.reasoning);
  $("thinkingRow").title = $("thinking").disabled ? "The loaded model has no reasoning mode" : "";
  if (!selected && info) selected = info.key;
  renderLoad();
  renderList();
}

async function loadModels() {
  $("modelList").innerHTML = `<p class="hint">Loading the model list…</p>`;
  try {
    const d = await (await fetch("/api/models")).json();
    if (!d.ok) throw new Error(d.error);
    models = d.models.sort((a, b) => (b.loaded - a.loaded) || a.name.localeCompare(b.name));
  } catch (e) {
    $("modelList").innerHTML = `<p class="bad-text">Could not list models: ${esc(e.message)}</p>`;
    return;
  }
  renderList();
  renderLoad();
}

function renderList() {
  const q = $("modelSearch").value.trim().toLowerCase();
  const list = models.filter((m) => !q || (m.name + " " + m.key).toLowerCase().includes(q));
  $("modelCount").textContent = models.length ? `${models.length} downloaded` : "";
  $("modelList").innerHTML = list.map((m) => `
    <div class="model-row${m.key === selected ? " selected" : ""}${m.loaded ? " loaded" : ""}" data-key="${esc(m.key)}">
      <div class="row"><span class="grow model-row-name">${m.loaded ? '<span class="dot on"></span>' : ""}${esc(m.name)}</span>
        <span class="caps">${capsHTML(m, { context: false })}</span></div>
      <div class="model-meta">${[m.params, m.quant, fmtBytes(m.size), m.max_context ? fmtK(m.max_context) + " ctx max" : ""]
        .filter(Boolean).map(esc).join(" · ")}</div>
    </div>`).join("") || `<p class="hint">No model matches.</p>`;
}

function renderLoad() {
  const m = models.find((x) => x.key === selected) || currentInfo();
  const btn = $("loadBtn");
  if (!m) { $("loadTarget").textContent = "—"; btn.disabled = true; return; }
  const inst = m.instances && m.instances[0];
  $("loadTarget").textContent = m.name;
  const ctx = $("ctxLen");
  ctx.max = m.max_context || 262144;
  ctx.placeholder = inst && inst.context_length ? String(inst.context_length) : String(Math.min(m.max_context || 8192, 8192));
  if (ctx.dataset.for !== m.key) {
    ctx.value = inst && inst.context_length ? inst.context_length : Math.min(m.max_context || 8192, 8192);
    $("flashAttn").checked = inst ? inst.flash_attention !== false : true;
    ctx.dataset.for = m.key;
  }
  $("ctxMax").textContent = m.max_context ? `max ${m.max_context.toLocaleString()}` : "";
  btn.disabled = loading;
  btn.textContent = loading ? "Loading…" : m.loaded ? "Reload with these settings" : "Load this model";
}

async function doLoad() {
  const m = models.find((x) => x.key === selected) || currentInfo();
  if (!m || loading) return;
  const context_length = parseInt($("ctxLen").value, 10) || null;
  if (!confirm(`Load "${m.name}" with a ${context_length ? context_length.toLocaleString() : "default"}-token context?\n` +
               "Other loaded chat models are unloaded first to free VRAM.")) return;
  loading = true;
  renderLoad();
  toast(`Loading ${m.name}… this can take a minute`, "info", 6000);
  try {
    const r = await fetch("/api/models/load", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: m.key, context_length, flash_attention: $("flashAttn").checked }),
    });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || `HTTP ${r.status}`);
    toast(`${m.name} is loaded`);
    $("ctxLen").dataset.for = "";
    hooks.statusChanged && hooks.statusChanged(d.status);
  } catch (e) {
    toast("Load failed: " + e.message, "error", 8000);
  } finally {
    loading = false;
    await loadModels();
  }
}

function buildGenFields() {
  const box = $("genFields");
  for (const [key, , label, min, max, step, hint] of GEN_FIELDS) {
    const wrap = document.createElement("label");
    wrap.className = "gen-field";
    wrap.title = hint;
    wrap.innerHTML = `<span>${label}</span><input type="number" min="${min}" max="${max}" step="${step}" data-key="${key}">`;
    const input = wrap.querySelector("input");
    input.onchange = () => {
      const v = input.value === "" ? null : Math.min(max, Math.max(min, parseFloat(input.value)));
      settings.set("model." + key, Number.isFinite(v) ? v : null);
    };
    box.append(wrap);
  }
}

function renderGen() {
  const m = settings.get("model");
  for (const [key, defKey] of GEN_FIELDS) {
    const input = document.querySelector(`#genFields input[data-key="${key}"]`);
    input.value = m[key] ?? "";
    const d = defaults[defKey];
    input.placeholder = d != null ? `${d} (default)` : "model default";
  }
  const sp = $("sysPrompt");
  if (document.activeElement !== sp) sp.value = m.systemPrompt || "";
  sp.placeholder = defaults.system_prompt ? "Default: " + defaults.system_prompt : "System prompt";
  $("sysState").textContent = m.systemPrompt ? "custom" : "default";
  $("thinking").checked = !!m.thinking;
}
