// Right-panel model page + centered picker (filters, sort, load settings).
import { icon } from "./icons.js";
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
let selected = null;
let loading = false;
let hooks = {};
const capOn = new Set();
let hardware = { vram_gb: null, gpu: false };
let hfTimer = null;
let hfToken = 0;
const hfJobs = new Map();

export function initModelPanel(h) {
  hooks = h;
  buildGenFields();
  $("modelSearch").oninput = renderList;
  $("hfIgnoreVram").onchange = () => {
    settings.set("model.hfIgnoreVram", $("hfIgnoreVram").checked);
    if (!$("hfBox").hidden) runHfSearch();
  };
  $("hfVramGb").onchange = () => {
    const n = parseFloat($("hfVramGb").value);
    settings.set("model.hfVramGb", Number.isFinite(n) && n > 0 ? n : null);
    if (!$("hfBox").hidden) runHfSearch();
  };
  $("hfList").onclick = onHfClick;
  fetch("/api/hardware").then((r) => r.json()).then((d) => {
    hardware = d;
    const saved = settings.get("model.hfVramGb");
    $("hfVramGb").placeholder = d.vram_gb ? String(d.vram_gb) : "auto";
    if (saved) $("hfVramGb").value = saved;
    $("hfIgnoreVram").checked = !!settings.get("model.hfIgnoreVram");
  }).catch(() => {});
  $("modelRefresh").onclick = loadModels;
  $("modelPickBtn").onclick = (e) => { e.stopPropagation(); openPicker(); };
  $("modelModalClose").onclick = closePicker;
  $("modelModal").onclick = (e) => { if (e.target.id === "modelModal") closePicker(); };
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("modelModal").hidden) { e.stopPropagation(); closePicker(); }
  });
  $("modelList").onclick = (e) => {
    const row = e.target.closest(".model-row");
    if (!row) return;
    selected = row.dataset.key;
    renderList();
    renderLoad();
  };
  $("modelCard").onclick = (e) => {
    if (e.target.closest("[data-act=pick]")) openPicker();
  };
  for (const b of document.querySelectorAll("#capFilter button")) {
    b.onclick = () => {
      if (capOn.has(b.dataset.cap)) capOn.delete(b.dataset.cap);
      else capOn.add(b.dataset.cap);
      b.classList.toggle("active", capOn.has(b.dataset.cap));
      renderList();
    };
  }
  for (const id of ["kindFilter", "archFilter", "quantFilter", "modelSort"]) {
    $(id).onchange = renderList;
  }
  $("rateNsfw").onchange = $("rateAblit").onchange = renderList;

  for (const b of document.querySelectorAll("#attnSeg button")) {
    b.onclick = () => {
      settings.set("model.attention", b.dataset.v);
      syncKv();
    };
  }
  $("kvGpu").onchange = () => settings.set("model.kvGpu", $("kvGpu").checked);
  $("kCache").onchange = () => settings.set("model.kCache", $("kCache").value);
  $("vCache").onchange = () => settings.set("model.vCache", $("vCache").value);

  $("loadBtn").onclick = doLoad;
  $("sysPrompt").onchange = () => settings.set("model.systemPrompt", $("sysPrompt").value.trim());
  $("sysDefault").onclick = () => { settings.set("model.systemPrompt", ""); renderGen(); };
  $("thinking").onchange = () => settings.set("model.thinking", $("thinking").checked);
  $("genReset").onclick = () => {
    for (const [key] of GEN_FIELDS) settings.set("model." + key, null);
    settings.set("model.thinking", false);
  };
  $("promptLib").onchange = applySavedPrompt;
  $("promptSave").onclick = savePrompt;
  $("promptUpdate").onclick = updatePrompt;
  $("promptDel").onclick = deletePrompt;

  settings.onChange((paths) => {
    if (paths.includes("*") || paths.some((p) => p.startsWith("model"))) {
      renderGen();
      syncKv();
    }
  });
  fetch("/api/llm_defaults").then((r) => r.json()).then((d) => { defaults = d; renderGen(); }).catch(() => {});
  renderGen();
  syncKv();
}

export function setStatus(llm) {
  status = llm;
  renderCurrent();
  if (!models.length && llm.ok) loadModels();
}

export function panelOpened() {
  if (!models.length) loadModels();
}

export function openPicker() {
  $("modelModal").hidden = false;
  if (!models.length) loadModels();
  else renderList();
  $("modelSearch").focus();
}

function closePicker() {
  $("modelModal").hidden = true;
}

function currentInfo() {
  return status && status.ok ? status.info : null;
}

function rateHTML(m) {
  const out = [];
  if (isNsfw(m)) out.push(`<span class="cap nsfw" title="Name contains uncensored or NSFW">+18</span>`);
  if (isAblit(m)) out.push(`<span class="cap ablit" title="Name contains Abliterated">${icon("unlink")} Not Filtered</span>`);
  return out.join("");
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
  const extraAttn = inst && inst.attention && inst.attention.filter((a) => a !== "flash" && a !== "standard");
  box.innerHTML = `<div class="model-card-head">
      <div class="model-name">${esc(info ? info.name : status.model)}</div>
      <button type="button" class="icon-btn small" data-act="pick" title="Change model">${icon("layers")}</button>
    </div>
    <div class="model-key">${esc(status.model)}</div>
    <div class="caps">${capsHTML(info)} ${rateHTML(info || {})}</div>
    ${info ? `<div class="model-meta">${[info.params, info.quant, fmtBytes(info.size),
      info.architecture, info.kind && info.kind !== "dense" ? info.kind.toUpperCase() : "",
      inst && inst.flash_attention != null ? "flash attention " + (inst.flash_attention ? "on" : "off") : "",
      extraAttn && extraAttn.length ? extraAttn.join(" · ") : ""]
      .filter(Boolean).map(esc).join(" · ")}</div>` : ""}`;
  $("thinking").disabled = !(info && info.caps && info.caps.reasoning);
  $("thinkingRow").title = $("thinking").disabled ? "The loaded model has no reasoning mode" : "";
  if (!selected && info) selected = info.key;
  renderLoad();
  renderList();
}

async function loadModels() {
  const btn = $("modelRefresh");
  if (btn) btn.classList.add("busy");
  if (!models.length) {
    $("modelList").innerHTML = `<p class="hint">Loading the model list…</p>`;
  }
  try {
    const d = await (await fetch("/api/models")).json();
    if (!d.ok) throw new Error(d.error);
    models = d.models;
  } catch (e) {
    $("modelList").innerHTML = `<p class="bad-text">Could not list models: ${esc(e.message)}</p>`;
    return;
  } finally {
    if (btn) btn.classList.remove("busy");
  }
  fillFilterSelects();
  renderList();
  renderLoad();
}

function blobOf(m) {
  return [m.name, m.key, m.id, m.repo, m.architecture, m.quant, m.params]
    .filter((x) => x != null && x !== "")
    .join(" ")
    .toLowerCase();
}

function familiesOf(m) {
  if (Array.isArray(m.families) && m.families.length) return m.families;
  const blob = blobOf(m);
  const arch = (m.architecture || "").toLowerCase();
  const out = [];
  if (/ternary|bitnet|\bbonsai\b/.test(blob)) out.push("Ternary");
  if (/\bopus\b/.test(blob)) out.push("Opus");
  const pairs = [
    ["qwen", "Qwen"], ["llama", "Llama"], ["gemma", "Gemma"], ["phi", "Phi"],
    ["lfm", "LFM"], ["granite", "Granite"], ["nemotron", "Nemotron"],
    ["gpt", "GPT"], ["mistral", "Mistral"], ["mixtral", "Mistral"],
  ];
  let hit = false;
  for (const [prefix, fam] of pairs) {
    if (arch.startsWith(prefix) || new RegExp("\\b" + prefix).test(blob)) {
      if (!out.includes(fam)) out.push(fam);
      hit = true;
      break;
    }
  }
  if (!hit && arch) {
    const pretty = arch.replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    if (pretty && !out.includes(pretty)) out.push(pretty);
  }
  return out;
}

function isMoe(m) {
  if (m.moe) return true;
  const blob = blobOf(m);
  const arch = (m.architecture || "").toLowerCase();
  if (arch.includes("moe")) return true;
  if ((m.experts || 0) > 1) return true;
  return /\bmoe\b|mixtral|a3b|a22b|\be[24]b\b/.test(blob);
}

function isMtp(m) {
  return !!(m.mtp || /\bmtp\b/.test(blobOf(m)));
}

function isNsfw(m) {
  if (m.nsfw) return true;
  const blob = blobOf(m);
  return /uncensored|nsfw|\+?18\+|erotica/.test(blob);
}

function isAblit(m) {
  if (m.abliterated) return true;
  return blobOf(m).includes("abliterated");
}

function fillFilterSelects() {
  const fill = (id, values, allLabel) => {
    const sel = $(id);
    const keep = sel.value;
    const uniq = [...new Set(values.filter(Boolean))].sort((a, b) => String(a).localeCompare(String(b)));
    sel.innerHTML = `<option value="">${allLabel}</option>` +
      uniq.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
    if (keep && uniq.includes(keep)) sel.value = keep;
  };
  fill("archFilter", models.flatMap((m) => familiesOf(m)), "All");
  fill("quantFilter", models.map((m) => m.quant), "All");
}

function parseParams(s) {
  if (s == null || s === "") return 0;
  if (typeof s === "number") return s;
  const m = String(s).replace(/,/g, "").match(/([\d.]+)\s*([TtMmBb])?/);
  if (!m) return 0;
  const n = parseFloat(m[1]);
  const u = (m[2] || "b").toLowerCase();
  return n * ({ t: 1e12, b: 1e9, m: 1e6 }[u] || 1);
}

function filteredModels() {
  const q = $("modelSearch").value.trim().toLowerCase();
  const kind = $("kindFilter").value;
  const arch = $("archFilter").value;
  const quant = $("quantFilter").value;
  const nsfw = $("rateNsfw").checked;
  const ablit = $("rateAblit").checked;
  const sort = $("modelSort").value;
  let list = models.filter((m) => {
    if (q && !blobOf(m).includes(q)) return false;
    if (capOn.has("reasoning") && !(m.caps && m.caps.reasoning)) return false;
    if (capOn.has("tools") && !(m.caps && m.caps.tools)) return false;
    if (capOn.has("vision") && !(m.caps && m.caps.vision)) return false;
    if (kind === "dense" && (isMoe(m) || isMtp(m))) return false;
    if (kind === "moe" && !isMoe(m)) return false;
    if (kind === "mtp" && !isMtp(m)) return false;
    if (arch && !familiesOf(m).includes(arch)) return false;
    if (quant && m.quant !== quant) return false;
    if (nsfw && !isNsfw(m)) return false;
    if (ablit && !isAblit(m)) return false;
    return true;
  });
  const cmp = {
    name: (a, b) => a.name.localeCompare(b.name),
    size: (a, b) => (b.size || 0) - (a.size || 0),
    context: (a, b) => (b.max_context || 0) - (a.max_context || 0),
    params: (a, b) => parseParams(b.params) - parseParams(a.params),
    experts: (a, b) => (b.experts || 0) - (a.experts || 0),
  }[sort] || ((a, b) => (b.loaded - a.loaded) || a.name.localeCompare(b.name));
  list.sort((a, b) => (b.loaded - a.loaded) || cmp(a, b));
  return list;
}

function renderList() {
  if (!$("modelList")) return;
  const list = filteredModels();
  const q = $("modelSearch").value.trim();
  $("modelCount").textContent = models.length
    ? `${list.length} of ${models.length} downloaded` : "";
  if (list.length) {
    $("hfBox").hidden = true;
    $("modelList").innerHTML = list.map((m) => `
    <div class="model-row${m.key === selected ? " selected" : ""}${m.loaded ? " loaded" : ""}" data-key="${esc(m.key)}">
      <div class="row"><span class="grow model-row-name">${m.loaded ? '<span class="dot on"></span>' : ""}${esc(m.name)}</span>
        <span class="caps">${capsHTML(m, { context: false })} ${rateHTML(m)}</span></div>
      <div class="model-meta">${[m.params, m.quant, fmtBytes(m.size),
        m.max_context ? fmtK(m.max_context) + " ctx max" : "",
        familiesOf(m).join(" · "), isMoe(m) && !isMtp(m) ? "MoE" : "", isMtp(m) ? "MTP" : "",
        m.experts ? m.experts + " experts" : ""]
        .filter(Boolean).map(esc).join(" · ")}</div>
    </div>`).join("");
    return;
  }
  $("modelList").innerHTML = q.length >= 2
    ? `<p class="hint">No downloaded model matches. Searching Hugging Face…</p>`
    : `<p class="hint">${q ? "No model matches." : "No models in the list."}</p>`;
  if (q.length >= 2) scheduleHfSearch();
  else $("hfBox").hidden = true;
}

function scheduleHfSearch() {
  clearTimeout(hfTimer);
  hfTimer = setTimeout(runHfSearch, 450);
}

function hfFilterRow(row) {
  const fake = { name: row.name, key: row.repo, repo: row.repo, architecture: "" };
  const kind = $("kindFilter").value;
  const arch = $("archFilter").value;
  if (capOn.size) return false;
  if (kind === "dense" && (isMoe(fake) || isMtp(fake))) return false;
  if (kind === "moe" && !isMoe(fake)) return false;
  if (kind === "mtp" && !isMtp(fake)) return false;
  if (arch && !familiesOf(fake).includes(arch)) return false;
  if ($("rateNsfw").checked && !isNsfw(fake)) return false;
  if ($("rateAblit").checked && !isAblit(fake)) return false;
  return true;
}

async function runHfSearch() {
  const q = $("modelSearch").value.trim();
  if (q.length < 2 || filteredModels().length) { $("hfBox").hidden = true; return; }
  const my = ++hfToken;
  $("hfBox").hidden = false;
  $("hfIgnoreVram").checked = !!settings.get("model.hfIgnoreVram");
  const vram = settings.get("model.hfVramGb") || hardware.vram_gb;
  if (document.activeElement !== $("hfVramGb") && settings.get("model.hfVramGb")) {
    $("hfVramGb").value = settings.get("model.hfVramGb");
  }
  $("hfVramHint").textContent = hardware.gpu || vram
    ? `Quants limited to about ${vram || "?"} GB VRAM (overridable).`
    : "No GPU reported — all GGUF quants are listed.";
  $("hfList").innerHTML = `<p class="hint">Searching Hugging Face for “${esc(q)}”…</p>`;
  const params = new URLSearchParams({ q });
  if (settings.get("model.hfIgnoreVram") || !vram) params.set("ignore", "1");
  if (vram) params.set("vram", String(vram));
  try {
    const d = await (await fetch("/api/models/hf?" + params)).json();
    if (my !== hfToken) return;
    if (!d.ok) throw new Error(d.error || "search failed");
    const rows = (d.results || []).filter(hfFilterRow);
    if (!rows.length) {
      $("hfList").innerHTML = `<p class="hint">Nothing on Hugging Face matched, or every GGUF is larger than the VRAM budget. Enable “Show quants that do not fit this GPU” to list them anyway.</p>`;
      return;
    }
    $("hfList").innerHTML = rows.map((r) => hfRowHTML(r)).join("");
  } catch (e) {
    if (my === hfToken) $("hfList").innerHTML = `<p class="bad-text">Hugging Face search failed: ${esc(e.message)}</p>`;
  }
}

function hfRowHTML(r) {
  const quants = r.quants || [];
  const opts = quants.map((q) =>
    `<option value="${esc(q.quant)}" ${q.quant === r.quant ? "selected" : ""}>${esc(q.quant)}${q.size ? " · " + fmtBytes(q.size) : ""}${q.fits === false ? " · large" : ""}</option>`
  ).join("");
  const job = hfJobs.get(r.repo);
  const pct = job && job.total ? Math.round(100 * job.got / job.total) : (job ? 1 : 0);
  return `<div class="model-row hf" data-repo="${esc(r.repo)}" data-name="${esc(r.name)}">
    <div class="row"><span class="grow model-row-name">${esc(r.name)}</span>
      <select class="hf-quant">${opts}</select>
      <button type="button" class="hf-dl" data-act="hf-dl" ${job ? "disabled" : ""}>${icon("download")} ${job ? "Downloading…" : "Download"}</button></div>
    <div class="model-meta">${[r.params ? (r.params / 1e9 >= 10 ? Math.round(r.params / 1e9) : +(r.params / 1e9).toFixed(1)) + "B" : "",
      r.quant, fmtBytes(r.size), r.fits === false ? "larger than VRAM budget" : "",
      r.downloads ? r.downloads.toLocaleString() + " downloads" : ""]
      .filter(Boolean).map(esc).join(" · ")}</div>
    ${job ? `<div class="hf-progress"><i style="width:${pct}%"></i></div>` : ""}
  </div>`;
}

function onHfClick(e) {
  const btn = e.target.closest("[data-act=hf-dl]");
  if (!btn) return;
  const row = btn.closest("[data-repo]");
  if (!row) return;
  const quant = row.querySelector(".hf-quant")?.value || "";
  startHfDownload(row.dataset.repo, quant, row.dataset.name || row.dataset.repo);
}

async function startHfDownload(repo, quant, name) {
  if (hfJobs.has(repo)) return;
  if (!confirm(`Download “${name}” (${quant || "default quant"}) with LM Studio?\n` +
               "You will get a notice here when it is ready to load.")) return;
  hfJobs.set(repo, { got: 0, total: 0 });
  runHfSearch();
  toast(`LM Studio is downloading ${name}…`, "info", 6000);
  try {
    const r = await fetch("/api/models/download", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo, quant }),
    });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || `HTTP ${r.status}`);
    if (d.status === "already_downloaded" || d.status === "completed") {
      hfJobs.delete(repo);
      toast(`${name} is already on disk and ready to use`, "info", 7000);
      await loadModels();
      return;
    }
    if (!d.job_id) throw new Error("LM Studio did not return a download job");
    await pollDownload(d.job_id, repo, name);
  } catch (e) {
    hfJobs.delete(repo);
    toast("Download failed: " + e.message, "error", 8000);
    runHfSearch();
  }
}

async function pollDownload(jobId, repo, name) {
  for (let i = 0; i < 2400; i++) {
    await new Promise((r) => setTimeout(r, 1500));
    let d;
    try {
      d = await (await fetch("/api/models/download/" + encodeURIComponent(jobId))).json();
    } catch (_) {
      continue;
    }
    if (d.downloaded_bytes != null) {
      hfJobs.set(repo, { got: d.downloaded_bytes, total: d.total_size_bytes || 0 });
      const bar = document.querySelector(`[data-repo="${CSS.escape(repo)}"] .hf-progress i`);
      if (bar && d.total_size_bytes) bar.style.width = Math.round(100 * d.downloaded_bytes / d.total_size_bytes) + "%";
    }
    if (d.status === "completed") {
      hfJobs.delete(repo);
      toast(`${name} is downloaded and ready to use`, "info", 9000);
      await loadModels();
      return;
    }
    if (d.status === "failed") {
      hfJobs.delete(repo);
      toast(`Download of ${name} failed${d.error ? ": " + d.error : ""}`, "error", 9000);
      runHfSearch();
      return;
    }
  }
  hfJobs.delete(repo);
  toast(`Still downloading ${name} in LM Studio — check its downloads panel`, "info", 8000);
}

function syncKv() {
  const flash = (settings.get("model.attention") || "flash") === "flash";
  for (const b of document.querySelectorAll("#attnSeg button")) {
    b.classList.toggle("active", b.dataset.v === (flash ? "flash" : "standard"));
  }
  $("kvGpu").checked = settings.get("model.kvGpu") !== false;
  $("kCache").value = settings.get("model.kCache") || "";
  $("vCache").value = flash ? (settings.get("model.vCache") || "") : "";
  $("vCache").disabled = !flash;
  $("vCacheRow").classList.toggle("disabled", !flash);
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
    if (inst && inst.flash_attention != null) {
      settings.set("model.attention", inst.flash_attention ? "flash" : "standard");
    }
    if (inst && inst.offload_kv_cache_to_gpu != null) {
      settings.set("model.kvGpu", !!inst.offload_kv_cache_to_gpu);
    }
    ctx.dataset.for = m.key;
  }
  $("ctxMax").textContent = m.max_context ? `max ${m.max_context.toLocaleString()}` : "";
  syncKv();
  btn.disabled = loading;
  btn.textContent = loading ? "Loading…" : m.loaded ? "Reload with these settings" : "Load this model";
}

async function doLoad() {
  const m = models.find((x) => x.key === selected) || currentInfo();
  if (!m || loading) return;
  const context_length = parseInt($("ctxLen").value, 10) || null;
  const flash = (settings.get("model.attention") || "flash") === "flash";
  if (!confirm(`Load "${m.name}" with a ${context_length ? context_length.toLocaleString() : "default"}-token context?\n` +
               "Other loaded chat models are unloaded first to free VRAM.")) return;
  loading = true;
  renderLoad();
  toast(`Loading ${m.name}… this can take a minute`, "info", 6000);
  try {
    const r = await fetch("/api/models/load", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        key: m.key, context_length,
        flash_attention: flash,
        offload_kv_cache_to_gpu: settings.get("model.kvGpu") !== false,
        k_cache: settings.get("model.kCache") || "",
        v_cache: flash ? (settings.get("model.vCache") || "") : "",
      }),
    });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || `HTTP ${r.status}`);
    toast(`${m.name} is loaded`);
    $("ctxLen").dataset.for = "";
    closePicker();
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
    if (!input) continue;
    const d = defaults[defKey];
    const shown = m[key] ?? d;
    if (document.activeElement !== input) input.value = shown != null && shown !== "" ? shown : "";
    input.placeholder = d != null ? String(d) : "model default";
  }
  const sp = $("sysPrompt");
  if (document.activeElement !== sp) sp.value = m.systemPrompt || "";
  sp.placeholder = defaults.system_prompt ? "Default: " + defaults.system_prompt : "System prompt";
  $("sysState").textContent = m.systemPrompt ? "custom" : "default";
  $("thinking").checked = !!m.thinking;
  renderPrompts();
}

function renderPrompts() {
  const lib = settings.get("model.promptLibrary") || [];
  const sel = $("promptLib");
  const cur = (settings.get("model.systemPrompt") || "").trim();
  sel.innerHTML = `<option value="">— saved prompts —</option>`;
  for (const p of lib) sel.append(new Option(p.name, p.id));
  const match = lib.find((p) => p.text === cur);
  sel.value = match ? match.id : "";
  const hasSel = !!sel.value;
  $("promptDel").disabled = !hasSel;
  $("promptUpdate").disabled = !hasSel;
}

function applySavedPrompt() {
  const lib = settings.get("model.promptLibrary") || [];
  const p = lib.find((x) => x.id === $("promptLib").value);
  if (p) settings.set("model.systemPrompt", p.text);
}

function savePrompt() {
  const text = $("sysPrompt").value.trim();
  if (!text) { toast("Type a system prompt first", "error"); return; }
  const name = prompt("Name this system prompt:", "Saved prompt");
  if (!name || !name.trim()) return;
  const lib = [...(settings.get("model.promptLibrary") || [])];
  const id = "p" + Date.now().toString(36);
  lib.push({ id, name: name.trim(), text });
  settings.set("model.systemPrompt", text);
  settings.set("model.promptLibrary", lib);
  toast(`Saved “${name.trim()}”`);
}

function updatePrompt() {
  const id = $("promptLib").value;
  if (!id) { toast("Select a saved prompt to overwrite", "error"); return; }
  const text = $("sysPrompt").value.trim();
  if (!text) { toast("Type a system prompt first", "error"); return; }
  const lib = (settings.get("model.promptLibrary") || []).map((p) =>
    p.id === id ? { ...p, text } : p);
  settings.set("model.systemPrompt", text);
  settings.set("model.promptLibrary", lib);
  const name = lib.find((p) => p.id === id)?.name || "preset";
  toast(`Updated “${name}”`);
}

function deletePrompt() {
  const id = $("promptLib").value;
  if (!id) return;
  const lib = (settings.get("model.promptLibrary") || []).filter((p) => p.id !== id);
  settings.set("model.promptLibrary", lib);
}
