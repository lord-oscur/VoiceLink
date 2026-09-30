// Spelling / grammar / redundancy checker for the message box.
// A backdrop behind the (transparent) textarea draws the underlines; clicking a
// marked word opens a suggestion popup. Checks run on the server (/api/check),
// plus an optional "Check with AI" pass (/api/check_ai) through LM Studio.
import { settings } from "./settings.js";
import { $, esc, toast } from "./ui.js";

const DICT_KEY = "voicelink.dictionary";
const LABEL = { spelling: "Spelling", grammar: "Grammar", style: "Redundancy / style", ai: "AI suggestion" };

let ta, backdrop, box;
let issues = [];          // {start, end, type, message, suggestions, text}
let aiIssues = [];
let lastText = "";
let timer = null;
let reqId = 0;
let lang = "en";
let open = -1;

const dictionary = () => { try { return JSON.parse(localStorage.getItem(DICT_KEY)) || []; } catch (_) { return []; } };

export function initEditor() {
  ta = $("textInput");
  backdrop = $("backdrop");
  box = $("suggestBox");
  ta.addEventListener("input", onInput);
  ta.addEventListener("scroll", () => { backdrop.scrollTop = ta.scrollTop; });
  ta.addEventListener("click", showAtCaret);
  ta.addEventListener("keyup", (e) => { if (e.key.startsWith("Arrow") || e.key === "Home" || e.key === "End") showAtCaret(); });
  ta.addEventListener("keydown", (e) => { if (e.key === "Escape" && !box.hidden) { e.stopPropagation(); hide(); } });
  ta.addEventListener("blur", () => setTimeout(() => { if (!box.contains(document.activeElement)) hide(); }, 150));
  new ResizeObserver(render).observe(ta);
  document.addEventListener("mousedown", (e) => { if (!box.contains(e.target) && e.target !== ta) hide(); });

  $("checkOn").onchange = () => settings.set("editor.check", $("checkOn").checked);
  $("checkLang").onchange = () => settings.set("editor.lang", $("checkLang").value);
  $("checkAI").onclick = checkWithAI;
  $("dictClear").onclick = () => {
    if (!dictionary().length || !confirm("Remove every word from your personal dictionary?")) return;
    localStorage.removeItem(DICT_KEY);
    renderPanel();
    schedule(0);
  };
  settings.onChange((paths) => {
    if (paths.includes("*") || paths.some((p) => p.startsWith("editor"))) { renderPanel(); schedule(0); }
  });
  renderPanel();
}

// The message box was cleared (sent) or replaced from outside.
export function reset() {
  issues = [];
  aiIssues = [];
  lastText = ta.value;
  hide();
  render();
}

function renderPanel() {
  const on = !!settings.get("editor.check");
  $("checkOn").checked = on;
  $("checkLang").value = settings.get("editor.lang") || "auto";
  ta.spellcheck = !on;                      // no double underlines from the browser
  $("dictCount").textContent = dictionary().length;
  $("checkBtn").classList.toggle("off", !on);
  if (!on) { issues = []; render(); }
}

// Keep marks in place while typing: shift the ones after the edit, drop the touched ones.
function onInput() {
  const text = ta.value;
  const old = lastText;
  let p = 0;
  while (p < old.length && p < text.length && old[p] === text[p]) p++;
  let s = 0;
  while (s < old.length - p && s < text.length - p && old[old.length - 1 - s] === text[text.length - 1 - s]) s++;
  const oldEnd = old.length - s, delta = text.length - old.length;
  const shift = (list) => list.filter((i) => i.end <= p || i.start >= oldEnd)
    .map((i) => (i.start >= oldEnd ? { ...i, start: i.start + delta, end: i.end + delta } : i));
  issues = shift(issues);
  aiIssues = shift(aiIssues);
  lastText = text;
  hide();
  render();
  schedule();
}

function schedule(ms = 550) {
  clearTimeout(timer);
  if (!settings.get("editor.check")) return;
  timer = setTimeout(runCheck, ms);
}

async function runCheck() {
  const text = ta.value;
  const id = ++reqId;
  if (!text.trim()) { issues = []; render(); return; }
  try {
    const r = await fetch("/api/check", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, lang: settings.get("editor.lang") || "auto", ignore: dictionary() }),
    });
    const d = await r.json();
    if (id !== reqId || ta.value !== text) return;
    issues = d.issues;
    lang = d.lang;
    $("checkDetected").textContent = lang === "es" ? "Spanish" : "English";
    render();
  } catch (_) { /* server restarting: try again on the next edit */ }
}

async function checkWithAI() {
  const text = ta.value;
  if (!text.trim()) { toast("Type something first"); return; }
  const btn = $("checkAI");
  btn.disabled = true;
  btn.classList.add("busy");
  try {
    const r = await fetch("/api/check_ai", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, lang: settings.get("editor.lang") || "auto" }),
    });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error);
    if (ta.value !== text) return;
    aiIssues = d.issues;
    render();
    const extra = allIssues().length - (settings.get("editor.check") ? issues.length : 0);
    toast(!aiIssues.length ? "The AI found nothing to change" :
      extra ? `The AI suggests ${extra} more change${extra > 1 ? "s" : ""} (green)` :
        "The AI agrees with the underlined issues");
  } catch (e) {
    toast("AI check failed: " + e.message, "error");
  } finally {
    btn.disabled = false;
    btn.classList.remove("busy");
  }
}

function allIssues() {
  // AI suggestions only where the rule checker found nothing.
  const extra = aiIssues.filter((a) => !issues.some((i) => i.start < a.end && a.start < i.end));
  return [...issues, ...extra].sort((a, b) => a.start - b.start);
}

function render() {
  if (!ta) return;
  const text = ta.value;
  const list = settings.get("editor.check") ? allIssues() : aiIssues;
  let html = "", pos = 0;
  list.forEach((i, n) => {
    if (i.start < pos || i.end > text.length) return;
    html += esc(text.slice(pos, i.start));
    html += `<mark class="t-${i.type}" data-i="${n}">${esc(text.slice(i.start, i.end)) || " "}</mark>`;
    pos = i.end;
  });
  html += esc(text.slice(pos)) + "\n";
  backdrop.innerHTML = html;
  backdrop.dataset.count = list.length;
  // Match the textarea's inner width when it shows a scrollbar.
  const sb = ta.offsetWidth - ta.clientWidth - 2;
  backdrop.style.paddingRight = `calc(14px + ${Math.max(0, sb)}px)`;
  backdrop.scrollTop = ta.scrollTop;
  const count = list.length;
  $("checkCount").textContent = count ? String(count) : "";
  $("checkCount").hidden = !count;
  render.list = list;
}

function showAtCaret() {
  const list = render.list || [];
  const c = ta.selectionStart;
  if (ta.selectionEnd !== c) { hide(); return; }
  const n = list.findIndex((i) => i.start <= c && c <= i.end);
  if (n < 0) { hide(); return; }
  show(n);
}

function show(n) {
  const i = render.list[n];
  const mark = backdrop.querySelector(`mark[data-i="${n}"]`);
  if (!i || !mark) return;
  open = n;
  const sugg = i.suggestions.length ? i.suggestions.map((s, k) =>
    `<button type="button" class="sugg" data-k="${k}">${s === "" ? "<em>remove</em>" : esc(s)}</button>`).join("")
    : `<span class="hint">No suggestion</span>`;
  box.innerHTML = `<div class="sugg-head t-${i.type}">${LABEL[i.type] || i.type}</div>
    <div class="sugg-msg">${esc(i.message)}</div><div class="sugg-list">${sugg}</div>
    <div class="sugg-foot"><button type="button" class="link-btn" data-act="ignore">Ignore</button>
    ${i.type === "spelling" ? `<button type="button" class="link-btn" data-act="learn">Add to dictionary</button>` : ""}</div>`;
  box.hidden = false;
  const r = mark.getBoundingClientRect();
  const host = box.offsetParent.getBoundingClientRect();
  const top = r.top - host.top - box.offsetHeight - 6;
  box.style.left = Math.max(0, Math.min(r.left - host.left, host.width - box.offsetWidth)) + "px";
  box.style.top = (top < -host.top + 50 ? r.bottom - host.top + 6 : top) + "px";
  box.onclick = (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.k != null) apply(i, i.suggestions[+b.dataset.k]);
    else if (b.dataset.act === "ignore") drop(i);
    else if (b.dataset.act === "learn") learn(i);
  };
}

function hide() {
  if (box) box.hidden = true;
  open = -1;
}

function apply(i, s) {
  let start = i.start, end = i.end;
  if (s === "" && ta.value[end] === " " && (start === 0 || ta.value[start - 1] === " ")) end++;
  ta.focus();
  ta.setRangeText(s, start, end, "end");
  ta.dispatchEvent(new Event("input"));
  hide();
}

function drop(i) {
  issues = issues.filter((x) => x !== i);
  aiIssues = aiIssues.filter((x) => x !== i);
  render();
  hide();
  ta.focus();
}

function learn(i) {
  const words = dictionary();
  const w = i.text.toLowerCase();
  if (!words.includes(w)) words.push(w);
  localStorage.setItem(DICT_KEY, JSON.stringify(words));
  renderPanel();
  issues = issues.filter((x) => !(x.type === "spelling" && x.text.toLowerCase() === w));
  render();
  hide();
  ta.focus();
}
