// Small shared UI helpers.
import { icon } from "./icons.js";

export const $ = (id) => document.getElementById(id);

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export function fmtK(n) {
  if (!n) return "";
  const k = n / 1024;
  return n >= 1000 ? (k >= 10 ? Math.round(k) : +k.toFixed(1)) + "k" : String(n);
}

export function fmtBytes(n) {
  if (!n) return "";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (n >= 10 || i === 0 ? Math.round(n) : n.toFixed(1)) + " " + u[i];
}

// Capability badges for a model entry from /api/models (see llm.py `_model_v1`).
export function capsHTML(info, { context = true } = {}) {
  if (!info) return "";
  const c = info.caps || {};
  const out = [];
  if (c.reasoning) {
    const opts = (c.reasoning.allowed_options || []).join(", ");
    const def = c.reasoning.default ? `, default ${c.reasoning.default}` : "";
    out.push(`<span class="cap think" title="LM Studio: reasoning / thinking${opts ? " (" + esc(opts + def) + ")" : ""}">${icon("brain")}</span>`);
  }
  if (c.tools) out.push(`<span class="cap tools" title="LM Studio: trained for tool use">${icon("tools")}</span>`);
  if (c.vision) out.push(`<span class="cap vision" title="LM Studio: vision (understands images)">${icon("eye")}</span>`);
  const ctx = info.instances && info.instances[0] && info.instances[0].context_length;
  if (context && (ctx || info.max_context)) {
    const t = ctx ? `Loaded context: ${ctx.toLocaleString()} tokens (model max ${(info.max_context || 0).toLocaleString()})`
                  : `Max context: ${info.max_context.toLocaleString()} tokens`;
    out.push(`<span class="cap ctx" title="${t}">${icon("context")}${fmtK(ctx || info.max_context)}</span>`);
  }
  return out.join("");
}

let toastBox = null;
export function toast(text, kind = "info", ms = 3500) {
  if (!toastBox) {
    toastBox = document.createElement("div");
    toastBox.className = "toasts";
    document.body.append(toastBox);
  }
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.textContent = text;
  toastBox.append(el);
  setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 300); }, ms);
}

export function download(name, text, type = "application/json") {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
