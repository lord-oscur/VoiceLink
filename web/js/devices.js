// Microphone / speaker lists, limited to physical devices unless asked otherwise.

// Drivers that create virtual endpoints (voice changers, mixers, streaming tools,
// audio enhancers, loopback cables...).
const VIRTUAL = new RegExp([
  "virtual", "voicemod", "animaze", "fxsound", "vb-audio", "voicemeeter", "nvidia broadcast", "rtx voice",
  "steam streaming", "\\bobs\\b", "krisp", "wave ?link", "droidcam", "iriun", "splitcam", "xsplit", "manycam",
  "snap camera", "zoomaudio", "sonic studio", "nahimic", "blackhole", "soundflower", "loopback", "stereo mix",
  "mezcla est", "what u hear", "epoccam", "\\bndi\\b", "streamlabs", "\\bcamo\\b", "discord", "vcable",
].join("|"), "i");

export const isVirtual = (label) => VIRTUAL.test(label || "");

// "Microphone (USB Audio) (0c76:161f)" -> "Microphone (USB Audio)"
export const cleanLabel = (label) => (label || "").replace(/\s*\([0-9a-f]{4}:[0-9a-f]{4}\)\s*$/i, "").trim();

export const canPickSpeaker = typeof AudioContext !== "undefined" && "setSinkId" in AudioContext.prototype;

// Returns {mic: [...], speaker: [...]} where each entry is {id, label, virtual}.
// The pseudo devices "default"/"communications" become one "System default" entry (id "").
export async function listDevices(showVirtual, keep = {}) {
  const all = await navigator.mediaDevices.enumerateDevices();
  const out = { mic: [], speaker: [], labelled: all.some((d) => d.label) };
  for (const [kind, key] of [["audioinput", "mic"], ["audiooutput", "speaker"]]) {
    const list = all.filter((d) => d.kind === kind);
    const def = list.find((d) => d.deviceId === "default");
    const defName = def ? cleanLabel(def.label.replace(/^(default|predeterminado)\s*-\s*/i, "")) : "";
    out[key].push({ id: "", label: "System default" + (defName ? ` (${defName})` : ""), virtual: false });
    for (const d of list) {
      if (d.deviceId === "default" || d.deviceId === "communications" || !d.deviceId) continue;
      const virtual = isVirtual(d.label);
      if (virtual && !showVirtual && d.deviceId !== keep[key]) continue;
      out[key].push({ id: d.deviceId, label: cleanLabel(d.label) || `${key === "mic" ? "Microphone" : "Speaker"} ${out[key].length}`, virtual });
    }
  }
  return out;
}

export function fillSelect(select, items, value) {
  select.innerHTML = "";
  for (const d of items) select.append(new Option(d.label + (d.virtual ? "  (virtual)" : ""), d.id));
  select.value = items.some((d) => d.id === value) ? value : "";
}
