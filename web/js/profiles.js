// Settings profiles popover: switch, save as, rename, delete, reset, export, import.
import { settings } from "./settings.js";
import { $, download, toast } from "./ui.js";

export function initProfiles() {
  const sel = $("profileSel");
  const render = () => {
    sel.innerHTML = "";
    for (const name of settings.profileNames) sel.append(new Option(name, name));
    sel.value = settings.active;
    $("profDelete").disabled = settings.profileNames.length <= 1;
    $("profileName").textContent = settings.active;
  };
  settings.onChange((paths) => { if (paths.includes("*") || paths.includes("profile")) render(); });
  render();

  sel.onchange = () => { settings.switchTo(sel.value); toast(`Profile "${sel.value}" loaded`); };
  $("profNew").onclick = () => {
    const name = prompt("Save the current settings as a new profile named:", settings.active + " copy");
    if (name && settings.saveAs(name)) toast(`Profile "${name.trim()}" saved`);
  };
  $("profRename").onclick = () => {
    const name = prompt("Rename this profile to:", settings.active);
    if (!name || name.trim() === settings.active) return;
    if (!settings.rename(name)) toast("A profile with that name already exists", "error");
  };
  $("profDelete").onclick = () => {
    const name = settings.active;
    if (confirm(`Delete the profile "${name}"?`) && settings.remove()) toast(`Profile "${name}" deleted`);
  };
  $("profReset").onclick = () => {
    if (confirm(`Reset every setting of "${settings.active}" to its default?`)) settings.resetProfile();
  };
  $("exportBtn").onclick = () => {
    const d = new Date();
    const stamp = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}`;
    download(`voicelink-settings-${stamp}.json`, settings.exportJSON());
  };
  $("importBtn").onclick = () => $("importFile").click();
  $("importFile").onchange = async () => {
    const file = $("importFile").files[0];
    $("importFile").value = "";
    if (!file) return;
    try {
      const names = settings.importJSON(await file.text());
      toast(`Imported ${names.length} profile${names.length > 1 ? "s" : ""}: ${names.join(", ")}`);
    } catch (e) {
      toast("Import failed: " + e.message, "error", 5000);
    }
  };
}
