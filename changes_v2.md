# VoiceLink: changes v2

New requirements after v1 (through commit `67bbb91` and the later
emotion / panel work), the decisions taken here, and the order of
implementation (easiest first). Status is updated as each item lands.

## Requirements

| # | Requirement (as requested) | Status |
|---|---|---|
| 1 | Move the top-right **+** (new chat) to the left of the top bar. | done |
| 2 | Voices and Devices each get their own toolbar icon; their content opens in the right panel (same place as model controls). | done |
| 3 | Settings profiles move to the bottom of the left conversation panel. | done |
| 4 | Change-model control: icon on the model card and left of the loaded name in the top bar; both open a **centered** model picker with filters (capabilities, type, architecture, quant, 18+/Abliterated) and sort (size, context, parameters, expert count). | done |
| 5 | Right-panel model extras: saved system prompts; generation fields show the values that will actually be sent; load settings gain K cache / V cache (V only with flash attention) and the attention types LM Studio can set. | done |
| 6 | Category labels on conversations, to group them. | done |
| 7 | Project folders that chats can be moved into. | done |
| 8 | Incognito mode: do not store that chat on disk. | done |

## Decisions

* **Right panel** is one column with three pages (Model / Voice / Devices).
  The three toolbar icons (and the model pill) open the matching page and
  show the panel. The old voice hover popover is gone.
* **New chat +** sits immediately after the conversations toggle, on the
  left of the header. The left panel still has its own New chat button.
* **Model picker** reads `/api/v1/models`. Type (dense / MoE / MTP),
  18+ (name contains NSFW or 18+) and Abliterated (name contains
  “abliterated”) are inferred from the model key/name when LM Studio does
  not expose them. Architecture and quant come from the API.
* **K / V cache.** REST load supports `offload_kv_cache_to_gpu` and
  `flash_attention`. Separate K/V *quantization* types are a TypeScript-SDK
  feature; the UI offers them and retries without them if LM Studio rejects
  the keys. V-cache controls stay disabled until flash attention is on.
  Attention choice is Standard vs Flash (what REST can set).
* **Saved prompts** live in the browser profile (`model.promptLibrary`).
* **Labels and projects** are stored with the conversation JSON
  (`data/conversations/`) and `data/projects.json`.
* **Incognito** is a session flag: nothing is written to disk, including
  titles. Turning it off does not retroactively save the current chat.
* **Empty picker search** queries Hugging Face (`filter=gguf`). GGUF quants
  are limited to about 72% of detected NVIDIA VRAM (8 GB here) so a load
  can still leave room for context. The budget GB and “show quants that
  do not fit” override that. A chosen row calls LM Studio
  `POST /api/v1/models/download`; the UI polls the job and toasts when
  the file is ready to load.
* **+18** matches `uncensored` or `NSFW` in the key/name (case-insensitive).
  **Not Filtered** matches `Abliterated`. The Change-model card keeps a
  fixed height so refresh does not resize the dialog.
* **Output language** (Voice page, default *Stay in the voice's language*)
  tells the LLM to reply in the picked voice's language so TTS is not
  given Spanish text with an English voice. *Follow what I speak* is the
  old auto behaviour.
* **System prompt** save (new preset) and check (overwrite selected) sit
  on the card title. **Reload** is pinned to the bottom of the right panel.
  Top K defaults to 40, Repeat penalty to 1.1.

## Implementation order (easy → hard)

1. Move **+** to the left of the top bar (1).
2. Split Voices / Devices into toolbar icons and right-panel pages (2).
3. Move the profile dock to the bottom of the left panel (3).
4. Saved system prompts and filled-in generation values (part of 5).
5. Load-settings: GPU KV offload, attention (Standard / Flash), optional
   K/V cache types with a safe fallback (rest of 5).
6. Centered model picker with filters and sort (4).
7. Labels and project folders (6, 7).
8. Incognito mode (8).
