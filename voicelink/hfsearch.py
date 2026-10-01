"""Hugging Face GGUF search for the model picker, sized against GPU VRAM."""
import asyncio
import logging
import re
import subprocess

log = logging.getLogger("hfsearch")

HF = "https://huggingface.co"
UA = {"User-Agent": "VoiceLink/1.0 (local LM Studio helper)"}

# Approximate bits-per-weight so we can reject files that will not fit in VRAM
# when Hugging Face does not report a size.
_BPW = {
    "IQ1_S": 1.7, "IQ1_M": 1.9,
    "IQ2_XXS": 2.1, "IQ2_XS": 2.3, "IQ2_S": 2.5, "IQ2_M": 2.7, "Q2_K": 2.6, "Q2_K_S": 2.5,
    "IQ3_XXS": 3.0, "IQ3_XS": 3.3, "IQ3_S": 3.5, "IQ3_M": 3.7,
    "Q3_K_S": 3.3, "Q3_K_M": 3.9, "Q3_K_L": 4.1,
    "IQ4_XS": 4.3, "IQ4_NL": 4.5, "Q4_0": 4.5, "Q4_1": 5.0, "Q4_K_S": 4.4, "Q4_K_M": 4.8,
    "Q5_0": 5.5, "Q5_1": 6.0, "Q5_K_S": 5.5, "Q5_K_M": 5.7,
    "Q6_K": 6.6, "Q8_0": 8.5, "F16": 16.0, "BF16": 16.0, "F32": 32.0,
}
_QUANT_TOKEN = re.compile(
    r"(IQ[1-4]_[A-Z]{1,3}|Q[2-8]_K_[SML]|Q[2-8]_K|Q[2-8]_[01]|Q[2-8]|F16|BF16|F32)",
    re.I,
)
_PARAMS = re.compile(r"(?:(\d+)\s*[x×]\s*)?(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\b")
_SKIP = re.compile(r"mmproj|embed|encoder|clip|vae|unet|diffusion", re.I)
# Prefer these when several quants fit (best quality that still fits).
_PREF = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "IQ4_XS", "Q4_0", "Q5_K_S", "Q3_K_M",
         "Q6_K", "Q3_K_S", "IQ3_M", "Q8_0", "Q2_K"]

_UNSET = object()
_vram_cache = _UNSET


def vram_gb():
    """Total memory of the largest NVIDIA GPU in GiB, or None if none."""
    global _vram_cache
    if _vram_cache is not _UNSET:
        return _vram_cache
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=4, text=True, stderr=subprocess.DEVNULL,
        )
        mibs = [float(x.strip()) for x in out.splitlines() if x.strip()]
        if not mibs:
            _vram_cache = None
            return None
        mib = max(mibs)
        _vram_cache = round(mib / 1024.0, 1) if mib > 32 else float(mib)
        return _vram_cache
    except Exception:
        _vram_cache = None
        return None


def _parse_quant(filename):
    matches = _QUANT_TOKEN.findall(filename.replace("-", "_"))
    if not matches:
        return None
    token = matches[-1].upper()
    if token in _BPW:
        return token
    if token.endswith("_K") and token + "_M" in _BPW:
        return token + "_M"
    return token


def _parse_params(text):
    m = _PARAMS.search(text.replace("_", " "))
    if not m:
        return None
    experts = float(m.group(1)) if m.group(1) else 1.0
    each = float(m.group(2))
    return experts * each * 1e9


def _est_bytes(params, quant, size):
    if size:
        return int(size)
    bpw = _BPW.get(quant or "", 4.8)
    if not params:
        return None
    return int(params * bpw / 8.0 * 1.08)


def _budget_bytes(vram, ignore):
    if ignore or not vram:
        return None
    # Leave headroom for the KV cache and the desktop.
    return int(float(vram) * 0.72 * 1024 ** 3)


def pick_quant(quants, budget):
    fitting = [q for q in quants if budget is None or (q.get("size") and q["size"] <= budget)
               or (q.get("size") is None)]
    if budget:
        fitting = [q for q in fitting if not q.get("size") or q["size"] <= budget]
    if not fitting:
        return None
    order = {n: i for i, n in enumerate(_PREF)}
    fitting.sort(key=lambda q: (order.get(q["quant"], 80), -(q.get("size") or 0)))
    return fitting[0]


async def _hf_json(http, url, params=None, timeout=15):
    async with http.get(url, params=params, headers=UA,
                        timeout=__import__("aiohttp").ClientTimeout(total=timeout)) as r:
        if r.status != 200:
            raise RuntimeError(f"Hugging Face HTTP {r.status}")
        return await r.json()


async def _repo_files(http, repo):
    info = await _hf_json(http, f"{HF}/api/models/{repo}", timeout=12)
    siblings = info.get("siblings") or []
    blob = f"{repo} {info.get('id') or ''}"
    params = _parse_params(blob)
    files = []
    for s in siblings:
        name = s.get("rfilename") or ""
        if not name.lower().endswith(".gguf") or _SKIP.search(name):
            continue
        quant = _parse_quant(name.split("/")[-1])
        if not quant:
            continue
        size = s.get("size")
        files.append({
            "file": name, "quant": quant,
            "size": _est_bytes(params, quant, size),
        })
    return params, files, info


async def search(http, query, vram=None, ignore=False, limit=10):
    """Search GGUF repos; return rows whose recommended quant fits `vram` GiB."""
    q = (query or "").strip()
    if len(q) < 2:
        return {"ok": True, "results": [], "vram_gb": vram, "budget_bytes": None}
    if vram is None and not ignore:
        vram = vram_gb()
    budget = _budget_bytes(vram, ignore)
    data = await _hf_json(http, f"{HF}/api/models", params={
        "search": q, "filter": "gguf", "sort": "downloads", "direction": "-1", "limit": "20",
    })
    repos = []
    for row in data if isinstance(data, list) else []:
        rid = row.get("id") or ""
        if not rid or "embed" in rid.lower():
            continue
        tags = " ".join(row.get("tags") or []).lower()
        if "gguf" not in tags and "gguf" not in rid.lower():
            # filter=gguf should already restrict this
            pass
        repos.append(row)
        if len(repos) >= 14:
            break

    async def one(row):
        try:
            params, files, info = await _repo_files(http, row["id"])
        except Exception as e:
            log.debug("HF repo %s: %s", row.get("id"), e)
            return None
        if not files:
            return None
        for f in files:
            # Unknown size: keep the file. Known size: must be under the VRAM budget.
            f["fits"] = budget is None or f.get("size") is None or f["size"] <= budget
        shown = files if ignore else [f for f in files if f["fits"]]
        if not shown:
            smallest = min(files, key=lambda f: f.get("size") or 10**18)
            return {
                "repo": row["id"], "name": row.get("id", "").split("/")[-1],
                "url": f"{HF}/{row['id']}", "downloads": row.get("downloads") or 0,
                "params": params, "quants": files, "quant": smallest["quant"],
                "size": smallest.get("size"), "fits": False,
            }
        choice = pick_quant(shown, None if ignore else budget) or shown[0]
        return {
            "repo": row["id"],
            "name": (info.get("id") or row["id"]).split("/")[-1],
            "url": f"{HF}/{row['id']}",
            "downloads": row.get("downloads") or 0,
            "params": params,
            "quants": shown,
            "quant": choice["quant"],
            "size": choice.get("size"),
            "fits": True,
        }

    packed = await asyncio.gather(*[one(r) for r in repos])
    results = [x for x in packed if x]
    results.sort(key=lambda r: (-int(r.get("fits") or 0), -(r.get("downloads") or 0)))
    return {
        "ok": True,
        "results": results[:limit],
        "vram_gb": vram,
        "budget_bytes": budget,
        "ignore": bool(ignore),
    }


