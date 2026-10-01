"""Quick Hugging Face GGUF search + VRAM budget check (needs network)."""
import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import aiohttp
from voicelink.hfsearch import search, vram_gb


async def main():
    print("VRAM GiB:", vram_gb())
    async with aiohttp.ClientSession() as http:
        data = await search(http, "qwen3.5 4b", vram=vram_gb(), ignore=False, limit=5)
    print("ok", data.get("ok"), "budget", data.get("budget_bytes"), "n", len(data.get("results") or []))
    for r in data.get("results") or []:
        print(f"  {r['repo']:50} {r['quant']:8} {r.get('size')} fits={r.get('fits')}")


if __name__ == "__main__":
    asyncio.run(main())
