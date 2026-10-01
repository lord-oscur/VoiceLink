"""Stateless MCP + clock/tools gating (no speech models, no network).

    .venv\\Scripts\\python tools/test_mcp.py
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from voicelink.llm import LMStudioClient, _merge_tool_delta  # noqa: E402
from voicelink.mcp import (  # noqa: E402
    LOCAL_TOOLS, TOOLS_HINT, clock_line, handle_rpc, openai_tools,
    reject_legacy, _safe_url,
)
from voicelink.session import Session  # noqa: E402
from voicelink.textproc import MOOD_TAG_HINT  # noqa: E402


def eq(got, want, label):
    if got != want:
        raise SystemExit(f"FAIL {label}\n  got  {got!r}\n  want {want!r}")
    print("ok ", label)


def ok(cond, label):
    if not cond:
        raise SystemExit(f"FAIL {label}")
    print("ok ", label)


# ------------------------------------------------------------------ clock

now = datetime(2026, 9, 30, 23, 54, tzinfo=timezone(timedelta(hours=-7)))
line = clock_line(now)
ok("Wednesday, September 30, 2026" in line, "clock has weekday and date")
ok("11:54 PM" in line, "clock has local time")
ok("2026-09-30T23:54" in line, "clock has ISO time")
ok("training data ends earlier" in line, "clock tells the model time has passed")
ok("web_search" in TOOLS_HINT and "weather" in TOOLS_HINT and "stock_quote" in TOOLS_HINT,
   "tools hint names search, weather, stocks")
ok("do not know" in TOOLS_HINT and "pretending to know" in TOOLS_HINT,
   "tools hint covers unknown facts, not only live data")
ok("examples, not the only cases" in TOOLS_HINT, "tools hint is not limited to weather/stocks")


# ------------------------------------------------------------------ legacy reject

reject_legacy({"name": "local", "url": "local"})
reject_legacy({"name": "remote", "url": "http://127.0.0.1:9000/mcp"})
try:
    reject_legacy({"name": "old", "command": "npx", "args": ["-y", "foo"]})
except ValueError as e:
    ok("stdio" in str(e), "reject command/stdio MCP")
else:
    raise SystemExit("FAIL expected stdio reject")
try:
    reject_legacy({"name": "sse", "url": "http://127.0.0.1:9000/sse"})
except ValueError as e:
    ok("SSE" in str(e) or "sse" in str(e), "reject HTTP+SSE MCP")
else:
    raise SystemExit("FAIL expected SSE reject")


# ------------------------------------------------------------------ catalog / openai

names = [t["name"] for t in LOCAL_TOOLS]
eq(names, ["web_search", "fetch_page", "weather", "stock_quote"], "local tool set")
oa = openai_tools(LOCAL_TOOLS)
eq(oa[0]["type"], "function", "openai wrapper type")
eq(oa[0]["function"]["name"], "web_search", "openai wrapper name")
ok("query" in oa[0]["function"]["parameters"]["properties"], "openai wrapper schema")


# ------------------------------------------------------------------ rpc

out, status = handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
eq(status, 200, "initialize ok")
eq(out["result"]["protocolVersion"], "2025-11-25", "initialize protocol")
ok(out["result"]["capabilities"]["tools"]["listChanged"] is False, "no listChanged sessions")

out, status = handle_rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
eq(status, 200, "tools/list ok")
eq([t["name"] for t in out["result"]["tools"]], names, "tools/list is local catalog")

out, status = handle_rpc({"jsonrpc": "2.0", "id": 3, "method": "ping"})
eq(out["result"], {}, "ping")

out, status = handle_rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
eq((out, status), (None, 202), "initialized notification is ack-only")

out, status = handle_rpc({
    "jsonrpc": "2.0", "id": 4, "method": "tools/list",
    "params": {"_meta": {"io.modelcontextprotocol/protocolVersion": "2024-11-05"}},
})
eq(status, 400, "old protocol version rejected")
eq(out["error"]["code"], -32022, "unsupported version code")

out, status = handle_rpc({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                          "params": {"name": "weather", "arguments": {"place": "X"}}})
eq((out, status), (None, 200), "tools/call defers to async handler")


# ------------------------------------------------------------------ url guard

ok(_safe_url("https://example.com/a") == "https://example.com/a", "public https ok")
ok(_safe_url("http://127.0.0.1/secret") is None, "block loopback")
ok(_safe_url("http://192.168.1.8/x") is None, "block RFC1918")
ok(_safe_url("file:///etc/passwd") is None, "block file")
ok(_safe_url("javascript:alert(1)") is None, "block javascript")


# ------------------------------------------------------------------ tool deltas

bag = {}
_merge_tool_delta(bag, {"index": 0, "id": "c1", "function": {"name": "we"}})
_merge_tool_delta(bag, {"index": 0, "function": {"name": "ather", "arguments": '{"pl'}})
_merge_tool_delta(bag, {"index": 0, "function": {"arguments": 'ace":"Oslo"}'}})
eq(bag[0]["name"], "weather", "streamed tool name")
eq(bag[0]["arguments"], '{"place":"Oslo"}', "streamed tool args")


# ------------------------------------------------------------------ can_tools

client = LMStudioClient.__new__(LMStudioClient)
client.info = {"caps": {"tools": True}}
ok(client.can_tools(), "tools cap on")
client.info = {"caps": {"vision": True}}
ok(not client.can_tools(), "tools cap off")
client.info = None
ok(not client.can_tools(), "no info → no tools")


# ------------------------------------------------------------------ system prompt assembly

class Box:
    pass


def fake_session(can, catalog):
    s = Box()
    s.system_prompt = "You are a helper."
    s.output_lang = "auto"
    s.voice = None
    s.gen = {"history": 10}
    s.cfg = {"llm": {"history_messages": 10}}
    s.eng = Box()
    s.eng.llm = Box()
    s.eng.llm.can_tools = lambda: can
    s.eng.llm.info = {"caps": {"tools": can, "vision": False}}
    s.eng.mcp = Box()
    s.eng.mcp.catalog = lambda: catalog
    s.eng.mcp.openai = lambda: openai_tools(catalog)
    s._tools = Session._tools.__get__(s, Session)
    s._locked_lang = Session._locked_lang.__get__(s, Session)
    return Session._messages(s, [{"role": "user", "content": "hi"}])


offered = fake_session(True, LOCAL_TOOLS)
sys_offered = offered[0]["content"]
ok(sys_offered.startswith("You are a helper."), "user prompt kept")
ok("current local date and time" in sys_offered, "clock always appended when tools on")
ok(TOOLS_HINT in sys_offered, "search instructions when model can use tools")
ok(MOOD_TAG_HINT in sys_offered, "mood hint still present")
eq(offered[1]["content"], "hi", "history unchanged")

blocked = fake_session(False, LOCAL_TOOLS)
sys_blocked = blocked[0]["content"]
ok("current local date and time" in sys_blocked, "clock always appended when tools off")
ok(TOOLS_HINT not in sys_blocked, "no search instructions without tool use")

empty = fake_session(True, [])
ok(TOOLS_HINT not in empty[0]["content"], "no search instructions without a catalog")


# ------------------------------------------------------------------ generate tool loop (no HTTP)

async def _tool_loop():
    calls = []

    async def execute(name, args):
        calls.append((name, args))
        return "Now in Paris: clear, 20 C"

    client = LMStudioClient.__new__(LMStudioClient)
    client.cfg = {"disable_thinking": True}
    client.model = "x"
    client.no_think = "effort"

    async def once(msgs, gen, tools):
        if any(m.get("role") == "tool" for m in msgs):
            yield "content", "Sunny in Paris."
            return
        yield "tool_call", {
            "index": 0, "id": "c1",
            "function": {"name": "weather", "arguments": '{"place":"Paris"}'},
        }

    client._stream_once = once
    out = []
    async for text in LMStudioClient.generate(
            client, [{"role": "user", "content": "weather?"}], {},
            tools=[{"type": "function", "function": {"name": "weather"}}],
            execute=execute):
        out.append(text)
    eq(calls, [("weather", {"place": "Paris"})], "execute once")
    eq("".join(out), "Sunny in Paris.", "only spoken content is yielded")


asyncio.run(_tool_loop())
print("all tests passed")
