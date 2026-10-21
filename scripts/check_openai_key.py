#!/usr/bin/env python3
"""Check that OPENAI_API_KEY can call the model the SO101 LLM policy uses.

Run it in the same terminal as colosseum-robot (after `export OPENAI_API_KEY=...`):
    python3 scripts/check_openai_key.py [model]
The key itself is never printed.
"""
import json
import os
import sys
import urllib.error
import urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-6-astra"
BASE = "https://api.openai.com/v1"


def call(method, path, body=None):
    request = urllib.request.Request(
        BASE + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.load(exc)
        except ValueError:
            return exc.code, {}


key = os.environ.get("OPENAI_API_KEY", "").strip()
if not key:
    sys.exit("OPENAI_API_KEY is not set in this shell.")
print(f"key: {key[:8]}...{key[-4:]} (length {len(key)})")

status, body = call("GET", "/models")
print(f"[1] list models -> HTTP {status}")
if status != 200:
    sys.exit(f"    key rejected: {body.get('error', body)}")
ids = sorted(m["id"] for m in body.get("data", []))
print(f"    {MODEL!r} visible to this key: {MODEL in ids}")
if MODEL not in ids:
    print("    gpt-6* models visible:", [i for i in ids if i.startswith("gpt-6")] or "none")

status, body = call("POST", "/responses", {
    "model": MODEL, "input": "Reply with the single word: ok",
    "reasoning": {"effort": "medium"}, "max_output_tokens": 256})
print(f"[2] responses call -> HTTP {status}")
if status != 200:
    sys.exit(f"    request failed: {body.get('error', body)}")
print("    OK, output:", body.get("output_text") or json.dumps(body.get("output"))[:200])
