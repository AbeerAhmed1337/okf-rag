"""
test_chat.py — Quick SSE stream tester for the /chat endpoint.

Run from the project root:
    .venv\Scripts\python.exe test_chat.py

Requires: pip install httpx (already in venv via openai dependency)
"""

import httpx
import json
import sys

BASE_URL = "http://127.0.0.1:8000"
QUESTION = sys.argv[1] if len(sys.argv) > 1 else "What is the Open Knowledge Format?"

print(f"\n{'='*60}")
print(f"Question: {QUESTION}")
print(f"{'='*60}\n")

thought_buffer = []
answer_buffer = []

with httpx.Client(timeout=120.0) as client:
    with client.stream(
        "POST",
        f"{BASE_URL}/api/v1/chat",
        json={"message": QUESTION},
        headers={"Accept": "text/event-stream"},
    ) as response:
        print(f"Status: {response.status_code}\n")

        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue

            payload = json.loads(line[6:])
            event_type = payload.get("event_type", "")
            content    = payload.get("content", "")

            if event_type == "context":
                print(f"[CONTEXT] {content}\n")

            elif event_type == "thought":
                thought_buffer.append(content)
                print(f"\033[90m[THINKING] {content}\033[0m", end="", flush=True)

            elif event_type == "answer":
                answer_buffer.append(content)
                print(content, end="", flush=True)

            elif event_type == "done":
                print(f"\n\n{'='*60}")
                print(f"Stream complete.")
                print(f"  Thought tokens : {len(''.join(thought_buffer))}")
                print(f"  Answer tokens  : {len(''.join(answer_buffer))}")
                print(f"{'='*60}\n")

            elif event_type == "error":
                print(f"\n[ERROR] {content}")
