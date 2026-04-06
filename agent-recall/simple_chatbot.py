"""
simple_chatbot.py — Minimal example showing agent-recall in a chatbot.

The chatbot remembers facts about you across conversations.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/simple_chatbot.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from agent_recall import Memory


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)

    mem = Memory(agent_id="chatbot-demo", api_key=api_key)
    client = OpenAI(api_key=api_key)

    print("\nagent-recall chatbot demo")
    print("Type 'stats' to see memory stats, 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            print(mem.stats())
            continue

        # Retrieve relevant context
        context = mem.recall(user_input)

        messages = [{"role": "system", "content": "You are a helpful assistant with persistent memory."}]
        if context:
            messages.append({"role": "system", "content": f"[Your memory about this user]\n{context}"})
        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
        )
        reply = response.choices[0].message.content
        print(f"\nAssistant: {reply}\n")

        # Store the exchange in memory
        mem.short_term.add(f"User: {user_input}")
        mem.short_term.add(f"Assistant: {reply}")

        # Auto-compress if buffer is getting large
        if mem.short_term.is_full():
            print("[Compressing memory...]\n")
            mem.compress()

        # Store any apparent facts/preferences in long-term
        if any(kw in user_input.lower() for kw in ["i like", "i prefer", "i am", "i work", "my name"]):
            mem.remember(user_input, layer="long_term", metadata={"type": "user_fact"})


if __name__ == "__main__":
    main()
