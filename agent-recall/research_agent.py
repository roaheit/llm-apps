"""
research_agent.py — The killer demo.

A research agent that runs across two sessions and demonstrates
what makes agent-recall different: it actually remembers.

Session 1: Tell the agent your preferences, do some research.
Session 2: Run again — the agent picks up exactly where it left off.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/research_agent.py --session 1
    python examples/research_agent.py --session 2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from agent_recall import Memory


SYSTEM_PROMPT = """You are a research assistant. 
You have access to a memory system that persists across sessions.
Use the provided memory context to personalize your responses and avoid repeating work.
Be concise. Reference prior memory when relevant."""


def build_prompt(user_message: str, memory_context: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if memory_context:
        messages.append({
            "role": "system",
            "content": f"[MEMORY CONTEXT]\n{memory_context}\n[END MEMORY]",
        })
    messages.append({"role": "user", "content": user_message})
    return messages


def run_session_1(mem: Memory, client: OpenAI) -> None:
    print("\n" + "═" * 60)
    print("  SESSION 1: Setting preferences + first research run")
    print("═" * 60)

    # Step 1: Tell the agent your preferences
    pref_message = (
        "Hi! I'm researching AI agent architectures. "
        "I prefer concise bullet-point summaries, no more than 5 points. "
        "I'm most interested in memory systems and multi-agent coordination."
    )
    print(f"\n[User] {pref_message}\n")

    context = mem.recall(pref_message)
    messages = build_prompt(pref_message, context)
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=300)
    reply = response.choices[0].message.content

    print(f"[Agent] {reply}\n")

    # Store the user preference in long-term memory
    mem.remember(
        "User prefers concise bullet-point summaries (max 5 points). "
        "Focused on memory systems and multi-agent coordination.",
        layer="long_term",
        metadata={"type": "user_preference"},
    )
    mem.short_term.add(f"User: {pref_message}")
    mem.short_term.add(f"Agent: {reply}")

    # Step 2: Do some research
    research_query = "What are the main approaches to agent memory in LLM systems?"
    print(f"[User] {research_query}\n")

    context = mem.recall(research_query)
    messages = build_prompt(research_query, context)
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=400)
    research_reply = response.choices[0].message.content

    print(f"[Agent] {research_reply}\n")

    # Log the research action to episodic memory
    mem.log_episode(
        action="research_query",
        result=f"Researched: {research_query}",
        tags=["research", "memory_systems"],
        metadata={"query": research_query},
    )

    # Store findings in long-term
    mem.remember(
        f"Already researched: main approaches to agent memory. Key topics covered: "
        f"in-context storage, external vector stores, episodic logs, compression strategies.",
        layer="long_term",
        metadata={"type": "research_completed"},
    )

    # Compress short-term to long-term before ending session
    print("[System] Compressing session buffer to long-term memory...\n")
    summary = mem.compress()
    if summary:
        print(f"[Summary saved] {summary[:200]}...\n")

    print("═" * 60)
    print("  Session 1 complete. Stats:")
    stats = mem.stats()
    print(f"  Long-term memories: {stats['long_term']['count']}")
    print(f"  Episodes logged:    {stats['episodic']['total_episodes']}")
    print("═" * 60)


def run_session_2(mem: Memory, client: OpenAI) -> None:
    print("\n" + "═" * 60)
    print("  SESSION 2: Resuming — watch the agent remember everything")
    print("═" * 60)

    # Ask something that requires memory of Session 1
    query = "What have we covered so far on agent memory? What should we look at next?"
    print(f"\n[User] {query}\n")

    context = mem.recall(query, layers=["long_term", "episodic"])
    print(f"[Memory retrieved]\n{context}\n")

    messages = build_prompt(query, context)
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=400)
    reply = response.choices[0].message.content

    print(f"[Agent] {reply}\n")

    # Ask a follow-up that tests preference memory
    pref_query = "Give me an overview of multi-agent coordination patterns."
    print(f"[User] {pref_query}\n")

    context = mem.recall(pref_query)
    messages = build_prompt(pref_query, context)
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=400)
    pref_reply = response.choices[0].message.content

    print(f"[Agent] {pref_reply}\n")
    print("(Notice: the agent still uses bullet points — it remembered your preference from Session 1)\n")

    mem.log_episode(
        action="session_resume",
        result="Successfully resumed research with full context from Session 1",
        tags=["session", "memory_test"],
    )

    print("═" * 60)
    print("  Session 2 complete. Final stats:")
    stats = mem.stats()
    print(f"  Long-term memories: {stats['long_term']['count']}")
    print(f"  Episodes logged:    {stats['episodic']['total_episodes']}")
    print("═" * 60)


def main():
    parser = argparse.ArgumentParser(description="agent-recall demo: research agent with persistent memory")
    parser.add_argument("--session", type=int, choices=[1, 2], required=True, help="Which session to run")
    parser.add_argument("--reset", action="store_true", help="Clear all memory before running")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    mem = Memory(agent_id="research-agent", api_key=api_key)
    client = OpenAI(api_key=api_key)

    if args.reset:
        mem.clear()
        print("[Memory cleared]\n")

    if args.session == 1:
        run_session_1(mem, client)
    else:
        run_session_2(mem, client)


if __name__ == "__main__":
    main()
