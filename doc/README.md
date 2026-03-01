# Build Your Own Multi-Agent Framework — Lab Guide

> **Stanford CS — Spring 2026**
> You will build a complete, working Python multi-agent AI framework from scratch over 5 phases.
> Each phase has its own guide file. Follow them in order.

---

## What You Are Building

A framework called **`myagents`** (name it whatever you like) that lets users do this:

```python
from myagents import SimpleAgent, ReActAgent, LLMClient, ToolRegistry

llm = LLMClient(provider="openai")
agent = ReActAgent(name="researcher", llm=llm, tool_registry=registry)
answer = agent.run("What is the GDP of France in 2024?")
```

By the end, your framework will support:

- A unified LLM client that works with OpenAI, DeepSeek, Ollama, and more
- 4 distinct agent paradigms (Simple, ReAct, Reflection, Plan-and-Solve)
- A pluggable tool system with a registry, chains, and async execution
- Built-in tools (calculator, web search)
- Clean utilities for logging and serialization

---

## Project Structure You Will Build

```
myagents/
├── __init__.py              ← Phase 4
├── core/
│   ├── __init__.py
│   ├── exceptions.py        ← Phase 1, Step 1
│   ├── message.py           ← Phase 1, Step 2
│   ├── config.py            ← Phase 1, Step 3
│   ├── llm.py               ← Phase 1, Step 4
│   └── agent.py             ← Phase 1, Step 5
├── tools/
│   ├── __init__.py
│   ├── base.py              ← Phase 2, Step 1
│   ├── registry.py          ← Phase 2, Step 2
│   ├── chain.py             ← Phase 2, Step 3
│   ├── async_executor.py    ← Phase 2, Step 4
│   └── builtin/
│       ├── __init__.py
│       ├── calculator.py    ← Phase 2, Step 5
│       └── search.py        ← Phase 2, Step 6
├── agents/
│   ├── __init__.py
│   ├── simple_agent.py      ← Phase 3, Step 1
│   ├── react_agent.py       ← Phase 3, Step 2
│   ├── reflection_agent.py  ← Phase 3, Step 3
│   └── plan_solve_agent.py  ← Phase 3, Step 4
└── utils/
    ├── __init__.py
    ├── logging.py           ← Phase 4, Step 1
    ├── serialization.py     ← Phase 4, Step 2
    └── helpers.py           ← Phase 4, Step 3
```

---

## The 5 Phases

| Phase | File                    | Topic                                                                | Estimated Time |
| ----- | ----------------------- | -------------------------------------------------------------------- | -------------- |
| **0** | `phase0_setup.md`       | Environment setup, project scaffold                                  | 30 min         |
| **1** | `phase1_core.md`        | Core primitives: exceptions, message, config, LLM client, agent base | 3–4 hours      |
| **2** | `phase2_tools.md`       | Tool system: base, registry, chain, async executor, builtins         | 3–4 hours      |
| **3** | `phase3_agents.md`      | Four agent paradigms                                                 | 4–6 hours      |
| **4** | `phase4_integration.md` | Utils, public API, end-to-end demo                                   | 2–3 hours      |

**Start here → [`phase0_setup.md`](./phase0_setup.md)**

---

## Rules

1. **No copying.** You may reference the concept descriptions in these guides, but write every line yourself.
2. **No `eval()`.** Ever. Use `ast.literal_eval` for data, AST traversal for expressions.
3. **Tests required.** Each phase ends with a self-check section. Run it before moving on.
4. **Type hints required** on all functions and class attributes.
5. **Ask before using external libraries** not listed in a phase's dependency list.
