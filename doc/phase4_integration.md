# Phase 4 — Utils, Public API & End-to-End Demo

> **Goal:** Build the utilities layer, wire up the public-facing API, and run a complete end-to-end demo that exercises all four agents.
> **Time:** 2–3 hours
> **Depends on:** Phases 1–3 complete
> **This is the final phase.**

You will build these files in order:

```
utils/logging.py         ← Step 1
utils/serialization.py   ← Step 2
utils/helpers.py         ← Step 3
utils/__init__.py        ← Step 4
myagents/__init__.py     ← Step 5  (the public API surface)
demo.py                  ← Step 6  (end-to-end demo)
```

---

## Step 1 — `utils/logging.py`: Logger Factory

### Why not just use `print()`?

The codebase already uses `print()` for progress output. But for a production-quality library, users need the ability to:

- Redirect output to a file
- Filter by severity (DEBUG, INFO, WARNING, ERROR)
- Turn off all output in production
- Integrate with their own logging setup

Python's `logging` module provides all of this. Your job is to create a factory function that configures a logger consistently.

### What to build

```python
import logging
import sys

def setup_logger(
    name: str = "myagents",
    level: str = "INFO",
    format_string: Optional[str] = None,
) -> logging.Logger:
```

Inside `setup_logger`:

1. Get or create the named logger: `logger = logging.getLogger(name)`
2. Set level: `logger.setLevel(getattr(logging, level.upper()))`
3. **Guard against duplicate handlers:** Only add a handler if `not logger.handlers`. Without this guard, calling `setup_logger` twice adds two handlers and every log message prints twice.
4. Create a `StreamHandler` writing to `sys.stdout`
5. Attach a `Formatter` using the provided format string or this default:
   `'%(asctime)s - %(name)s - %(levelname)s - %(message)s'`
6. Return the logger

Also add:

```python
def get_logger(name: str = "myagents") -> logging.Logger:
    return logging.getLogger(name)
```

### Self-check

```python
from myagents.utils.logging import setup_logger

log = setup_logger("test", level="DEBUG")
log.debug("debug message")
log.info("info message")
log.warning("warning message")

# Test duplicate handler guard:
log2 = setup_logger("test")  # same name, called again
log2.info("only one line, not two")
assert len(logging.getLogger("test").handlers) == 1, "Should have exactly 1 handler"
print("Logger OK")
```

---

## Step 2 — `utils/serialization.py`: Save and Load Objects

### What to build

Four functions for serializing Python objects to disk and loading them back:

```python
def serialize_object(obj: Any, format: str = "json") -> str | bytes:
    """Convert object to string (json) or bytes (pickle)."""

def deserialize_object(data: str | bytes, format: str = "json") -> Any:
    """Restore object from serialized data."""

def save_to_file(obj: Any, filepath: str | Path, format: str = "json") -> None:
    """Serialize and write to file."""

def load_from_file(filepath: str | Path, format: str = "json") -> Any:
    """Read from file and deserialize."""
```

### Requirements

- **`"json"` format:** Use `json.dumps(obj, ensure_ascii=False, indent=2)` and `json.loads(data)`. Open files in text mode (`"w"` / `"r"`).
- **`"pickle"` format:** Use `pickle.dumps(obj)` and `pickle.loads(data)`. Open files in binary mode (`"wb"` / `"rb"`).
- For unsupported format strings, raise `ValueError`.
- `filepath` should accept both `str` and `pathlib.Path`. Convert with `Path(filepath)` at the start of each function.

### Self-check

```python
from myagents.utils.serialization import save_to_file, load_from_file
import tempfile, os

data = {"agent": "SimpleAgent", "turns": 3, "history": ["hello", "world"]}

with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
    path = f.name

save_to_file(data, path)
loaded = load_from_file(path)
assert loaded == data
os.unlink(path)
print("Serialization OK")
```

---

## Step 3 — `utils/helpers.py`: Utility Functions

### What to build

Five small utility functions used throughout the framework:

**`format_time(timestamp=None, format_str="%Y-%m-%d %H:%M:%S") -> str`**
Returns the given datetime (or `datetime.now()` if `None`) formatted as a string.

**`validate_config(config: dict, required_keys: list) -> bool`**
Checks that all keys in `required_keys` are present in `config`. Raises `ValueError` with the list of missing keys if any are absent. Returns `True` otherwise.

**`safe_import(module_name: str, class_name: Optional[str] = None) -> Any`**
Wraps `importlib.import_module(module_name)` in a try/except. If `class_name` is given, returns `getattr(module, class_name)`. On failure, raises `ImportError` with a descriptive message including both the module and class name.

**`ensure_dir(path: Path) -> Path`**
Calls `path.mkdir(parents=True, exist_ok=True)` and returns the path. Use this anywhere you need to create directories without worrying about whether they already exist.

**`merge_dicts(dict1: dict, dict2: dict) -> dict`**
Deep-merges two dicts. If both dicts have the same key and both values are dicts, recurse. Otherwise, `dict2`'s value wins. Does NOT mutate either input — returns a new dict.

### Self-check

```python
from myagents.utils.helpers import validate_config, safe_import, merge_dicts

# validate_config
assert validate_config({"a": 1, "b": 2}, ["a", "b"]) == True
try:
    validate_config({"a": 1}, ["a", "b", "c"])
except ValueError as e:
    assert "b" in str(e) and "c" in str(e)

# safe_import
json_mod = safe_import("json")
assert json_mod.dumps({"x": 1}) == '{"x": 1}'
try:
    safe_import("nonexistent_module_xyz")
except ImportError as e:
    print(f"Safe import correctly failed: {e}")

# merge_dicts
a = {"x": 1, "nested": {"a": 1, "b": 2}}
b = {"y": 2, "nested": {"b": 99, "c": 3}}
result = merge_dicts(a, b)
assert result == {"x": 1, "y": 2, "nested": {"a": 1, "b": 99, "c": 3}}
print("Helpers OK")
```

---

## Step 4 — `utils/__init__.py`

```python
from .logging import setup_logger, get_logger
from .serialization import serialize_object, deserialize_object, save_to_file, load_from_file
from .helpers import format_time, validate_config, safe_import, ensure_dir, merge_dicts

__all__ = [
    "setup_logger", "get_logger",
    "serialize_object", "deserialize_object", "save_to_file", "load_from_file",
    "format_time", "validate_config", "safe_import", "ensure_dir", "merge_dicts",
]
```

### Self-check

```python
from myagents.utils import setup_logger, save_to_file, validate_config
print("Utils imports OK")
```

---

## Step 5 — `myagents/__init__.py`: The Public API Surface

This is the file users interact with. It re-exports the most important classes from every layer so users never need to know your internal file structure.

A user should be able to do any of these:

```python
from myagents import SimpleAgent, ReActAgent, LLMClient, ToolRegistry
from myagents import ReflectionAgent, PlanAndSolveAgent
from myagents import CalculatorTool, SearchTool, calculate
from myagents import ToolChain, AsyncToolExecutor
from myagents import Message, Config
```

### What to put in `__init__.py`

```python
from .version import __version__, __author__   # create version.py too (see below)

# Core
from .core.llm import LLMClient
from .core.config import Config
from .core.message import Message
from .core.exceptions import AgentFrameworkError, LLMError, AgentError, ConfigError, ToolError

# Agents
from .agents.simple_agent import SimpleAgent
from .agents.react_agent import ReActAgent
from .agents.reflection_agent import ReflectionAgent
from .agents.plan_solve_agent import PlanAndSolveAgent

# Tools
from .tools.registry import ToolRegistry, global_registry
from .tools.chain import ToolChain, ToolChainManager
from .tools.async_executor import AsyncToolExecutor
from .tools.builtin.calculator import CalculatorTool, calculate
from .tools.builtin.search import SearchTool, search

# Utils
from .utils.logging import setup_logger, get_logger
from .utils.serialization import save_to_file, load_from_file

__all__ = [
    "__version__", "__author__",
    "LLMClient", "Config", "Message",
    "AgentFrameworkError", "LLMError", "AgentError", "ConfigError", "ToolError",
    "SimpleAgent", "ReActAgent", "ReflectionAgent", "PlanAndSolveAgent",
    "ToolRegistry", "global_registry", "ToolChain", "ToolChainManager",
    "AsyncToolExecutor",
    "CalculatorTool", "calculate", "SearchTool", "search",
    "setup_logger", "get_logger", "save_to_file", "load_from_file",
]
```

### Also create `version.py`

```python
# myagents/version.py
__version__ = "0.1.0"
__author__ = "Your Name"
```

### Self-check

```python
import myagents

print(f"myagents v{myagents.__version__}")
print(dir(myagents))

# Spot-check key imports
assert hasattr(myagents, "SimpleAgent")
assert hasattr(myagents, "ReActAgent")
assert hasattr(myagents, "LLMClient")
assert hasattr(myagents, "ToolRegistry")
assert hasattr(myagents, "CalculatorTool")
print("Public API OK")
```

---

## Step 6 — `demo.py`: End-to-End Demo

Create `demo.py` at the project root (next to the `myagents/` folder). This script must demonstrate all four agents working with a real LLM.

### Requirements

- Call `load_dotenv()` at the top before any framework imports that might read env vars.
- Use `print()` headers to clearly separate each agent's output.
- Use a real, non-trivial input for each agent.
- The calculator tool must be registered and used in the ReAct demo.

### Template to implement

```python
# demo.py
from dotenv import load_dotenv
load_dotenv()

from myagents import (
    LLMClient,
    SimpleAgent, ReActAgent, ReflectionAgent, PlanAndSolveAgent,
    ToolRegistry, CalculatorTool,
)

def separator(title: str):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def demo_simple_agent(llm: LLMClient):
    separator("DEMO 1: SimpleAgent — Multi-turn Conversation")
    agent = SimpleAgent(
        name="assistant",
        llm=llm,
        system_prompt="You are a helpful assistant. Keep all answers under 3 sentences."
    )
    questions = [
        "My name is Alex and I'm studying computer science.",
        "What field am I studying?",
        "Recommend one book for someone in my field.",
    ]
    for q in questions:
        print(f"\nUser: {q}")
        response = agent.run(q)
        print(f"Agent: {response}")

def demo_react_agent(llm: LLMClient):
    separator("DEMO 2: ReActAgent — Tool-Using Reasoning Loop")
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    agent = ReActAgent(
        name="calculator_agent",
        llm=llm,
        tool_registry=registry,
        max_steps=6,
    )
    question = "What is (17 * 23) + (144 / 12)? Show your work."
    print(f"\nQuestion: {question}")
    answer = agent.run(question)
    print(f"\nFinal Answer: {answer}")

def demo_reflection_agent(llm: LLMClient):
    separator("DEMO 3: ReflectionAgent — Self-Critique and Refinement")
    agent = ReflectionAgent(
        name="writer",
        llm=llm,
        max_iterations=2,
    )
    task = "Write a 2-sentence explanation of what a neural network is, suitable for a high school student."
    print(f"\nTask: {task}")
    result = agent.run(task)
    print(f"\nFinal Result:\n{result}")

def demo_plan_solve_agent(llm: LLMClient):
    separator("DEMO 4: PlanAndSolveAgent — Decompose and Execute")
    agent = PlanAndSolveAgent(
        name="planner",
        llm=llm,
    )
    question = "Compare Python and JavaScript: list 3 key differences and which is better for backend web development."
    print(f"\nQuestion: {question}")
    answer = agent.run(question)
    print(f"\nFinal Answer:\n{answer}")

if __name__ == "__main__":
    print("Initializing LLM client...")
    llm = LLMClient()
    print(f"Using provider: {llm.provider}, model: {llm.model}")

    demo_simple_agent(llm)
    demo_react_agent(llm)
    demo_reflection_agent(llm)
    demo_plan_solve_agent(llm)

    separator("ALL DEMOS COMPLETE")
```

Run it:

```bash
python demo.py
```

All four demos must complete without exceptions.

---

## ✅ Phase 4 Complete Checklist

- [ ] `utils/logging.py` — `setup_logger` with duplicate-handler guard, `get_logger`
- [ ] `utils/serialization.py` — JSON and pickle save/load
- [ ] `utils/helpers.py` — all 5 utility functions
- [ ] `utils/__init__.py` — all utils exported
- [ ] `myagents/version.py` — version and author set
- [ ] `myagents/__init__.py` — all major classes importable from top-level
- [ ] `demo.py` — all 4 demos run end-to-end without errors
- [ ] `python test_import.py` from Phase 0 now shows all exported names

---

## ✅ Full Project Final Checklist

Run through this before submitting:

|     | Requirement                                                              |
| --- | ------------------------------------------------------------------------ |
| [ ] | `python demo.py` completes all 4 demos without exceptions                |
| [ ] | No use of `eval()` anywhere (search with `grep -r "eval(" myagents/`)    |
| [ ] | All files have type hints on every function signature                    |
| [ ] | `core/`, `tools/`, `agents/`, `utils/` each have a working `__init__.py` |
| [ ] | `ReActAgent` `current_history` resets at start of each `run()` call      |
| [ ] | `CalculatorTool` uses AST traversal, not `eval()`                        |
| [ ] | `Planner.plan()` uses `ast.literal_eval`, not `eval()`                   |
| [ ] | `get_history()` returns a copy, not the internal list                    |
| [ ] | All tool errors are returned as strings, not raised as exceptions        |
| [ ] | `.env` is in `.gitignore`                                                |
| [ ] | `requirements.txt` is present with pinned versions                       |

---

## Bonus Challenges

If you finish early, pick any of these:

**+1 — Async Tool Demo**
Add a `demo_async_tools()` function to `demo.py` that runs at least 3 calculator expressions in parallel using `AsyncToolExecutor` and prints all results.

**+2 — Custom Agent Paradigm**
Implement a 5th agent called `ChainOfThoughtAgent` that forces the LLM to reason step-by-step by prefixing every prompt with `"Let's think step by step:"` and parsing numbered reasoning steps before giving a final answer.

**+3 — Function Calling Schema**
Add a method `to_openai_function_schema() -> dict` to the `Tool` base class that returns a JSON schema dict in OpenAI function-calling format, e.g.:

```json
{
  "name": "python_calculator",
  "description": "...",
  "parameters": {
    "type": "object",
    "properties": {"input": {"type": "string", "description": "..."}},
    "required": ["input"]
  }
}
```

**+4 — Pytest Suite**
Write `tests/test_core.py` and `tests/test_tools.py` using `pytest` and `unittest.mock`. Mock the LLM client so tests do not make real API calls. Achieve at least 70% code coverage (`pytest --cov=myagents`).

**+5 — Multi-Agent Orchestration**
Create a `MultiAgentOrchestrator` class that holds multiple named agents and routes tasks to them based on simple keyword matching or LLM-based routing. One agent should be callable as a "tool" by another.
