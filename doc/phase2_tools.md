# Phase 2 — Build the Tools Layer

> **Goal:** Build a complete, pluggable tool system with a registry, pipeline chain, async executor, and two built-in tools.
> **Time:** 3–4 hours
> **Depends on:** Phase 1 complete
> **Next:** [phase3_agents.md](./phase3_agents.md)

You will build these files in order:

```
tools/base.py              ← Step 1
tools/registry.py          ← Step 2
tools/chain.py             ← Step 3
tools/async_executor.py    ← Step 4
tools/builtin/calculator.py ← Step 5
tools/builtin/search.py    ← Step 6
tools/__init__.py          ← Step 7
tools/builtin/__init__.py  ← Step 7
```

---

## Step 1 — `tools/base.py`: The Abstract Tool Contract

### Why define an abstract base?

You want every tool — whether it searches the web, runs code, queries a database, or reads files — to have the exact same interface. That way, the registry can call any tool without knowing what it does. This is the **Strategy pattern**.

### What to build

Two classes: a parameter schema model and an abstract tool class.

**`ToolParameter`** — describes one input parameter a tool expects:

```
ToolParameter
  name: str              e.g. "query"
  type: str              e.g. "string", "integer", "boolean"
  description: str       e.g. "The search query to run"
  required: bool         default True
  default: Any           default None
```

Use Pydantic `BaseModel` for this. This schema will later be used to generate documentation and validate inputs.

**`Tool`** — the abstract base all tools inherit from:

| Method | Abstract? | What it does |
| ------ | --------- | ------------ |
| `__init__(name, description)` | No | Stores name and description |
| `run(parameters: dict) -> str` | **Yes** | Execute the tool, always returns a string |
| `get_parameters() -> List[ToolParameter]` | **Yes** | Declare what inputs this tool expects |
| `validate_parameters(parameters: dict) -> bool` | No | Check all required params are present |
| `to_dict() -> dict` | No | Serialize the tool to a dict (for docs/function-calling) |

### Exact requirements for `run()`

- Input is **always** `Dict[str, Any]` — a dict of named parameters.
- Output is **always** `str` — even errors should be returned as strings, not raised as exceptions.
- This uniformity is what makes the registry work without knowing tool types.

### Exact requirements for `validate_parameters()`

Implement it concretely in the base class — no need to override in subclasses:

```python
def validate_parameters(self, parameters: dict) -> bool:
    required = [p.name for p in self.get_parameters() if p.required]
    return all(p in parameters for p in required)
```

### Exact requirements for `to_dict()`

Return a dict shaped like:

```python
{
    "name": self.name,
    "description": self.description,
    "parameters": [p.model_dump() for p in self.get_parameters()]
}
```

### Self-check

```python
from myagents.tools.base import Tool, ToolParameter

# A minimal concrete tool for testing:
class EchoTool(Tool):
    def __init__(self):
        super().__init__(name="echo", description="Echoes the input back")

    def run(self, parameters: dict) -> str:
        return parameters.get("input", "")

    def get_parameters(self):
        return [ToolParameter(name="input", type="string", description="text to echo")]

tool = EchoTool()
assert tool.run({"input": "hello"}) == "hello"
assert tool.validate_parameters({"input": "test"}) == True
assert tool.validate_parameters({}) == False
assert tool.to_dict()["name"] == "echo"
print("Tool base OK")
```

---

## Step 2 — `tools/registry.py`: The Tool Registry

### What it does

The registry is the **central directory** of all tools available to agents. It stores tools and dispatches execution by name. Agents don't call tools directly — they ask the registry to run a tool by name. This decouples agents from tool implementations entirely.

### Two registration styles

You must support both:

**Style 1 — Register a `Tool` object (recommended, full-featured):**
```python
registry.register_tool(CalculatorTool())
```

**Style 2 — Register a plain function (quick, for simple cases):**
```python
registry.register_function("greet", "Says hello", lambda name: f"Hello, {name}!")
```

### Internal storage

Use two separate dicts:
```python
self._tools: dict[str, Tool] = {}
self._functions: dict[str, dict] = {}
# _functions stores: {"description": str, "func": Callable[[str], str]}
```

### Methods to implement

**`register_tool(tool: Tool) -> None`**
- Warn (print) if the name already exists — then overwrite.
- Store in `self._tools`.

**`register_function(name: str, description: str, func: Callable[[str], str]) -> None`**
- Same overwrite warning behavior.
- Store in `self._functions`.

**`execute_tool(name: str, input_text: str) -> str`**
- Look in `self._tools` first, then `self._functions`.
- For `Tool` objects: call `tool.run({"input": input_text})`.
- For functions: call `func(input_text)`.
- **Wrap every execution in try/except.** If anything fails, return an error string — do NOT let exceptions propagate. The agent loop must continue even if a tool crashes.
- If no tool found: return `f"Error: tool '{name}' not found"`.

**`get_tools_description() -> str`**
- This is injected into agent prompts so the LLM knows what tools exist.
- Format each tool as: `- tool_name: description`
- Return all tools (both `_tools` and `_functions`) joined by newlines.
- If no tools registered, return `"No tools available"`.

**`unregister(name: str) -> None`**, **`list_tools() -> list[str]`**, **`clear() -> None`** — implement these as well, they are useful for testing.

### The global singleton

At the bottom of the file, after the class definition, add:

```python
global_registry = ToolRegistry()
```

This creates one shared instance at module load time. Any code that imports `global_registry` gets the same object, making it easy to register tools from anywhere in the codebase.

### Self-check

```python
from myagents.tools.registry import ToolRegistry

r = ToolRegistry()

# Test function registration
r.register_function("reverse", "Reverses a string", lambda s: s[::-1])
assert r.execute_tool("reverse", "hello") == "olleh"

# Test error handling — tool that always crashes
r.register_function("broken", "Always fails", lambda s: 1/0)
result = r.execute_tool("broken", "test")
assert "error" in result.lower() or "Error" in result   # should not raise

# Test missing tool
result = r.execute_tool("nonexistent", "test")
assert "not found" in result.lower()

# Test description
r.register_function("add", "Adds two numbers", lambda s: str(eval(s)))  # eval OK in test only
desc = r.get_tools_description()
assert "reverse" in desc
assert "add" in desc
print("Registry OK")
```

---

## Step 3 — `tools/chain.py`: Sequential Tool Pipeline

### Concept

A `ToolChain` is a named sequence of tool-execution steps. The output of one step can be fed as input to the next step via **template variable substitution**. This lets you compose tools into pipelines without writing custom glue code.

Example: a "research" chain might run `search[{input}]` then pass the result to `summarize[{search_result}]`.

### What to build: `ToolChain`

**Constructor:**
```python
def __init__(self, name: str, description: str):
    self.name = name
    self.description = description
    self.steps: List[Dict[str, Any]] = []
```

**`add_step(tool_name, input_template, output_key=None)`**

Appends a step dict to `self.steps`:
```python
{
    "tool_name": tool_name,
    "input_template": input_template,  # e.g. "{search_result}"
    "output_key": output_key or f"step_{len(self.steps)}_result"
}
```

**`execute(registry: ToolRegistry, input_data: str, context: dict = None) -> str`**

This is the core method. Walk through all steps in order:

1. Initialize context: `context = context or {}; context["input"] = input_data`
2. For each step:
   - Format the template: `actual_input = step["input_template"].format(**context)` — wrap in try/except `KeyError` (missing variable in template → return error string)
   - Execute: `result = registry.execute_tool(step["tool_name"], actual_input)` — wrap in try/except
   - Store: `context[step["output_key"]] = result`
   - Update `final_result = result`
3. Return `final_result`

If `self.steps` is empty, return an error string immediately.

### What to build: `ToolChainManager`

A manager that holds multiple named chains and dispatches by name:

```python
class ToolChainManager:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: Dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain) -> None: ...
    def execute_chain(self, chain_name: str, input_data: str, context: dict = None) -> str: ...
    def list_chains(self) -> List[str]: ...
```

### Self-check

```python
from myagents.tools.registry import ToolRegistry
from myagents.tools.chain import ToolChain, ToolChainManager

r = ToolRegistry()
r.register_function("shout", "Uppercases input", lambda s: s.upper())
r.register_function("wrap", "Wraps in brackets", lambda s: f"[{s}]")

chain = ToolChain(name="process", description="Shout then wrap")
chain.add_step("shout", "{input}", "shouted")
chain.add_step("wrap", "{shouted}", "final")

result = chain.execute(r, "hello")
assert result == "[HELLO]", f"Got: {result}"
print(f"Chain OK: {result}")
```

---

## Step 4 — `tools/async_executor.py`: Parallel Tool Execution

### Why this matters

If an agent needs to search for three things simultaneously, running them sequentially wastes time. The async executor runs multiple tools at the same time using `asyncio` + a thread pool.

### Key concept: bridging sync and async

Your tool implementations (like HTTP API calls) are **synchronous** functions. Python's `asyncio` is single-threaded and cannot parallelize synchronous work directly. The bridge is `loop.run_in_executor()`:

```python
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(self.thread_pool, sync_function)
```

This runs `sync_function` in a thread from the pool, freeing the event loop to handle other coroutines while waiting.

### What to build: `AsyncToolExecutor`

```python
class AsyncToolExecutor:
    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
```

**`async execute_tool_async(tool_name, input_data) -> str`**

Run a single tool asynchronously using `run_in_executor`. Catch all exceptions and return error strings.

**`async execute_tools_parallel(tasks: List[Dict[str, str]]) -> List[Dict[str, Any]]`**

- `tasks` is a list of `{"tool_name": "...", "input_data": "..."}` dicts.
- Create a coroutine for each task using `execute_tool_async`.
- Await them and collect results.
- Return a list of result dicts:
  ```python
  {"task_id": i, "tool_name": ..., "input_data": ..., "result": ..., "status": "success"|"error"}
  ```

**`async execute_tools_batch(tool_name, input_list) -> List[Dict]`**

Convenience: runs the same tool against multiple inputs in parallel. Internally calls `execute_tools_parallel`.

**`close()`, `__enter__`, `__exit__`** — implement context manager protocol so the thread pool is properly shut down.

### Module-level convenience functions

```python
async def run_parallel_tools(registry, tasks, max_workers=4) -> List[Dict]: ...
async def run_batch_tool(registry, tool_name, input_list, max_workers=4) -> List[Dict]: ...

# Sync wrappers (for use outside async contexts):
def run_parallel_tools_sync(...) -> List[Dict]:
    return asyncio.run(run_parallel_tools(...))

def run_batch_tool_sync(...) -> List[Dict]:
    return asyncio.run(run_batch_tool(...))
```

### Self-check

```python
import asyncio
from myagents.tools.registry import ToolRegistry
from myagents.tools.async_executor import AsyncToolExecutor

r = ToolRegistry()
r.register_function("double", "Doubles input", lambda s: str(int(s) * 2))

async def test():
    async with AsyncToolExecutor(r, max_workers=2) as executor:
        results = await executor.execute_tools_parallel([
            {"tool_name": "double", "input_data": "3"},
            {"tool_name": "double", "input_data": "7"},
            {"tool_name": "double", "input_data": "10"},
        ])
    assert len(results) == 3
    assert results[0]["result"] == "6"
    assert all(r["status"] == "success" for r in results)
    print("Async executor OK")

asyncio.run(test())
```

---

## Step 5 — `tools/builtin/calculator.py`: Safe Math Evaluator

### The core challenge

You need to evaluate math expressions like `"2 + 3 * sqrt(16)"` from user/LLM input. The obvious solution — `eval("2 + 3 * sqrt(16)")` — is a **critical security vulnerability**. Anyone could pass `"__import__('os').system('rm -rf /')"` and you'd execute it.

The safe alternative: parse the expression into an **Abstract Syntax Tree (AST)** using Python's built-in `ast` module, then walk the tree manually, only evaluating node types you explicitly whitelist.

### How the AST approach works

```python
import ast

expr = "2 + 3 * 4"
tree = ast.parse(expr, mode='eval')
# tree.body is an ast.BinOp node representing the addition
# tree.body.left is ast.Constant(value=2)
# tree.body.right is ast.BinOp representing 3 * 4
# ...and so on recursively
```

You write a recursive `_eval_node(node)` that handles exactly these node types:

| AST Node Type | What it represents | What you return |
| ------------- | ------------------ | --------------- |
| `ast.Constant` | A number literal like `3` or `2.5` | `node.value` |
| `ast.Num` | Same, for Python < 3.8 compatibility | `node.n` |
| `ast.BinOp` | Binary operation: `left op right` | `OPERATORS[type(node.op)](eval(left), eval(right))` |
| `ast.UnaryOp` | Unary op: e.g. `-x` | `OPERATORS[type(node.op)](eval(operand))` |
| `ast.Call` | Function call: `sqrt(x)` | Look up name in `FUNCTIONS` whitelist, evaluate args, call |
| `ast.Name` | A bare name like `pi` or `e` | Look up in `FUNCTIONS` dict (for math constants) |
| Anything else | Unknown/unsafe | `raise ValueError(f"Unsupported: {type(node)}")` |

### Whitelists to define

```python
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,   # unary minus
}

FUNCTIONS = {
    'abs': abs, 'round': round, 'max': max, 'min': min,
    'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
    'tan': math.tan, 'log': math.log, 'exp': math.exp,
    'pi': math.pi, 'e': math.e,
}
```

### What to build: `CalculatorTool`

A `Tool` subclass with:
- `name = "python_calculator"`
- `description` explaining it evaluates math expressions
- `run(parameters)` that:
  1. Gets `expression = parameters.get("input", "") or parameters.get("expression", "")`
  2. Calls `ast.parse(expression, mode='eval')`
  3. Calls `self._eval_node(tree.body)`
  4. Returns the result as a string
  5. Catches all exceptions and returns error strings — **never raises**
- `_eval_node(node)` — the recursive evaluator described above
- `get_parameters()` returning one `ToolParameter` for `"input"`

### Also implement a convenience function

```python
def calculate(expression: str) -> str:
    return CalculatorTool().run({"input": expression})
```

### Self-check

```python
from myagents.tools.builtin.calculator import CalculatorTool, calculate

tool = CalculatorTool()

# Basic arithmetic
assert tool.run({"input": "2 + 3"}) == "5"
assert tool.run({"input": "10 / 4"}) == "2.5"
assert tool.run({"input": "2 ** 8"}) == "256"

# Math functions
result = tool.run({"input": "sqrt(16)"})
assert result == "4.0", f"Got: {result}"

# Security: should NOT execute arbitrary code
result = tool.run({"input": "__import__('os').getcwd()"})
assert "error" in result.lower() or "unsupported" in result.lower()
print("Calculator OK, security check passed")

# Convenience function
assert calculate("3 * 7") == "21"
```

---

## Step 6 — `tools/builtin/search.py`: Web Search Tool

### Overview

A tool that searches the web. Supports two backends (Tavily AI search and SerpApi Google search) and a "hybrid" mode that tries Tavily first, falls back to SerpApi.

### Key design: check availability at `__init__`, not at `run()`

You must determine whether each backend is available when the tool is first created — not when it's first used. This gives early, clear error messages at startup rather than mysterious failures mid-run.

Availability requires TWO things to be true:
1. The API key is configured (env var or constructor arg)
2. The Python library is installed

Check both like this:
```python
if self.tavily_key:
    try:
        from tavily import TavilyClient
        self.tavily_client = TavilyClient(api_key=self.tavily_key)
        self.available_backends.append("tavily")
    except ImportError:
        print("Warning: tavily not installed. Run: pip install tavily-python")
```

### What to build: `SearchTool`

**Constructor:**
```python
def __init__(self, backend: str = "hybrid", tavily_key=None, serpapi_key=None):
    super().__init__(name="search", description="Web search tool...")
    self.backend = backend
    self.tavily_key = tavily_key or os.getenv("TAVILY_API_KEY")
    self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
    self.available_backends = []
    self._setup_backends()   # checks availability
```

**`run(parameters)` dispatch logic:**
```
if backend == "hybrid":  → _search_hybrid(query)
if backend == "tavily":  → _search_tavily(query)
if backend == "serpapi": → _search_serpapi(query)
```

**`_search_hybrid(query)`:**
- If tavily is available: try it, return result
- If tavily fails or unavailable: try serpapi
- If both fail: return a helpful error message listing what's missing

**`_search_tavily(query)`** using the Tavily client:
```python
response = self.tavily_client.search(query=query, search_depth="basic",
                                     include_answer=True, max_results=3)
# Format: answer + top 3 results with title, snippet, URL
```

**`_search_serpapi(query)`** using SerpApi:
```python
from serpapi import SerpApiClient
params = {"engine": "google", "q": query, "api_key": self.serpapi_key}
results = SerpApiClient(params).get_dict()
# Format: answer_box (if present) + top 3 organic results
```

**`_get_error_message()`:** Returns a clear, actionable message showing what's missing and how to fix it (which env var to set, which package to install).

**`get_parameters()`:** Returns one `ToolParameter` for `"input"` (the search query).

### Convenience functions to add at module level

```python
def search(query: str, backend: str = "hybrid") -> str: ...
def search_tavily(query: str) -> str: ...
def search_serpapi(query: str) -> str: ...
```

### Self-check

```python
from myagents.tools.builtin.search import SearchTool

# Even without API keys, the tool should instantiate without crashing
tool = SearchTool()
result = tool.run({"input": "test query"})
# If no backends available, should get a helpful error message, not an exception
print(f"Search result (may be error if no keys): {result[:100]}")

# With a valid key configured in .env:
# from dotenv import load_dotenv; load_dotenv()
# tool = SearchTool()
# result = tool.run({"input": "Python programming language"})
# assert "Python" in result
```

---

## Step 7 — Wire Up `__init__.py` Files

### `tools/builtin/__init__.py`

```python
from .calculator import CalculatorTool, calculate
from .search import SearchTool, search

__all__ = ["CalculatorTool", "calculate", "SearchTool", "search"]
```

### `tools/__init__.py`

```python
from .base import Tool, ToolParameter
from .registry import ToolRegistry, global_registry
from .chain import ToolChain, ToolChainManager
from .async_executor import AsyncToolExecutor, run_parallel_tools_sync, run_batch_tool_sync
from .builtin.calculator import CalculatorTool, calculate
from .builtin.search import SearchTool, search

__all__ = [
    "Tool", "ToolParameter",
    "ToolRegistry", "global_registry",
    "ToolChain", "ToolChainManager",
    "AsyncToolExecutor", "run_parallel_tools_sync", "run_batch_tool_sync",
    "CalculatorTool", "calculate",
    "SearchTool", "search",
]
```

### Self-check

```python
from myagents.tools import (
    Tool, ToolParameter, ToolRegistry, global_registry,
    ToolChain, ToolChainManager, AsyncToolExecutor,
    CalculatorTool, calculate, SearchTool
)
print("All tool imports OK")
```

---

## ✅ Phase 2 Complete Checklist

- [ ] `tools/base.py` — `ToolParameter` and abstract `Tool` with `run`, `get_parameters`, `validate_parameters`, `to_dict`
- [ ] `tools/registry.py` — dual storage, `execute_tool` catches all errors, `get_tools_description`, `global_registry` singleton
- [ ] `tools/chain.py` — `ToolChain` with template variable substitution, `ToolChainManager`
- [ ] `tools/async_executor.py` — `ThreadPoolExecutor` bridge, parallel + batch execution, sync wrappers
- [ ] `tools/builtin/calculator.py` — AST traversal, no `eval()`, all errors returned as strings
- [ ] `tools/builtin/search.py` — availability check at `__init__`, hybrid fallback, helpful error messages
- [ ] Both `__init__.py` files export everything
- [ ] All self-checks pass

**Next → [Phase 3: Build the Agent Implementations](./phase3_agents.md)**
