# CortexGraph: A Comprehensive Framework Specification Guide

### For Stanford CS Students — Build Your Own Multi-Agent Framework from Scratch

---

> **Purpose of This Document**
> This guide is a complete technical specification of the **CortexGraph** (internally named *HelloAgents*) multi-agent AI framework. You will deeply understand every design decision, every abstraction, and every line of code that makes this system work. By the end, you will have enough understanding to **re-implement a similar framework entirely on your own**.

---

## Table of Contents

1. [Project Overview & Design Philosophy](#1-project-overview--design-philosophy)
2. [Architecture Map](#2-architecture-map)
3. [Module-by-Module Deep Dive](#3-module-by-module-deep-dive)
   - 3.1 [Core Layer](#31-core-layer)
   - 3.2 [Agent Layer](#32-agent-layer)
   - 3.3 [Tools Layer](#33-tools-layer)
   - 3.4 [Utils Layer](#34-utils-layer)
4. [Data Flow & Execution Lifecycle](#4-data-flow--execution-lifecycle)
5. [Design Patterns Used](#5-design-patterns-used)
6. [Key Algorithms & Logic Explained](#6-key-algorithms--logic-explained)
7. [Configuration & Environment Management](#7-configuration--environment-management)
8. [Extension Points — How to Add New Agents and Tools](#8-extension-points--how-to-add-new-agents-and-tools)
9. [Implementation Specification for Students](#9-implementation-specification-for-students)
10. [Grading Rubric & Milestone Checklist](#10-grading-rubric--milestone-checklist)

---

## 1. Project Overview & Design Philosophy

### 1.1 What is CortexGraph?

CortexGraph is a lightweight, extensible Python framework for building **AI agents** — software systems that use large language models (LLMs) to reason and act. It is built directly on top of OpenAI's Python SDK (the `openai` library) and imposes **no heavy dependency** on frameworks like LangChain or LlamaIndex. This is a deliberate architectural choice: students can read every line and understand every abstraction.

The system provides:

- A **unified LLM client** that works across 10+ model providers without changing your agent code.
- **Four agent paradigms** (Simple, ReAct, Reflection, Plan-and-Solve) each representing a well-studied approach to building reasoning AI systems.
- A **pluggable tool system** with a registry, chains, and async execution.
- A clean **utils layer** for logging, serialization, and helpers.

### 1.2 Core Design Principles

| Principle                        | How It Is Applied                                                                             |
| -------------------------------- | --------------------------------------------------------------------------------------------- |
| **Parameters over environment**  | Every class accepts explicit arguments; env vars are fallback only.                           |
| **Streaming-first**              | The primary LLM call method (`think`) is a generator that yields tokens.                      |
| **Composition over inheritance** | Agents hold an LLM instance; they don't inherit from it.                                      |
| **Single responsibility**        | Each file has exactly one job: `config.py` manages config, `message.py` models messages, etc. |
| **Fail loudly**                  | A custom exception hierarchy catches and re-raises errors with context.                       |
| **Extensibility**                | Both agents and tools are abstract base classes — extension is the intended use.              |

### 1.3 Supported LLM Providers

The framework supports the following model providers through a unified adapter:

```
openai, deepseek, qwen, modelscope, kimi, zhipu, ollama, vllm, local, auto
```

All providers expose an **OpenAI-compatible REST API**, so the same `openai.OpenAI` SDK client can call any of them — only the `base_url` and `api_key` differ.

---

## 2. Architecture Map

```
CortexGraph/
├── __init__.py              # Public API surface — re-exports everything
├── version.py               # Version metadata
│
├── core/                    # ← Layer 1: Foundation primitives
│   ├── __init__.py
│   ├── config.py            # Pydantic settings model
│   ├── message.py           # Message data model (user/assistant/system/tool)
│   ├── exceptions.py        # Exception hierarchy
│   ├── llm.py               # Unified LLM client (the "brain connector")
│   └── agent.py             # Abstract base class for all agents
│
├── agents/                  # ← Layer 2: Agent paradigm implementations
│   ├── __init__.py
│   ├── simple_agent.py      # Stateful conversational agent
│   ├── react_agent.py       # Reason + Act loop with tool calling
│   ├── reflection_agent.py  # Self-critique and iterative refinement
│   └── plan_solve_agent.py  # Two-phase: plan decomposition + execution
│
├── tools/                   # ← Layer 3: Tool ecosystem
│   ├── __init__.py
│   ├── base.py              # Abstract Tool + ToolParameter models
│   ├── registry.py          # Tool registration and dispatch
│   ├── chain.py             # Sequential tool pipeline (ToolChain)
│   ├── async_executor.py    # Parallel async tool execution
│   └── builtin/
│       ├── __init__.py
│       ├── search.py        # Web search (Tavily + SerpApi hybrid)
│       └── calculator.py    # Safe mathematical expression evaluator
│
├── utils/                   # ← Layer 4: Cross-cutting utilities
│   ├── __init__.py
│   ├── logging.py           # Logger factory
│   ├── serialization.py     # JSON / Pickle serialization helpers
│   └── helpers.py           # Time, validation, import, path utilities
│
└── doc/                     # ← Documentation (this file lives here)
```

### Dependency Graph (what imports what)

```
utils  ←──────────────────── (no dependencies on other layers)
  ↑
core   ←── depends on: utils (optional), openai SDK, pydantic
  ↑
tools  ←── depends on: core (exceptions, base types)
  ↑
agents ←── depends on: core + tools
  ↑
__init__.py  ←── re-exports from all layers
```

**Key insight**: The dependency only flows upward. Utils knows nothing about agents. Core knows nothing about tools. This enforces clean separation and makes each layer independently testable.

---

## 3. Module-by-Module Deep Dive

### 3.1 Core Layer

#### 3.1.1 `core/exceptions.py` — Exception Hierarchy

```python
HelloAgentsException          # base
├── LLMException              # errors from LLM API calls
├── AgentException            # errors within agent logic
├── ConfigException           # missing or invalid config
└── ToolException             # errors during tool execution
```

**Why this matters**: Rather than letting raw Python exceptions (like `openai.APIError` or `ValueError`) bubble up unintelligibly to users, every module wraps exceptions in semantically meaningful types. This is a standard software engineering practice called **exception translation**.

**Design decision**: All custom exceptions inherit from a single `HelloAgentsException` so callers can either catch the specific subtype or catch everything with one `except HelloAgentsException`.

---

#### 3.1.2 `core/message.py` — The Message Model

```python
MessageRole = Literal["user", "assistant", "system", "tool"]

class Message(BaseModel):
    content: str
    role: MessageRole
    timestamp: datetime       # auto-set to now()
    metadata: Optional[Dict]  # arbitrary extra data

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}
```

**What this models**: The OpenAI Chat Completions API communicates via a list of message objects, each with a `role` and `content`. This class is the in-memory representation of those objects.

**Why Pydantic?** Pydantic `BaseModel` provides automatic type validation, clear field declarations, and a `.dict()` method. It also forces you to think carefully about your data schema upfront.

**The `to_dict()` method**: This converts a `Message` back into the raw dict format (`{"role": ..., "content": ...}`) that the OpenAI API expects. Note that `timestamp` and `metadata` are **not** included — they are framework-internal and must not be sent to the API.

---

#### 3.1.3 `core/config.py` — Configuration Management

```python
class Config(BaseModel):
    default_model: str = "gpt-3.5-turbo"
    default_provider: str = "openai"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    debug: bool = False
    log_level: str = "INFO"
    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        """Construct config from environment variables."""
```

**Purpose**: Centralizes all runtime-tunable parameters. Every agent receives a `Config` instance, so tuning behavior is done in one place rather than scattered across constructors.

**`from_env()` class method**: This is a **factory method** pattern — it constructs a `Config` from environment variables instead of requiring the caller to read the environment themselves. Note that `from_env()` only reads a subset of variables (debug, log_level, temperature, max_tokens). LLM provider credentials are handled separately in `HelloAgentsLLM`.

---

#### 3.1.4 `core/llm.py` — The Unified LLM Client

This is the most complex file in the framework. It solves a real practical problem: students want to use DeepSeek, Qwen, Ollama, or OpenAI without changing their agent code.

**Constructor signature:**

```python
HelloAgentsLLM(
    model=None,          # falls back to LLM_MODEL_ID env var
    api_key=None,        # falls back to provider-specific env vars
    base_url=None,       # falls back to LLM_BASE_URL env var
    provider=None,       # auto-detected if not given
    temperature=0.7,
    max_tokens=None,
    timeout=None,        # falls back to LLM_TIMEOUT env var, default 60s
)
```

**Initialization flow (4 steps):**

```
Step 1: Set self.model, temperature, max_tokens, timeout from args/env
Step 2: _auto_detect_provider() → determine which service to call
Step 3: _resolve_credentials() → get api_key + base_url for that provider
Step 4: _create_client() → instantiate openai.OpenAI(api_key, base_url, timeout)
```

**Provider auto-detection logic (`_auto_detect_provider`):**

The method checks three things in order:

1. **Specific environment variables** (e.g., `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`) — most definitive signal
2. **API key format** (e.g., keys starting with `ms-` → ModelScope; value `"ollama"` → Ollama)
3. **base_url domain** (e.g., `api.deepseek.com` → deepseek; `localhost:11434` → ollama)

If nothing matches, returns `"auto"` — a generic pass-through mode.

**Credential resolution (`_resolve_credentials`):**

For each provider, there is a prioritized fallback chain:

```
explicit arg → provider-specific env var → generic LLM_API_KEY env var → hardcoded default
```

For example, for Ollama:

```python
api_key = api_key or os.getenv("OLLAMA_API_KEY") or os.getenv("LLM_API_KEY") or "ollama"
base_url = base_url or os.getenv("OLLAMA_HOST") or os.getenv("LLM_BASE_URL") or "http://localhost:11434/v1"
```

**The three call methods:**

| Method                              | Returns         | Streaming       | Use Case                                |
| ----------------------------------- | --------------- | --------------- | --------------------------------------- |
| `think(messages, temperature)`      | `Iterator[str]` | Yes (generator) | Primary method; real-time output        |
| `invoke(messages, **kwargs)`        | `str`           | No              | When you need the full response at once |
| `stream_invoke(messages, **kwargs)` | `Iterator[str]` | Yes             | Alias for `think`; backward compat      |

**Why streaming is the default (`think`)**: Streaming (`stream=True` in the API call) allows the framework to yield tokens as they arrive. This means an agent can display reasoning in real-time as the model generates it, which is essential for interactive use cases.

**The streaming implementation:**

```python
response = self._client.chat.completions.create(..., stream=True)
for chunk in response:
    content = chunk.choices[0].delta.content or ""
    if content:
        print(content, end="", flush=True)
        yield content
```

Each `chunk` has a `delta` containing the incremental new text. The `or ""` handles the `None` case for chunks that carry metadata but no text.

---

#### 3.1.5 `core/agent.py` — The Abstract Agent Base Class

```python
class Agent(ABC):
    def __init__(self, name, llm, system_prompt=None, config=None):
        self.name = name
        self.llm = llm                    # composition, not inheritance
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = [] # conversation memory

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        pass

    def add_message(self, message: Message): ...
    def clear_history(self): ...
    def get_history(self) -> list[Message]: ...
```

**Key design decisions:**

1. **`run()` is abstract**: Every concrete agent *must* implement it. This is the **Template Method** pattern's interface contract. The base class defines what an agent must do (accept text, return text) without specifying how.

2. **`_history` is a list of `Message` objects**: This gives every agent persistent short-term memory across multiple `run()` calls. The leading underscore signals it is private — subclasses should use `add_message()` to write to it.

3. **`get_history()` returns a copy**: `self._history.copy()` prevents external callers from mutating the internal list — a defensive programming pattern.

4. **`config or Config()`**: If no config is passed, a default `Config()` is constructed. This makes the config optional without requiring callers to know about it.

---

### 3.2 Agent Layer

All four agent implementations live in `agents/`. They all inherit from `Agent` and must implement `run()`.

#### 3.2.1 `SimpleAgent` — Stateful Conversational Agent

**Concept**: The simplest possible agent. It builds a message list from the system prompt, history, and current input, calls the LLM, saves the exchange, and returns the response.

**`run()` logic:**

```
messages = []
if system_prompt: messages.append({"role": "system", "content": system_prompt})
for msg in self._history: messages.append(msg.to_dict())
messages.append({"role": "user", "content": input_text})

response = self.llm.invoke(messages)

self.add_message(Message(input_text, "user"))
self.add_message(Message(response, "assistant"))
return response
```

**The conversation memory mechanism**: Because `_history` is accumulated across calls, each new `run()` call passes all previous turns to the LLM. This gives the agent **multi-turn conversational memory** without any external storage.

**`stream_run()`**: A separate method that calls `self.llm.stream_invoke()` and yields chunks. It accumulates chunks into `full_response` before saving to history (because history must store the complete text).

---

#### 3.2.2 `ReActAgent` — Reasoning and Acting Loop

**Concept**: Based on the 2022 paper *"ReAct: Synergizing Reasoning and Acting in Language Models"* (Yao et al.). The agent alternates between **thinking** (generating reasoning traces) and **acting** (calling tools), iterating until it can provide a `Finish` action.

**The ReAct prompt template structure:**

```
System context:
  - Available tools (from registry)
  - Output format rules: Thought: ... / Action: toolname[input]
  - Execution history (Thought/Action/Observation log)
  - The current question

LLM output:
  Thought: <reasoning>
  Action: toolname[input]   OR   Finish[final answer]
```

**`run()` control loop:**

```python
while current_step < self.max_steps:
    # 1. Build prompt (inject tools + history + question)
    # 2. Call LLM → get response_text
    # 3. Parse: extract (thought, action) via regex
    # 4. If action == Finish → extract answer, return
    # 5. Else: parse tool_name + tool_input from action
    # 6. Execute: observation = tool_registry.execute_tool(tool_name, tool_input)
    # 7. Append "Action: ..." and "Observation: ..." to current_history
    # 8. Repeat
```

**Parsing with regex:**

```python
# Extract Thought:
thought_match = re.search(r"Thought: (.*)", text)

# Extract Action:
action_match = re.search(r"Action: (.*)", text)

# Parse tool_name[tool_input] from action:
match = re.match(r"(\w+)\[(.*)\]", action_text)
tool_name, tool_input = match.group(1), match.group(2)
```

**Why regex?**: The LLM is instructed to follow a strict format. Regex is a lightweight, zero-dependency way to extract structured data from free text when the format is well-controlled. In production systems, you would use JSON-mode or function-calling APIs, but plain-text parsing is educationally more illuminating.

**State management**: `self.current_history` is a list of strings (not `Message` objects) accumulated per `run()` call. It is reset at the start of each call, separating it from the persistent `self._history` which stores the overall conversation.

**Tool execution**: The agent calls `self.tool_registry.execute_tool(tool_name, tool_input)` — it does not know about the tool implementation at all. This is the **Strategy** pattern: the agent is decoupled from how tools work.

---

#### 3.2.3 `ReflectionAgent` — Self-Critique and Iterative Refinement

**Concept**: Based on the observation that LLMs can judge their own outputs. The agent generates an initial answer, then uses the LLM as a critic to identify weaknesses, then refines. This loops for `max_iterations` cycles.

**Internal Memory class**: The `ReflectionAgent` defines a lightweight `Memory` class:

```python
class Memory:
    records: List[Dict]   # [{"type": "execution"|"reflection", "content": str}]

    def add_record(type, content): ...
    def get_trajectory() -> str: ...   # formats all records into a readable log
    def get_last_execution() -> str:   # finds the most recent "execution" record
```

This is a **simple trajectory buffer** — it stores the agent's history of attempts and reflections as text strings.

**Three prompt templates:**

| Template  | Role                      | Variables                                |
| --------- | ------------------------- | ---------------------------------------- |
| `initial` | Generate first attempt    | `{task}`                                 |
| `reflect` | Critique an attempt       | `{task}`, `{content}`                    |
| `refine`  | Improve based on feedback | `{task}`, `{last_attempt}`, `{feedback}` |

**`run()` flow:**

```
1. Generate initial result via "initial" prompt
2. Store result in memory as type="execution"
3. For i in range(max_iterations):
   a. Generate feedback via "reflect" prompt (uses last execution)
   b. Store feedback in memory as type="reflection"
   c. If feedback contains "无需改进" (no improvement needed) → break early
   d. Generate refined result via "refine" prompt
   e. Store refined result as type="execution"
4. Return memory.get_last_execution()
```

**Early stopping**: The check `"无需改进" in feedback` is a simple heuristic for knowing when the agent is satisfied. A more robust implementation would use a structured output (e.g., JSON with a boolean `needs_improvement` field).

---

#### 3.2.4 `PlanAndSolveAgent` — Two-Phase Decomposition

**Concept**: Based on the 2023 paper *"Plan-and-Solve Prompting"* (Wang et al.). Complex problems are first **decomposed into a plan** (list of simpler steps) and then **each step is executed sequentially** with full context from previous steps.

**Internal classes:**

**`Planner`**: Accepts a question, calls the LLM with a planning prompt, parses the output as a Python list.

```python
class Planner:
    def plan(self, question: str) -> List[str]:
        # 1. Format planning prompt with {question}
        # 2. Call LLM
        # 3. Extract python code block: response.split("```python")[1].split("```")[0]
        # 4. ast.literal_eval() to parse safely
        # Returns: ["Step 1 description", "Step 2 description", ...]
```

**Why `ast.literal_eval` instead of `eval`?**: `eval()` executes arbitrary Python code — a major security risk. `ast.literal_eval()` only evaluates Python **literals** (strings, numbers, lists, dicts, etc.) and raises `ValueError` for anything else. This is the correct way to safely parse user/LLM-provided data.

**`Executor`**: Accepts the original question, the full plan, and executes each step:

```python
class Executor:
    def execute(self, question, plan) -> str:
        history = ""
        for i, step in enumerate(plan):
            prompt = EXECUTOR_PROMPT.format(
                question=question,
                plan=plan,
                history=history or "None",
                current_step=step
            )
            result = self.llm.invoke([{"role": "user", "content": prompt}])
            history += f"Step {i}: {step}\nResult: {result}\n\n"
            final_answer = result
        return final_answer
```

**Accumulating history**: Each step's result is appended to `history`, which is injected into the next step's prompt. This gives the executor "memory" across steps without an external database.

**`PlanAndSolveAgent.run()` calls both**:

```python
plan = self.planner.plan(input_text)
if not plan: return "Unable to generate plan."
final_answer = self.executor.execute(input_text, plan)
```

---

### 3.3 Tools Layer

#### 3.3.1 `tools/base.py` — Abstract Tool Contract

```python
class ToolParameter(BaseModel):
    name: str
    type: str           # "string", "integer", "float", "boolean"
    description: str
    required: bool = True
    default: Any = None

class Tool(ABC):
    def __init__(self, name: str, description: str): ...

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str: ...

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]: ...

    def validate_parameters(self, parameters: Dict) -> bool: ...
    def to_dict(self) -> Dict: ...
```

**Design principles:**

- Tools always accept a `dict` of parameters and return a `str`. This uniform interface allows the registry to call any tool without knowing its type.
- `get_parameters()` returns a schema — this can be used to generate LLM function-calling specs, documentation, or validation logic.
- `validate_parameters()` checks that all required parameters are present.

---

#### 3.3.2 `tools/registry.py` — The Tool Registry

The registry is the **central directory of all available tools**. It is both a storage system and a dispatcher.

**Dual storage model:**

```python
self._tools: dict[str, Tool]              # Tool object instances
self._functions: dict[str, dict]          # {"description": ..., "func": callable}
```

This supports two registration styles:

**Style 1 (OOP — recommended):**

```python
registry.register_tool(CalculatorTool())
```

**Style 2 (functional — quick and dirty):**

```python
registry.register_function("greet", "Says hello to a name", lambda name: f"Hello, {name}!")
```

**`execute_tool(name, input_text)`**: The dispatch method. It looks up by name, checks `_tools` first, then `_functions`, and calls the appropriate runner. All errors are caught and returned as error strings (not raised) — this allows the agent loop to continue even if a tool fails.

**`get_tools_description()`**: Generates the natural-language tool description string injected into ReAct prompts:

```
- search: A web search engine. Use when you need current information...
- python_calculator: Performs mathematical calculations...
```

**Global registry singleton:**

```python
global_registry = ToolRegistry()
```

A module-level singleton so tools registered anywhere are accessible everywhere without passing the registry explicitly.

---

#### 3.3.3 `tools/chain.py` — Sequential Tool Pipeline

**`ToolChain`**: A named sequence of tool execution steps. Each step specifies:

- `tool_name`: which tool to call
- `input_template`: a format string, e.g., `"{search_result}"` — variables are filled from a shared `context` dict
- `output_key`: the key under which the result is stored in `context` for downstream steps

**Execution:**

```python
context = {"input": input_data}
for step in self.steps:
    actual_input = step["input_template"].format(**context)
    result = registry.execute_tool(step["tool_name"], actual_input)
    context[step["output_key"]] = result
return final_result
```

This is a **dataflow pipeline** pattern — the context dictionary passes data between steps, with each step consuming outputs of previous steps as inputs.

**`ToolChainManager`**: Stores multiple named chains and dispatches `execute_chain(chain_name, input_data)`. This acts as a registry of pipelines.

---

#### 3.3.4 `tools/async_executor.py` — Parallel Tool Execution

**`AsyncToolExecutor`**: Wraps a `ToolRegistry` and runs tools in a `ThreadPoolExecutor`. Since most tools are I/O bound (API calls), threading achieves true parallelism.

```python
class AsyncToolExecutor:
    def __init__(self, registry, max_workers=4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers)

    async def execute_tool_async(self, tool_name, input_data) -> str:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, 
                    lambda: self.registry.execute_tool(tool_name, input_data))
        return result

    async def execute_tools_parallel(self, tasks: List[Dict]) -> List[Dict]:
        # Creates all coroutines, awaits each, collects results with metadata
```

**Why `run_in_executor`?**: The underlying tool implementations (like HTTP API calls) are synchronous. `run_in_executor` bridges the sync/async gap by running sync functions in a thread pool while the event loop awaits them.

**`execute_tools_batch`**: Convenience method that runs the **same tool** against a list of inputs in parallel.

**Context manager support**: The executor implements `__enter__` / `__exit__` so it can be used with `async with AsyncToolExecutor(...) as executor:`.

---

#### 3.3.5 Builtin Tools

**`CalculatorTool`** — Safe Mathematical Expression Evaluator

The key innovation is using Python's **AST (Abstract Syntax Tree)** to evaluate expressions safely:

```python
node = ast.parse(expression, mode='eval')
result = self._eval_node(node.body)
```

`_eval_node()` recursively evaluates the AST, but **only** handles whitelisted node types:

- `ast.Constant` / `ast.Num` — literal numbers
- `ast.BinOp` — binary operations (+, -, *, /, **, ^)
- `ast.UnaryOp` — unary negation
- `ast.Call` — function calls, **only if the function name is in `FUNCTIONS`**

If the expression contains anything else (e.g., `__import__`, `open`, `exec`), it raises `ValueError`. This prevents **code injection**.

**`SearchTool`** — Hybrid Web Search

```python
SearchTool(backend="hybrid", tavily_key=None, serpapi_key=None)
```

The tool supports three modes:

- `"tavily"` — Uses the Tavily AI search API (designed for LLM agents)
- `"serpapi"` — Uses SerpApi to scrape Google results
- `"hybrid"` — Auto-selects: tries Tavily first, falls back to SerpApi on failure

Backend availability is determined at `__init__` time by checking both the API key AND whether the library is installed (`try: import tavily`). This gives clear error messages at startup rather than mysterious failures at search time.

---

### 3.4 Utils Layer

#### `utils/logging.py`

```python
def setup_logger(name="hello_agents", level="INFO", format_string=None) -> Logger:
    # Creates a named logger with a StreamHandler to stdout
    # Guards against duplicate handlers with `if not logger.handlers`
```

The duplicate-handler guard is important: without it, calling `setup_logger` multiple times in the same process doubles all log output.

#### `utils/serialization.py`

```python
serialize_object(obj, format="json") -> str | bytes
deserialize_object(data, format="json") -> Any
save_to_file(obj, filepath, format="json") -> None
load_from_file(filepath, format="json") -> Any
```

Supports both `json` (human-readable, text) and `pickle` (binary, Python-native). JSON is the default because it is interoperable.

#### `utils/helpers.py`

```python
format_time(timestamp, format_str) -> str
validate_config(config_dict, required_keys) -> bool
safe_import(module_name, class_name=None) -> Any
ensure_dir(path) -> Path
get_project_root() -> Path
merge_dicts(dict1, dict2) -> dict  # deep merge
```

`safe_import` is particularly useful: it wraps `importlib.import_module` in a try/except and raises a cleaner `ImportError` with context. This is used to handle optional dependencies gracefully.

---

## 4. Data Flow & Execution Lifecycle

### 4.1 SimpleAgent — Single Turn

```
User calls: agent.run("What is the capital of France?")
│
├─ Build messages list:
│   [{"role": "system", "content": "You are a helpful assistant"},
│    {"role": "user",   "content": "What is the capital of France?"}]
│
├─ llm.invoke(messages)
│   └─ openai.Client.chat.completions.create(model=..., messages=..., stream=False)
│   └─ returns: "The capital of France is Paris."
│
├─ add_message(Message("What is the capital of France?", "user"))
├─ add_message(Message("The capital of France is Paris.", "assistant"))
│
└─ return "The capital of France is Paris."
```

### 4.2 ReActAgent — Multi-Step Loop

```
User calls: react_agent.run("What is 2024 US GDP and how does it compare to 2023?")
│
iteration 1:
├─ Build prompt: tools=[search, calculator], history=[], question=...
├─ llm.invoke(prompt)
│   └─ LLM returns:
│       Thought: I need to find 2024 US GDP first.
│       Action: search[US GDP 2024]
├─ Parse: thought="I need...", action="search[US GDP 2024]"
├─ Execute: tool_registry.execute_tool("search", "US GDP 2024")
│   └─ returns: "US GDP in 2024 was estimated at $29 trillion..."
├─ Append to history:
│   "Action: search[US GDP 2024]"
│   "Observation: US GDP in 2024 was estimated at $29 trillion..."
│
iteration 2:
├─ Build prompt: tools=..., history=<above>, question=...
├─ llm.invoke(prompt)
│   └─ LLM returns:
│       Thought: Now I need 2023 GDP to compare.
│       Action: search[US GDP 2023]
├─ Execute, append...
│
iteration 3:
├─ llm.invoke(prompt)
│   └─ LLM returns:
│       Thought: I have both figures. 2024 was $29T, 2023 was $27.4T. I can answer.
│       Action: Finish[US GDP grew from $27.4T in 2023 to $29T in 2024, a ~5.8% increase.]
├─ Parse: action starts with "Finish"
├─ Extract final answer, save to self._history, return.
```

### 4.3 ReflectionAgent — Iterative Refinement

```
User calls: reflection_agent.run("Write a Python quicksort implementation")
│
initial attempt:
├─ Prompt: "Complete this task: Write a Python quicksort implementation"
├─ LLM → generates initial code (possibly buggy or incomplete)
├─ memory.add_record("execution", initial_code)
│
iteration 1:
├─ reflect prompt: "Review this code for the task... find problems..."
├─ LLM → "The code lacks base case handling and doesn't handle duplicates correctly."
├─ memory.add_record("reflection", feedback)
│  (feedback does NOT contain "无需改进", continue)
├─ refine prompt: "Improve this code based on feedback: ..."
├─ LLM → improved_code
├─ memory.add_record("execution", improved_code)
│
iteration 2:
├─ reflect prompt: "Review improved code..."
├─ LLM → "无需改进" (No improvement needed)
├─ memory.add_record("reflection", "无需改进")
├─ break early ✓
│
└─ return memory.get_last_execution()  # the improved_code
```

### 4.4 PlanAndSolveAgent — Two-Phase

```
User calls: agent.run("Plan a 7-day trip to Japan with budget $3000")
│
PHASE 1 — Planning:
├─ planner.plan("Plan a 7-day trip to Japan...")
├─ LLM → returns:
│   ```python
│   ["Research flights and accommodation costs",
│    "Create daily itinerary for Tokyo (days 1-3)",
│    "Create daily itinerary for Kyoto (days 4-5)",
│    "Create daily itinerary for Osaka (days 6-7)",
│    "Calculate total budget allocation"]
│   ```
├─ ast.literal_eval() → plan = [list of 5 strings]
│
PHASE 2 — Execution:
├─ executor.execute(question, plan)
│   ├─ Step 1: "Research flights..." → LLM generates cost estimates → history updated
│   ├─ Step 2: "Tokyo itinerary..." → LLM generates daily plan with history context
│   ├─ Step 3: "Kyoto itinerary..." → similarly
│   ├─ Step 4: "Osaka itinerary..." → similarly
│   └─ Step 5: "Budget calculation..." → final calculation using prior steps
│
└─ return final_answer (result of last step)
```

---

## 5. Design Patterns Used

| Pattern                                   | Where Applied                                     | Purpose                                       |
| ----------------------------------------- | ------------------------------------------------- | --------------------------------------------- |
| **Abstract Base Class (Template Method)** | `Agent`, `Tool`                                   | Enforce interface contract on subclasses      |
| **Strategy**                              | `Agent` holds `llm`; `SearchTool` holds `backend` | Swap implementations without changing caller  |
| **Factory Method**                        | `Config.from_env()`                               | Alternate constructors for different contexts |
| **Registry / Service Locator**            | `ToolRegistry`, `global_registry`                 | Central tool lookup by name                   |
| **Decorator / Adapter**                   | `HelloAgentsLLM` wraps `openai.OpenAI`            | Unified interface over multiple providers     |
| **Composite / Pipeline**                  | `ToolChain` + `ToolChainManager`                  | Chain tools into multi-step pipelines         |
| **Observer (lightweight)**                | `Memory` in `ReflectionAgent`                     | Store trajectory of agent actions             |
| **Singleton**                             | `global_registry`                                 | One shared instance at module level           |
| **Command**                               | Each `Tool.run(parameters)` call                  | Encapsulate an operation with its parameters  |

---

## 6. Key Algorithms & Logic Explained

### 6.1 ReAct Output Parsing

The regex-based parser must handle LLM outputs that may vary in whitespace, capitalization, or formatting. The current implementation uses `re.search` (finds anywhere in text, not just at the start):

```python
# These patterns search anywhere in the text:
thought_match = re.search(r"Thought: (.*)", text)
action_match = re.search(r"Action: (.*)", text)

# Tool call parsing: word characters + content in brackets
match = re.match(r"(\w+)\[(.*)\]", action_text)
```

**Potential failure modes:**

1. LLM outputs multiline thought — `(.*)` only matches up to newline. Fix: use `re.DOTALL` flag.
2. LLM uses different capitalization ("THOUGHT:" vs "Thought:"). Fix: use `re.IGNORECASE`.
3. Tool input contains `]` — the greedy `(.*)` in `\[(.*)\]` would consume too much. Fix: use `(.*?)` (non-greedy) or match until the last `]`.

Understanding these failure modes is critical for building a robust implementation.

### 6.2 Safe AST-based Expression Evaluation

```python
def _eval_node(self, node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        op_func = self.OPERATORS[type(node.op)]  # KeyError if unsupported op
        left = self._eval_node(node.left)
        right = self._eval_node(node.right)
        return op_func(left, right)
    elif isinstance(node, ast.Call):
        func_name = node.func.id
        if func_name not in self.FUNCTIONS:
            raise ValueError(f"Unsupported function: {func_name}")
        args = [self._eval_node(arg) for arg in node.args]
        return self.FUNCTIONS[func_name](*args)
    else:
        raise ValueError(f"Unsupported node type: {type(node)}")
```

This is a **tree-walking interpreter**. The key security property: if the expression contains anything not in the explicit whitelist (`OPERATORS` + `FUNCTIONS`), it raises `ValueError`. No arbitrary code can execute.

### 6.3 Provider Auto-Detection Priority

The `_auto_detect_provider()` checks signals in order of **specificity** (most specific → least specific):

```
Level 1: Named env vars (OPENAI_API_KEY)        ← most specific
Level 2: API key format ("ms-" prefix)
Level 3: base_url domain (api.deepseek.com)
Level 4: Localhost port patterns (:11434 = Ollama)
Level 5: Generic (LLM_API_KEY / LLM_BASE_URL)   ← least specific → "auto"
```

This avoids false positives: a key starting with `sk-` could be OpenAI or DeepSeek, so just the key format is insufficient — the URL is checked too.

---

## 7. Configuration & Environment Management

### 7.1 The `.env` file

Located at `core/.env` (though typically at the project root in production). **Never commit secrets to version control.**

```ini
# LLM Provider (choose one approach):
LLM_MODEL_ID=llama3.2:3b
LLM_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama

# OR for OpenAI:
# OPENAI_API_KEY=sk-...

# Search tools (optional):
TAVILY_API_KEY=tvly-...
SERPAPI_API_KEY=...
```

### 7.2 Priority Order for LLM Configuration

```
1. Explicit constructor argument          → highest priority
2. Provider-specific env var             → (e.g., OPENAI_API_KEY)
3. Generic env var                       → (LLM_API_KEY, LLM_BASE_URL)
4. Hardcoded default in _resolve_credentials
```

### 7.3 Why Not Load `.env` Automatically?

The framework does **not** call `dotenv.load_dotenv()` automatically. The reason: in production deployments, environment variables are typically set by the deployment platform (Docker, Kubernetes, etc.), not `.env` files. Calling `load_dotenv()` in a library is considered bad practice because it modifies the process environment as a side effect.

Users who want `.env` file support should call `load_dotenv()` in their own application code before creating any framework objects.

---

## 8. Extension Points — How to Add New Agents and Tools

### 8.1 Creating a Custom Tool

```python
from CortexGraph.tools.base import Tool, ToolParameter

class WeatherTool(Tool):
    def __init__(self):
        super().__init__(
            name="weather",
            description="Get current weather for a city. Input: city name."
        )

    def run(self, parameters: dict) -> str:
        city = parameters.get("input", "")
        # call weather API here...
        return f"Weather in {city}: 72°F, partly cloudy"

    def get_parameters(self):
        return [
            ToolParameter(name="input", type="string",
                         description="City name", required=True)
        ]

# Register it:
from CortexGraph.tools.registry import global_registry
global_registry.register_tool(WeatherTool())
```

### 8.2 Creating a Custom Agent

```python
from CortexGraph.core.agent import Agent
from CortexGraph.core.message import Message

class ChainOfThoughtAgent(Agent):
    """Agent that forces explicit step-by-step reasoning."""

    def run(self, input_text: str, **kwargs) -> str:
        cot_prompt = f"Solve this step by step:\n{input_text}\n\nStep 1:"
        messages = [{"role": "user", "content": cot_prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        response = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))

        return response
```

### 8.3 Custom Prompt Templates

Both `ReActAgent` and `PlanAndSolveAgent` accept `custom_prompt` / `custom_prompts` arguments:

```python
# ReActAgent with custom prompt:
my_prompt = "You are a financial analyst. Tools: {tools}\nQ: {question}\nHistory: {history}\n"
react_agent = ReActAgent(..., custom_prompt=my_prompt)

# PlanAndSolveAgent with custom prompts:
ps_agent = PlanAndSolveAgent(..., custom_prompts={
    "planner": "You are a project manager. Break this into tasks: {question}\n```python\n",
    "executor": "Execute this task: {current_step}\nContext: {history}\nAnswer:"
})
```

---

## 9. Implementation Specification for Students

This section specifies what you must build. Implement it **without looking at the original code** after reading this document. You are building a similar but independent framework.

---

### Phase 1: Core Layer (Week 1)

#### Task 1.1 — Exception Hierarchy

Create `core/exceptions.py` with a base exception class `AgentFrameworkError` and at least four subclasses: `LLMError`, `AgentError`, `ConfigError`, `ToolError`.

**Requirement**: All your framework errors should catch external exceptions (like `openai.APIError`) and re-raise them wrapped in your custom types.

#### Task 1.2 — Message Model

Create `core/message.py`. Your `Message` class must:

- Have fields: `role` (restricted to `"user" | "assistant" | "system" | "tool"`), `content` (str), `timestamp` (auto-set), `metadata` (optional dict).
- Have a `to_dict()` method returning only `{"role": ..., "content": ...}`.
- Use Pydantic `BaseModel` for validation.

#### Task 1.3 — Configuration

Create `core/config.py` with a `Config` class (Pydantic) holding at minimum: `default_model`, `temperature`, `max_tokens`, `debug`, `log_level`, `max_history_length`. Implement a `from_env()` class method.

#### Task 1.4 — LLM Client

Create `core/llm.py` with a class `LLMClient` (or similar). **Requirements**:

1. Constructor accepts `model`, `api_key`, `base_url`, `provider`, `temperature`, `max_tokens`.
2. Support at minimum 3 different LLM providers (OpenAI + 2 others of your choice).
3. Implement provider auto-detection from environment variables.
4. Implement `invoke(messages) -> str` (non-streaming).
5. Implement `stream_invoke(messages) -> Iterator[str]` (streaming generator).
6. All API errors must be caught and re-raised as `LLMError`.

#### Task 1.5 — Abstract Agent Base

Create `core/agent.py` with an abstract `Agent` class. **Requirements**:

1. Constructor stores `name`, `llm`, `system_prompt`, `config`, and initializes `self._history: list[Message]`.
2. Abstract method `run(input_text: str, **kwargs) -> str`.
3. Concrete methods: `add_message()`, `clear_history()`, `get_history()` (returns a copy).

---

### Phase 2: Tools Layer (Week 2)

#### Task 2.1 — Tool Base

Create `tools/base.py` with:

- `ToolParameter` (Pydantic model): `name`, `type`, `description`, `required`, `default`
- Abstract `Tool` class with abstract `run(parameters: dict) -> str` and `get_parameters() -> List[ToolParameter]`
- Concrete `validate_parameters()` and `to_dict()` methods

#### Task 2.2 — Tool Registry

Create `tools/registry.py` with `ToolRegistry`. **Requirements**:

1. Support both `register_tool(Tool)` and `register_function(name, description, callable)`.
2. `execute_tool(name, input_text) -> str` — dispatch to registered tool, catch all errors, return error string (do not propagate).
3. `get_tools_description() -> str` — format all tools as a natural-language list.
4. Module-level `global_registry = ToolRegistry()` singleton.

#### Task 2.3 — Calculator Tool

Implement a `CalculatorTool` that uses `ast.parse()` + a recursive node evaluator. **Requirements**:

- Support: `+`, `-`, `*`, `/`, `**` (power), unary `-`
- Support functions: `sqrt`, `sin`, `cos`, `abs`, `round`
- Must NOT use `eval()` — only AST traversal.
- Invalid expressions must return an error string, not raise exceptions.

#### Task 2.4 — Search Tool (Optional but Recommended)

Implement a search tool supporting at least one backend (Tavily or SerpApi). The tool must:

- Check at `__init__` whether the required library and API key are available.
- Return helpful error messages if the backend is not configured.

#### Task 2.5 — Tool Chain

Implement `ToolChain` with:

- `add_step(tool_name, input_template, output_key)` — build a pipeline step-by-step
- `execute(registry, input_data, context=None) -> str` — run all steps, passing context between them via format string interpolation

---

### Phase 3: Agent Implementations (Weeks 3–4)

#### Task 3.1 — SimpleAgent

Implement a `SimpleAgent` that:

1. Builds message list: system prompt + history + current input
2. Calls `llm.invoke()`
3. Saves to `_history` after each call
4. Has both `run()` (blocking) and `stream_run()` (generator) methods

#### Task 3.2 — ReActAgent

Implement a `ReActAgent` that:

1. Accepts a `ToolRegistry` in its constructor
2. Implements the Thought/Action/Observation loop
3. Parses LLM output using regex to extract `Thought:` and `Action:`
4. Parses `tool_name[tool_input]` format from actions
5. Recognizes `Finish[answer]` as termination
6. Has a configurable `max_steps` limit
7. Accepts custom prompt templates

**Prompt template must include**: `{tools}`, `{question}`, `{history}` variables.

#### Task 3.3 — ReflectionAgent

Implement a `ReflectionAgent` with:

1. An internal `Memory` class tracking `execution` and `reflection` records
2. Three prompt templates: `initial`, `reflect`, `refine`
3. An iteration loop with early stopping when feedback says no improvement needed
4. Configurable `max_iterations`

#### Task 3.4 — PlanAndSolveAgent

Implement a `PlanAndSolveAgent` with:

1. Separate `Planner` class that generates a Python list via LLM + parses with `ast.literal_eval`
2. Separate `Executor` class that runs steps sequentially, accumulating history
3. The agent orchestrates both, handling the case where planning fails

---

### Phase 4: Utils + Integration (Week 4–5)

#### Task 4.1 — Logging

Create a logger factory: `setup_logger(name, level, format) -> Logger`. Must guard against duplicate handlers.

#### Task 4.2 — Serialization

Implement `save_to_file` / `load_from_file` supporting both JSON and pickle formats.

#### Task 4.3 — Public API

Create a top-level `__init__.py` that re-exports all major classes. Users should be able to do:

```python
from your_framework import SimpleAgent, ReActAgent, LLMClient, ToolRegistry
```

#### Task 4.4 — End-to-End Demo

Write a script `demo.py` that:

1. Creates an `LLMClient` pointing to any model you have access to
2. Runs all four agent types on a non-trivial input
3. Demonstrates registering and using at least one tool with `ReActAgent`

---

## 10. Grading Rubric & Milestone Checklist

### Milestone 1 — Core Layer (25 points)

| Criterion                                                   | Points |
| ----------------------------------------------------------- | ------ |
| `Message` model with proper role validation and `to_dict()` | 4      |
| `Config` with Pydantic and `from_env()` factory             | 3      |
| `LLMClient` supporting 3+ providers with auto-detection     | 8      |
| Both `invoke()` and `stream_invoke()` implemented correctly | 4      |
| Exception hierarchy with correct wrapping                   | 3      |
| Abstract `Agent` base class with history management         | 3      |

### Milestone 2 — Tools Layer (25 points)

| Criterion                                              | Points |
| ------------------------------------------------------ | ------ |
| Abstract `Tool` + `ToolParameter` schema               | 4      |
| `ToolRegistry` with both registration styles           | 5      |
| `execute_tool()` catches errors, returns error strings | 3      |
| `get_tools_description()` for prompt injection         | 3      |
| `CalculatorTool` using AST traversal (NO `eval`)       | 6      |
| `ToolChain` with template-based variable passing       | 4      |

### Milestone 3 — Agent Implementations (35 points)

| Criterion                                                      | Points |
| -------------------------------------------------------------- | ------ |
| `SimpleAgent` with history, `run()` and `stream_run()`         | 5      |
| `ReActAgent` — correct Thought/Action/Observation loop         | 10     |
| `ReActAgent` — regex parsing robust to edge cases              | 3      |
| `ReflectionAgent` — three-phase loop with Memory class         | 8      |
| `PlanAndSolveAgent` — separate Planner/Executor + safe parsing | 9      |

### Milestone 4 — Quality & Integration (15 points)

| Criterion                                            | Points |
| ---------------------------------------------------- | ------ |
| Clean public API in `__init__.py`                    | 2      |
| Logging utility with duplicate-handler guard         | 2      |
| All four agents run successfully end-to-end in demo  | 5      |
| Code style: type hints, docstrings, no magic numbers | 3      |
| No use of `eval()` anywhere in the codebase          | 3      |

---

### Bonus Challenges (up to 15 extra points)

| Challenge                                                                   | Points |
| --------------------------------------------------------------------------- | ------ |
| Implement `AsyncToolExecutor` with `asyncio` + `ThreadPoolExecutor`         | 5      |
| Add a 5th agent paradigm (e.g., Tree-of-Thought, MCTS-based)                | 5      |
| Implement tool parameter JSON schema generation for OpenAI function calling | 3      |
| Add multi-agent orchestration: one agent calls another as a tool            | 5      |
| Write unit tests with `pytest` achieving >70% code coverage                 | 4      |

---

## Appendix A — Recommended Background Reading

1. **ReAct Paper**: Yao et al. (2022). *"ReAct: Synergizing Reasoning and Acting in Language Models."* arXiv:2210.03629
2. **Plan-and-Solve Paper**: Wang et al. (2023). *"Plan-and-Solve Prompting."* arXiv:2305.04091
3. **Reflection/Self-Critique**: Shinn et al. (2023). *"Reflexion: Language Agents with Verbal Reinforcement Learning."* arXiv:2303.11366
4. **OpenAI Chat Completions API**: https://platform.openai.com/docs/guides/chat
5. **Python `ast` module**: https://docs.python.org/3/library/ast.html
6. **Pydantic v2 docs**: https://docs.pydantic.dev/latest/
7. **asyncio + ThreadPoolExecutor**: https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor

## Appendix B — Common Pitfalls to Avoid

1. **Using `eval()` for expression evaluation** — Always use `ast.literal_eval` for data, AST traversal for expressions.
2. **Mutating history from outside the agent** — `get_history()` should return a copy.
3. **Hardcoding model names or API keys** — Always read from env vars with explicit fallback.
4. **Calling `load_dotenv()` inside library code** — Let the application do this; libraries should not modify the process environment.
5. **Missing the streaming aggregation in `stream_run()`** — You must accumulate chunks before saving to history; never store partial text.
6. **Not resetting per-run state** — In `ReActAgent`, `self.current_history` must be reset at the start of each `run()` call, otherwise previous runs contaminate new ones.
7. **Forgetting the `or ""` in streaming** — `chunk.choices[0].delta.content` can be `None` for non-text chunks; always apply `or ""`.
8. **No early stopping in ReflectionAgent** — Without it, the agent always runs the full `max_iterations`, wasting tokens.

## Appendix C — Quick Reference: Class Relationships

```
Config ──────────────────────────────────────────┐
                                                  │
HelloAgentsLLM ──────────────────────────────────┤
  │                                              │
  │ (owns)                                       ▼
  │                                    Agent (abstract)
  │                                        │
  ├──────────────────────────────────────  │
  │                              ┌─────────┴──────────────────────┐
  │                              │         │          │            │
  │                         SimpleAgent ReActAgent ReflectionAgent PlanAndSolveAgent
  │                                          │
  │                                    ToolRegistry
  │                                          │
  │                                     ┌───┴───┐
  │                                  Tool    function
  │                                  (ABC)   (callable)
  │                                    │
  │                           ┌────────┴────────┐
  │                      SearchTool      CalculatorTool
  │
  └─ (used by) ──────────────────────────────────────────────────▶ openai.OpenAI
```

---

*End of CortexGraph Comprehensive Framework Specification Guide*
*Prepared for Stanford CS — Spring 2026*
*Version 1.0*
