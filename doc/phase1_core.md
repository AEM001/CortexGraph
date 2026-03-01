# Phase 1 — Build the Core Layer

> **Goal:** Build the five foundational primitives every other part of the framework depends on.
> **Time:** 3–4 hours
> **Depends on:** Phase 0 complete
> **Next:** [phase2_tools.md](./phase2_tools.md)

You will build these files in order:

```
core/exceptions.py   ← Step 1
core/message.py      ← Step 2
core/config.py       ← Step 3
core/llm.py          ← Step 4  (most complex)
core/agent.py        ← Step 5
core/__init__.py     ← Step 6
```

---

## Step 1 — `core/exceptions.py`: The Exception Hierarchy

### Why this comes first

Every other module will `raise` and `except` your custom exceptions. You need them defined before anything else can use them. This is also the simplest file — a good warm-up.

### What to build

A base exception class and four typed subclasses:

```
AgentFrameworkError          ← everything inherits from this
├── LLMError                 ← LLM API call failures
├── AgentError               ← logic errors inside an agent's run()
├── ConfigError              ← missing or invalid configuration
└── ToolError                ← tool execution failures
```

### Exact requirements

- All classes inherit from Python's built-in `Exception` (through your base class).
- Each subclass adds no extra logic — just `pass`. The type itself carries the meaning.
- Your base class should accept an optional `message: str` argument and store it.

### Hint on usage pattern

Every module that calls an external API should do this:

```python
try:
    result = some_external_call()
except SomeExternalError as e:
    raise LLMError(f"Call failed: {e}") from e
```

The `from e` preserves the original traceback while surfacing your typed error.

### Self-check

```python
# Run this at the bottom of the file to verify:
try:
    raise LLMError("test error")
except AgentFrameworkError as e:
    print(f"Caught as base type: {e}")   # should print

try:
    raise ToolError("tool broke")
except LLMError:
    print("Wrong!")
except ToolError as e:
    print(f"Caught as correct type: {e}")  # should print
```

---

## Step 2 — `core/message.py`: The Message Model

### Why this matters

The OpenAI Chat Completions API takes a list of messages, each shaped like `{"role": "user", "content": "hello"}`. Your `Message` class is the in-memory object that represents one of these. It adds useful fields (timestamp, metadata) that are framework-internal and must be stripped before sending to the API.

### What to build

A Pydantic `BaseModel` with these fields:

| Field       | Type                                             | Notes                                        |
| ----------- | ------------------------------------------------ | -------------------------------------------- |
| `content`   | `str`                                            | The text of the message                      |
| `role`      | `Literal["user", "assistant", "system", "tool"]` | Must be one of exactly these four strings    |
| `timestamp` | `datetime`                                       | Auto-set to `datetime.now()` if not provided |
| `metadata`  | `Optional[Dict[str, Any]]`                       | Default to empty dict `{}`                   |

### Exact requirements

1. Use `pydantic.BaseModel` — do not use a plain `dataclass`.
2. The `role` field must use `Literal` typing so Pydantic rejects invalid roles at construction time.
3. Implement `to_dict(self) -> Dict[str, Any]` that returns **only** `{"role": ..., "content": ...}`. The `timestamp` and `metadata` fields must NOT appear in this output — they are never sent to the API.
4. Implement `__str__` for readable printing, e.g. `[user] Hello there`.

### Common mistake to avoid

Do not set `timestamp: datetime = datetime.now()` as a default. In Pydantic v2, mutable defaults are evaluated once at class definition time, meaning all messages get the same timestamp. Instead, use `default_factory`:

```python
from pydantic import Field
timestamp: datetime = Field(default_factory=datetime.now)
```

### Self-check

```python
m = Message(content="Hello", role="user")
assert m.role == "user"
assert "timestamp" not in m.to_dict()
assert m.to_dict() == {"role": "user", "content": "Hello"}

try:
    Message(content="oops", role="robot")   # should raise ValidationError
except Exception as e:
    print(f"Correctly rejected: {e}")
```

---

## Step 3 — `core/config.py`: Configuration Management

### Why this matters

Without a central config object, you'd scatter magic numbers and environment variable reads throughout every file. `Config` is the single source of truth for tunable parameters.

### What to build

A Pydantic `BaseModel` with these fields and defaults:

| Field                | Type            | Default           | Purpose                           |
| -------------------- | --------------- | ----------------- | --------------------------------- |
| `default_model`      | `str`           | `"gpt-3.5-turbo"` | Fallback model name               |
| `default_provider`   | `str`           | `"openai"`        | Fallback provider                 |
| `temperature`        | `float`         | `0.7`             | LLM sampling temperature          |
| `max_tokens`         | `Optional[int]` | `None`            | Token limit (None = no limit)     |
| `debug`              | `bool`          | `False`           | Enable debug output               |
| `log_level`          | `str`           | `"INFO"`          | Logging verbosity                 |
| `max_history_length` | `int`           | `100`             | Max messages kept in agent memory |

### Exact requirements

1. Implement a `@classmethod from_env(cls) -> "Config"` that reads these env vars:
   - `DEBUG` → bool (compare lowercase string to `"true"`)
   - `LOG_LEVEL` → string
   - `TEMPERATURE` → float
   - `MAX_TOKENS` → int (only if the env var is set; otherwise `None`)
2. Implement `to_dict(self) -> Dict[str, Any]` — use Pydantic's built-in `.model_dump()` (Pydantic v2) or `.dict()` (Pydantic v1).

### Self-check

```python
import os
os.environ["TEMPERATURE"] = "0.3"
os.environ["DEBUG"] = "true"

cfg = Config.from_env()
assert cfg.temperature == 0.3
assert cfg.debug == True
assert cfg.max_tokens is None
print("Config OK")
```

---

## Step 4 — `core/llm.py`: The Unified LLM Client

### Why this is the most important file

This is the "brain connector." Every agent talks to an LLM through this class. Its job is to hide all the differences between providers (OpenAI, DeepSeek, Ollama, Qwen, etc.) behind a single, clean interface.

All supported providers expose an OpenAI-compatible REST API — the only differences are the `base_url` and how you find the `api_key`. This means you can use the exact same `openai.OpenAI` SDK client for all of them, just pointed at different URLs.

### The big picture before you write a line

```
User creates: LLMClient(provider="deepseek")
                    ↓
Step 1: Store model, temperature, max_tokens, timeout from args/env vars
Step 2: Auto-detect provider (if not given explicitly)
Step 3: Resolve api_key + base_url for that provider
Step 4: Create openai.OpenAI(api_key=..., base_url=..., timeout=...)
                    ↓
User calls: client.invoke(messages)  →  returns full string
       or:  client.stream_invoke(messages)  →  yields string chunks
```

### Supported providers and their credentials

You must support at minimum these providers. Each has a specific env var to check:

| Provider   | Env var for API key                    | Default base_url                                    |
| ---------- | -------------------------------------- | --------------------------------------------------- |
| `openai`   | `OPENAI_API_KEY`                       | `https://api.openai.com/v1`                         |
| `deepseek` | `DEEPSEEK_API_KEY`                     | `https://api.deepseek.com`                          |
| `qwen`     | `DASHSCOPE_API_KEY`                    | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `kimi`     | `KIMI_API_KEY` or `MOONSHOT_API_KEY`   | `https://api.moonshot.cn/v1`                        |
| `zhipu`    | `ZHIPU_API_KEY`                        | `https://open.bigmodel.cn/api/paas/v4`              |
| `ollama`   | `OLLAMA_API_KEY` (default: `"ollama"`) | `http://localhost:11434/v1`                         |
| `auto`     | `LLM_API_KEY`                          | `LLM_BASE_URL` env var                              |

### Step 4a — Constructor

```python
def __init__(
    self,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
):
```

Inside `__init__`, follow this exact order:

1. Set `self.model = model or os.getenv("LLM_MODEL_ID")`
2. Set `self.temperature`, `self.max_tokens`, `self.timeout`
3. Call `self.provider = provider or self._auto_detect_provider(api_key, base_url)`
4. Call `self.api_key, self.base_url = self._resolve_credentials(api_key, base_url)`
5. If `self.model` is still `None`, call `self.model = self._get_default_model()`
6. Validate: if `api_key` or `base_url` is still missing, raise `ConfigError`
7. Create `self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)`

### Step 4b — Auto-detect provider (`_auto_detect_provider`)

Check signals in this exact priority order (most specific first):

1. **Named provider env vars** — if `OPENAI_API_KEY` is set → return `"openai"`, if `DEEPSEEK_API_KEY` → return `"deepseek"`, etc.
2. **Generic key format** — if `LLM_API_KEY` value equals `"ollama"` → return `"ollama"`, if it starts with `"ms-"` → return `"modelscope"`, etc.
3. **Base URL domain** — if `LLM_BASE_URL` contains `"api.deepseek.com"` → return `"deepseek"`, if it contains `"localhost:11434"` → return `"ollama"`, etc.
4. **Fallback** — return `"auto"` (use generic `LLM_API_KEY` / `LLM_BASE_URL`)

### Step 4c — Resolve credentials (`_resolve_credentials`)

For each provider, follow this fallback chain:

```
explicit argument → provider-specific env var → generic LLM_API_KEY → hardcoded default
```

Example for `ollama`:

```python
api_key = api_key or os.getenv("OLLAMA_API_KEY") or os.getenv("LLM_API_KEY") or "ollama"
base_url = base_url or os.getenv("OLLAMA_HOST") or os.getenv("LLM_BASE_URL") or "http://localhost:11434/v1"
```

Implement this pattern for every provider in the table above.

### Step 4d — Default model per provider (`_get_default_model`)

Return a sensible default model name when none is specified:

| Provider   | Default model      |
| ---------- | ------------------ |
| `openai`   | `"gpt-3.5-turbo"`  |
| `deepseek` | `"deepseek-chat"`  |
| `qwen`     | `"qwen-plus"`      |
| `kimi`     | `"moonshot-v1-8k"` |
| `zhipu`    | `"glm-4"`          |
| `ollama`   | `"llama3.2"`       |
| `auto`     | `"gpt-3.5-turbo"`  |

### Step 4e — The two call methods

**`invoke(messages, **kwargs) -> str`** — non-streaming, returns the full response at once:

```python
response = self._client.chat.completions.create(
    model=self.model,
    messages=messages,
    temperature=kwargs.get("temperature", self.temperature),
    max_tokens=kwargs.get("max_tokens", self.max_tokens),
    stream=False,
)
return response.choices[0].message.content
```

Wrap the whole body in `try/except Exception as e: raise LLMError(...) from e`.

**`stream_invoke(messages, **kwargs) -> Iterator[str]`** — streaming generator, yields token chunks:

```python
response = self._client.chat.completions.create(
    ...,
    stream=True,
)
for chunk in response:
    content = chunk.choices[0].delta.content or ""
    if content:
        yield content
```

Key detail: `chunk.choices[0].delta.content` can be `None` for metadata-only chunks. The `or ""` handles this — and you `if content:` before yielding so you never yield empty strings.

### Self-check

```python
# Load your .env first
from dotenv import load_dotenv
load_dotenv()

from myagents.core.llm import LLMClient

llm = LLMClient()  # auto-detects provider from env
response = llm.invoke([{"role": "user", "content": "Say exactly: hello world"}])
print(response)   # should print something containing "hello world"

# Test streaming
chunks = list(llm.stream_invoke([{"role": "user", "content": "Count to 3"}]))
print("".join(chunks))   # should print "1, 2, 3" or similar
```

---

## Step 5 — `core/agent.py`: The Abstract Base Class

### Why an abstract base?

You are about to build 4 different agents. All of them share common behavior: they hold an LLM, maintain conversation history, and accept a system prompt. Rather than copy-pasting this into each agent, you define it once in a base class. Each agent then only implements what makes it unique — its `run()` method.

This is the **Template Method** pattern: the base class defines the structure, subclasses fill in the details.

### What to build

```python
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """Every subclass must implement this."""
        pass

    def add_message(self, message: Message) -> None: ...
    def clear_history(self) -> None: ...
    def get_history(self) -> list[Message]: ...   # returns a COPY
```

### Exact requirements

1. `run()` is `@abstractmethod` — Python will raise `TypeError` if anyone tries to instantiate a subclass that hasn't implemented it.
2. `_history` uses a leading underscore — it is private. Subclasses write to it via `add_message()`, never directly.
3. `get_history()` must return `self._history.copy()`, not `self._history`. Returning the list directly allows external callers to mutate the internal state.
4. `config or Config()` — if `None` is passed, create a default config. Don't require callers to know about `Config`.
5. Implement `__str__` and `__repr__` for debugging: e.g. `Agent(name=mybot, provider=openai)`.

### Self-check

```python
from myagents.core.agent import Agent

# Try to instantiate the abstract class directly — should fail:
try:
    a = Agent("test", llm=None)
    print("ERROR: should have raised TypeError")
except TypeError as e:
    print(f"Correctly blocked: {e}")

# A minimal concrete subclass for testing:
class TestAgent(Agent):
    def run(self, input_text, **kwargs):
        return f"Echo: {input_text}"

agent = TestAgent(name="echo", llm=None)
agent.add_message(Message(content="hi", role="user"))
history = agent.get_history()
history.clear()   # mutating the returned copy should NOT affect the agent
assert len(agent.get_history()) == 1   # internal history intact
print("Agent base class OK")
```

---

## Step 6 — `core/__init__.py`: Wire It Together

Export everything from the core layer so other modules can import cleanly:

```python
from .exceptions import AgentFrameworkError, LLMError, AgentError, ConfigError, ToolError
from .message import Message
from .config import Config
from .llm import LLMClient
from .agent import Agent

__all__ = [
    "AgentFrameworkError", "LLMError", "AgentError", "ConfigError", "ToolError",
    "Message", "Config", "LLMClient", "Agent",
]
```

### Self-check

```python
from myagents.core import Agent, LLMClient, Message, Config, LLMError
print("All core imports OK")
```

---

## ✅ Phase 1 Complete Checklist

Run each check before moving to Phase 2:

- [ ] `core/exceptions.py` — 5 exception classes, base catches all subtypes
- [ ] `core/message.py` — Pydantic model, role validation, `to_dict()` strips timestamp/metadata
- [ ] `core/config.py` — Pydantic model, `from_env()` reads 4 env vars
- [ ] `core/llm.py` — auto-detects provider, resolves credentials, `invoke()` and `stream_invoke()` both work
- [ ] `core/agent.py` — abstract `run()`, private history, `get_history()` returns copy
- [ ] `core/__init__.py` — all 5 modules exported
- [ ] All self-checks pass without errors

**Next → [Phase 2: Build the Tools Layer](./phase2_tools.md)**
