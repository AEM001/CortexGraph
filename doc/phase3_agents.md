# Phase 3 — Build the Four Agent Implementations

> **Goal:** Implement four distinct agent paradigms, each representing a different strategy for using LLMs to solve problems.
> **Time:** 4–6 hours
> **Depends on:** Phase 1 and Phase 2 complete
> **Next:** [phase4_integration.md](./phase4_integration.md)

You will build these files in order:

```
agents/simple_agent.py       ← Step 1  (~1 hour)
agents/react_agent.py        ← Step 2  (~2 hours)
agents/reflection_agent.py   ← Step 3  (~1.5 hours)
agents/plan_solve_agent.py   ← Step 4  (~1.5 hours)
agents/__init__.py           ← Step 5
```

Before writing any code, read the concept section for each agent. Understanding *why* it works the way it does is just as important as implementing it.

---

## Step 1 — `agents/simple_agent.py`: Stateful Conversational Agent

### Concept

The simplest useful agent. It wraps an LLM call with two additions that make it far more useful than a raw API call:
1. **A system prompt** — sets the LLM's persona and instructions for all turns
2. **Conversation history** — every exchange is remembered, so the agent can refer back to earlier messages

Without history, every call is independent (the LLM has no memory). With history, you get a coherent multi-turn conversation.

### How message construction works

Every time `run()` is called, you build the full message list from scratch:

```
[system message]         ← if system_prompt is set
[all previous messages]  ← from self._history (user + assistant alternating)
[current user message]   ← the new input
```

After the LLM responds, you save both the user message and the assistant response to `self._history`. The next call will include them automatically.

### What to build

```python
class SimpleAgent(Agent):
    def __init__(self, name, llm, system_prompt=None, config=None):
        super().__init__(name, llm, system_prompt, config)

    def run(self, input_text: str, **kwargs) -> str:
        # 1. Build messages list
        # 2. Call self.llm.invoke(messages)
        # 3. Save to history with self.add_message()
        # 4. Return response

    def stream_run(self, input_text: str, **kwargs):
        # Generator version of run()
        # Yield chunks from self.llm.stream_invoke()
        # IMPORTANT: accumulate full response before saving to history
```

### Critical detail for `stream_run()`

You must accumulate all chunks into a `full_response` string *before* calling `add_message()`. History must store the complete text — never a partial response:

```python
full_response = ""
for chunk in self.llm.stream_invoke(messages, **kwargs):
    full_response += chunk
    yield chunk

# Only after the loop ends:
self.add_message(Message(input_text, "user"))
self.add_message(Message(full_response, "assistant"))
```

### Self-check

```python
from dotenv import load_dotenv; load_dotenv()
from myagents.core.llm import LLMClient
from myagents.agents.simple_agent import SimpleAgent

llm = LLMClient()
agent = SimpleAgent(
    name="assistant",
    llm=llm,
    system_prompt="You are a helpful assistant. Keep answers to one sentence."
)

r1 = agent.run("My name is Alice.")
print(f"Turn 1: {r1}")

r2 = agent.run("What is my name?")
print(f"Turn 2: {r2}")
assert "Alice" in r2, "Agent should remember the name from turn 1"

# Test streaming
agent2 = SimpleAgent(name="streamer", llm=llm)
chunks = list(agent2.stream_run("Say 'streaming works'"))
full = "".join(chunks)
print(f"Stream: {full}")
assert len(chunks) > 1, "Should yield multiple chunks"
```

---

## Step 2 — `agents/react_agent.py`: Reasoning and Acting Loop

### Concept

ReAct (Reasoning + Acting) is the most widely-used agentic pattern. The agent operates in a loop:

```
Thought → Action → Observation → Thought → Action → Observation → ... → Finish
```

- **Thought:** The LLM reasons about what it knows and what it needs
- **Action:** The LLM decides to call a specific tool with specific input
- **Observation:** The tool runs and returns a result
- This repeats until the LLM decides it has enough information to answer

The key insight: you never give the LLM the answer — you give it tools, and it figures out which tools to call and in what order.

### The prompt structure

The LLM only works within the context you give it. Your prompt must tell it:

1. What tools are available (their names and descriptions)
2. What format to respond in (`Thought: ...` then `Action: toolname[input]`)
3. The original question
4. The history of what it has done so far (all previous Thought/Action/Observation lines)

Here is the exact prompt template you must implement:

```
You are an AI assistant with reasoning and acting capabilities.

## Available Tools
{tools}

## Response Format
You MUST respond with exactly this structure every time:
Thought: <your reasoning about what to do next>
Action: <one of the following>
  - toolname[input]   ← call a tool
  - Finish[answer]    ← when you have the final answer

## Rules
1. Every response must have both Thought and Action.
2. Tool call format must be exact: name[input] with no spaces around brackets.
3. Only use Finish when you have enough information to fully answer.
4. If a tool result is unhelpful, try a different tool or rephrase your query.

## Question
{question}

## History So Far
{history}

Now respond:
```

Store this as `DEFAULT_REACT_PROMPT` at the module level. Accept a `custom_prompt` parameter in the constructor to override it.

### Parsing LLM output

The LLM will output something like:

```
Thought: I need to find the current GDP of France.
Action: search[France GDP 2024]
```

You need to extract the `Thought` and `Action` using regex:

```python
import re

def _parse_output(self, text: str) -> tuple[Optional[str], Optional[str]]:
    thought_match = re.search(r"Thought:\s*(.*)", text)
    action_match = re.search(r"Action:\s*(.*)", text)
    thought = thought_match.group(1).strip() if thought_match else None
    action = action_match.group(1).strip() if action_match else None
    return thought, action
```

Then parse the action text into tool name and input:

```python
def _parse_action(self, action_text: str) -> tuple[Optional[str], Optional[str]]:
    match = re.match(r"(\w+)\[(.*)\]", action_text)
    if match:
        return match.group(1), match.group(2)
    return None, None

def _parse_finish(self, action_text: str) -> str:
    match = re.match(r"\w+\[(.*)\]", action_text)
    return match.group(1) if match else action_text
```

### The main loop in `run()`

```python
def run(self, input_text: str, **kwargs) -> str:
    self.current_history = []   # reset per-run state
    
    for step in range(self.max_steps):
        # 1. Build prompt
        tools_desc = self.tool_registry.get_tools_description()
        history_str = "\n".join(self.current_history)
        prompt = self.prompt_template.format(
            tools=tools_desc,
            question=input_text,
            history=history_str if history_str else "None yet."
        )
        
        # 2. Call LLM (non-streaming — you need the full output to parse it)
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages, **kwargs)
        if not response:
            break
        
        # 3. Parse output
        thought, action = self._parse_output(response)
        if thought:
            print(f"Thought: {thought}")
        if not action:
            break
        
        # 4. Check for Finish
        if action.startswith("Finish"):
            final_answer = self._parse_finish(action)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            return final_answer
        
        # 5. Execute tool
        tool_name, tool_input = self._parse_action(action)
        if not tool_name:
            self.current_history.append("Observation: Invalid action format.")
            continue
        
        observation = self.tool_registry.execute_tool(tool_name, tool_input)
        print(f"Action: {action}")
        print(f"Observation: {observation}")
        
        # 6. Update history for next iteration
        self.current_history.append(f"Thought: {thought}")
        self.current_history.append(f"Action: {action}")
        self.current_history.append(f"Observation: {observation}")
    
    # Fallback if max_steps reached
    fallback = "I could not complete this task within the step limit."
    self.add_message(Message(input_text, "user"))
    self.add_message(Message(fallback, "assistant"))
    return fallback
```

### Important: two kinds of history

- `self.current_history` — a `list[str]` that accumulates Thought/Action/Observation for the **current `run()` call only**. Reset to `[]` at the start of every `run()`.
- `self._history` — inherited from `Agent`, stores `Message` objects across **all `run()` calls**. Only appended to when returning the final answer.

Never confuse these two.

### Constructor

```python
def __init__(
    self,
    name: str,
    llm: LLMClient,
    tool_registry: ToolRegistry,
    system_prompt: Optional[str] = None,
    config: Optional[Config] = None,
    max_steps: int = 5,
    custom_prompt: Optional[str] = None,
):
    super().__init__(name, llm, system_prompt, config)
    self.tool_registry = tool_registry
    self.max_steps = max_steps
    self.current_history: list[str] = []
    self.prompt_template = custom_prompt or DEFAULT_REACT_PROMPT
```

### Self-check

```python
from dotenv import load_dotenv; load_dotenv()
from myagents.core.llm import LLMClient
from myagents.tools.registry import ToolRegistry
from myagents.tools.builtin.calculator import CalculatorTool
from myagents.agents.react_agent import ReActAgent

llm = LLMClient()
registry = ToolRegistry()
registry.register_tool(CalculatorTool())

agent = ReActAgent(
    name="math_agent",
    llm=llm,
    tool_registry=registry,
    max_steps=5
)

result = agent.run("What is 17 multiplied by 23, then add 156?")
print(f"Result: {result}")
# Should use the calculator tool and get 547
```

### Known edge cases to handle

1. **LLM doesn't follow format** — `_parse_output` returns `(None, None)`. Break the loop gracefully.
2. **Tool name in action doesn't exist** — `execute_tool` returns an error string as observation. The LLM gets it and can try again.
3. **`max_steps` reached** — return the fallback message.
4. **`Finish[]` with empty content** — `_parse_finish` should return an empty string, which the caller handles.

---

## Step 3 — `agents/reflection_agent.py`: Self-Critique and Iterative Refinement

### Concept

LLMs can critique their own outputs. This agent exploits that ability by running a generate → critique → refine loop. It is particularly effective for tasks where quality matters over speed: writing, code generation, analysis reports.

The loop:
```
Initial attempt
     ↓
Reflection (critique the attempt)
     ↓ (if no improvement needed → stop early)
Refinement (improve based on critique)
     ↓
Reflection again
     ↓
... up to max_iterations
```

### The `Memory` class

Before writing the agent, build a simple trajectory store inside the same file:

```python
class Memory:
    def __init__(self):
        self.records: list[dict] = []
        # Each record: {"type": "execution" | "reflection", "content": str}

    def add_record(self, record_type: str, content: str) -> None:
        self.records.append({"type": record_type, "content": content})

    def get_last_execution(self) -> str:
        """Find the most recent 'execution' record."""
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]
        return ""

    def get_trajectory(self) -> str:
        """Format all records as a readable log string."""
        lines = []
        for r in self.records:
            if r["type"] == "execution":
                lines.append(f"--- Attempt ---\n{r['content']}")
            elif r["type"] == "reflection":
                lines.append(f"--- Critique ---\n{r['content']}")
        return "\n\n".join(lines)
```

### The three prompt templates

Define these as a dict `DEFAULT_PROMPTS` at the module level:

**`"initial"`** — generate the first attempt:
```
Complete the following task thoroughly and accurately:

Task: {task}

Provide a complete response:
```

**`"reflect"`** — critique an attempt:
```
Review the following response to a task and identify specific issues or areas for improvement.

Task: {task}

Response to review:
{content}

Analyze this response critically. If it is already excellent with nothing to improve, reply with exactly: "No improvement needed."
Otherwise, list specific problems and concrete suggestions for improvement:
```

**`"refine"`** — improve based on feedback:
```
Improve the following response based on the critique provided.

Task: {task}

Previous response:
{last_attempt}

Critique and suggestions:
{feedback}

Write an improved response that addresses all the issues raised:
```

Accept `custom_prompts: Optional[Dict[str, str]]` in the constructor to override any or all templates.

### The `run()` method

```python
def run(self, input_text: str, **kwargs) -> str:
    self.memory = Memory()   # fresh memory each run
    
    print(f"--- Initial Attempt ---")
    initial_prompt = self.prompts["initial"].format(task=input_text)
    initial_result = self._call_llm(initial_prompt, **kwargs)
    self.memory.add_record("execution", initial_result)
    
    for i in range(self.max_iterations):
        print(f"--- Reflection {i+1}/{self.max_iterations} ---")
        
        last = self.memory.get_last_execution()
        reflect_prompt = self.prompts["reflect"].format(task=input_text, content=last)
        feedback = self._call_llm(reflect_prompt, **kwargs)
        self.memory.add_record("reflection", feedback)
        
        # Early stopping: LLM says no improvement needed
        if "no improvement needed" in feedback.lower():
            print("Agent satisfied. Stopping early.")
            break
        
        print(f"--- Refinement {i+1} ---")
        refine_prompt = self.prompts["refine"].format(
            task=input_text,
            last_attempt=last,
            feedback=feedback
        )
        refined = self._call_llm(refine_prompt, **kwargs)
        self.memory.add_record("execution", refined)
    
    final = self.memory.get_last_execution()
    self.add_message(Message(input_text, "user"))
    self.add_message(Message(final, "assistant"))
    return final

def _call_llm(self, prompt: str, **kwargs) -> str:
    messages = [{"role": "user", "content": prompt}]
    return self.llm.invoke(messages, **kwargs) or ""
```

### Self-check

```python
from dotenv import load_dotenv; load_dotenv()
from myagents.core.llm import LLMClient
from myagents.agents.reflection_agent import ReflectionAgent

llm = LLMClient()
agent = ReflectionAgent(
    name="writer",
    llm=llm,
    max_iterations=2
)

result = agent.run("Write a one-paragraph explanation of what recursion is in programming.")
print(f"\nFinal result:\n{result}")
assert len(result) > 50, "Result should be a real paragraph"
```

---

## Step 4 — `agents/plan_solve_agent.py`: Two-Phase Decomposition

### Concept

Complex problems are hard to solve in one shot. This agent breaks the problem into two explicit phases:

**Phase 1 — Plan:** Ask the LLM to decompose the question into an ordered list of simpler sub-tasks.

**Phase 2 — Solve:** Execute each sub-task sequentially, feeding earlier results as context into later steps.

This is most effective for: multi-step reasoning, research tasks, math word problems, and anything requiring sequential dependencies between subtasks.

### Build `Planner` first (a helper class in the same file)

The planner's job: call the LLM with a planning prompt and parse the response as a Python list.

**Planner prompt template (`DEFAULT_PLANNER_PROMPT`):**
```
You are an expert task planner. Break the following problem into a clear, ordered list of simple, self-contained steps.
Each step should be something that can be answered independently.
Output ONLY a Python list of strings, with no other text.

Problem: {question}

Output format:
```python
["Step 1 description", "Step 2 description", "Step 3 description"]
```
```

**The `plan(question) -> list[str]` method:**

```python
def plan(self, question: str, **kwargs) -> list[str]:
    prompt = self.prompt_template.format(question=question)
    messages = [{"role": "user", "content": prompt}]
    response = self.llm_client.invoke(messages, **kwargs) or ""
    
    try:
        # Extract the Python list from inside the ```python ... ``` block
        code_block = response.split("```python")[1].split("```")[0].strip()
        plan = ast.literal_eval(code_block)
        return plan if isinstance(plan, list) else []
    except (IndexError, ValueError, SyntaxError) as e:
        print(f"Failed to parse plan: {e}\nRaw response: {response}")
        return []
```

**Why `ast.literal_eval` and not `eval`?** `ast.literal_eval` only evaluates Python *literals* (strings, numbers, lists, dicts, tuples, booleans, None). It raises `ValueError` for anything else — including function calls, imports, or arbitrary expressions. You cannot accidentally execute `["step1", __import__('os').system('rm -rf /')]` with it.

### Build `Executor` next (another helper class in the same file)

The executor's job: execute each step one at a time, accumulating results into a `history` string that gets injected into the next step's prompt.

**Executor prompt template (`DEFAULT_EXECUTOR_PROMPT`):**
```
You are an expert at executing plans step by step.
You will be given the original problem, the full plan, what has been done so far, and your current step.
Focus ONLY on answering the current step. Be specific and concise.

Original problem: {question}

Full plan:
{plan}

Steps completed so far:
{history}

Current step to execute: {current_step}

Your answer for this step only:
```

**The `execute(question, plan, **kwargs) -> str` method:**

```python
def execute(self, question: str, plan: list[str], **kwargs) -> str:
    history = ""
    final_answer = ""
    
    for i, step in enumerate(plan, 1):
        print(f"Executing step {i}/{len(plan)}: {step}")
        prompt = self.prompt_template.format(
            question=question,
            plan="\n".join(f"{j+1}. {s}" for j, s in enumerate(plan)),
            history=history if history else "Nothing completed yet.",
            current_step=step
        )
        messages = [{"role": "user", "content": prompt}]
        result = self.llm_client.invoke(messages, **kwargs) or ""
        
        history += f"Step {i}: {step}\nResult: {result}\n\n"
        final_answer = result
        print(f"Step {i} result: {result[:100]}...")
    
    return final_answer
```

### Build `PlanAndSolveAgent`

```python
class PlanAndSolveAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[dict] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        planner_prompt = custom_prompts.get("planner") if custom_prompts else None
        executor_prompt = custom_prompts.get("executor") if custom_prompts else None
        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(self.llm, executor_prompt)

    def run(self, input_text: str, **kwargs) -> str:
        print(f"Planning...")
        plan = self.planner.plan(input_text, **kwargs)
        
        if not plan:
            answer = "Could not generate a valid plan for this task."
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(answer, "assistant"))
            return answer
        
        print(f"Plan generated ({len(plan)} steps). Executing...")
        final_answer = self.executor.execute(input_text, plan, **kwargs)
        
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer
```

### Self-check

```python
from dotenv import load_dotenv; load_dotenv()
from myagents.core.llm import LLMClient
from myagents.agents.plan_solve_agent import PlanAndSolveAgent

llm = LLMClient()
agent = PlanAndSolveAgent(name="planner", llm=llm)

result = agent.run(
    "Explain three key differences between Python lists and tuples, "
    "and give a code example for each difference."
)
print(f"\nFinal answer:\n{result}")
assert len(result) > 100, "Should be a detailed answer"
```

---

## Step 5 — `agents/__init__.py`

```python
from .simple_agent import SimpleAgent
from .react_agent import ReActAgent
from .reflection_agent import ReflectionAgent
from .plan_solve_agent import PlanAndSolveAgent

__all__ = [
    "SimpleAgent",
    "ReActAgent",
    "ReflectionAgent",
    "PlanAndSolveAgent",
]
```

### Final Phase 3 self-check

```python
from myagents.agents import SimpleAgent, ReActAgent, ReflectionAgent, PlanAndSolveAgent
print("All agent imports OK")
print(f"Agents available: {[cls.__name__ for cls in [SimpleAgent, ReActAgent, ReflectionAgent, PlanAndSolveAgent]]}")
```

---

## ✅ Phase 3 Complete Checklist

- [ ] `SimpleAgent` — message construction correct (system + history + input), both `run()` and `stream_run()`, history accumulated correctly
- [ ] `ReActAgent` — `current_history` reset each run, Thought/Action/Observation loop, regex parsing of both Thought/Action and `toolname[input]`, `Finish[answer]` detection, `max_steps` enforced
- [ ] `ReflectionAgent` — `Memory` class with `add_record` / `get_last_execution`, three prompt templates, early stopping on "no improvement needed", `max_iterations` enforced
- [ ] `PlanAndSolveAgent` — separate `Planner` and `Executor` classes, plan parsed with `ast.literal_eval` (not `eval`), executor accumulates history across steps, empty plan handled gracefully
- [ ] `agents/__init__.py` exports all four
- [ ] All self-checks pass with a real LLM response

**Next → [Phase 4: Integration, Utils, and Demo](./phase4_integration.md)**
