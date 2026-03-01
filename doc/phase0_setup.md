# Phase 0 — Environment Setup & Project Scaffold

> **Goal:** Get your environment ready and create the skeleton folder structure.
> **Time:** ~30 minutes
> **Next:** [phase1_core.md](./phase1_core.md)

---

## Step 0.1 — Prerequisites

Make sure you have:

- Python 3.10 or higher (`python --version`)
- `pip` or `uv` for package management
- An API key for at least one LLM provider (OpenAI, DeepSeek, or a local Ollama installation)

---

## Step 0.2 — Create Your Project

Create a new directory for your framework. Name it whatever you like — this guide uses `myagents`.

```bash
mkdir myagents
cd myagents
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

---

## Step 0.3 — Install Dependencies

Create a `requirements.txt` with the following content:

```
openai>=1.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
tavily-python>=0.3.0
```

Then install:

```bash
pip install -r requirements.txt
```

**What each package is for:**

| Package         | Purpose                                                              |
| --------------- | -------------------------------------------------------------------- |
| `openai`        | The official SDK — used to call *any* OpenAI-compatible API endpoint |
| `pydantic`      | Data validation and settings management via Python type annotations  |
| `python-dotenv` | Loads `.env` files into environment variables for local development  |
| `tavily-python` | Optional: web search via the Tavily AI search API                    |

---

## Step 0.4 — Create the Folder Skeleton

Run these commands to create all the directories and empty `__init__.py` files:

```bash
mkdir -p myagents/core
mkdir -p myagents/tools/builtin
mkdir -p myagents/agents
mkdir -p myagents/utils

touch myagents/__init__.py
touch myagents/core/__init__.py
touch myagents/tools/__init__.py
touch myagents/tools/builtin/__init__.py
touch myagents/agents/__init__.py
touch myagents/utils/__init__.py
```

Your directory should now look like:

```
myagents/
├── __init__.py
├── core/
│   └── __init__.py
├── tools/
│   ├── __init__.py
│   └── builtin/
│       └── __init__.py
├── agents/
│   └── __init__.py
└── utils/
    └── __init__.py
```

---

## Step 0.5 — Create Your `.env` File

At the root of your project (next to the `myagents/` folder), create a `.env` file:

```ini
# Fill in the credentials for whichever provider you have access to.
# The framework will auto-detect which provider to use.

# Option A: OpenAI
# OPENAI_API_KEY=sk-...

# Option B: DeepSeek
# DEEPSEEK_API_KEY=sk-...

# Option C: Local Ollama (no key needed)
# LLM_BASE_URL=http://localhost:11434/v1
# OLLAMA_API_KEY=ollama
# LLM_MODEL_ID=llama3.2

# Option D: Generic (any OpenAI-compatible endpoint)
# LLM_API_KEY=your-key
# LLM_BASE_URL=https://your-endpoint/v1
# LLM_MODEL_ID=your-model-name
```

**Critical rule:** Never commit `.env` to version control. Add it to `.gitignore` right now:

```bash
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".venv/" >> .gitignore
```

---

## Step 0.6 — Create a Top-Level Test Script

Create `test_import.py` at the project root:

```python
# test_import.py
import myagents
print("Import successful!")
print(dir(myagents))
```

Run it:

```bash
python test_import.py
```

You should see `Import successful!` and an empty-ish list. This confirms Python can find your package. The list will grow as you build each phase.

---

## ✅ Phase 0 Checklist

Before moving on, confirm:

- [x] `python --version` shows 3.10+
- [x] Virtual environment is active
- [x] All packages installed (`pip list | grep openai` shows a result)
- [x] Folder structure matches the tree above
- [x] `.env` file exists with at least one provider configured
- [x] `.gitignore` excludes `.env`
- [x] `python test_import.py` prints `Import successful!`

---

**Next → [Phase 1: Build the Core Layer](./phase1_core.md)**
