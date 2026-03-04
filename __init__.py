from .version import __version__, __author__, __email__, __description__

from .core.llm import llm
from .core.config import Config
from .core.message import Message
from .core.exception import AgentFramworkError

from .agents.simple_agent import SimpleAgent

# Alias for compatibility
HelloAgentsLLM = llm

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",

    "llm",
    "Config", 
    "Message",
    "AgentFramworkError",
    "SimpleAgent",
    "HelloAgentsLLM",
]

