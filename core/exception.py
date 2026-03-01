class AgentFramworkError(Exception):
    """Base exception for all agent errors."""
    def __init__(self,message:str|None=None):
        self.message=message or "An Agent error occurred"
        super().__init__(self.message)
    

class LLMError(AgentFramworkError):
    """Exception for LLM errors."""
    pass

class AgentError(AgentFramworkError):
    """Exception for agent errors."""
    pass
class ConfigError(AgentFramworkError):
    """Exception for config errors."""
    pass
class ToolError(AgentFramworkError):
    """Exception for tool call"""
    pass

# raise ConfigError("config error")

# try:
#     raise LLMError("test error")
# except AgentFramworkError as e:
#     print(f"caught as base type:{e}")

# try:
#     raise ToolError("Tool broke")
# except LLMError:
#     print("llm error")
# except ToolError as e:
#     print(f"caught as correct type:{e}")