from abc import ABC, abstractmethod
from typing import Optional, Any
from .message import Message
from .llm import llm
from .config import Config

class Agent(ABC):
    def __init__(self,name:str,llm:llm,system_prompt:Optional[str]=None,config:Optional[Config]=None):
        self.name=name
        self.llm=llm
        self.system_prompt=system_prompt
        self.config=config or Config()
        self.history:list[Message]=[]

    @abstractmethod
    def run(self,input_text:str,**kwargs)->str:
        pass
    def add_message(self,message:Message):
        self.history.append(message)
    def clear_history(self):
        self.history.clear()
    def get_history(self)->list[Message]:
        return self.history.copy()
    def __str__(self)->str:
        return f"Agent(name={self.name},provider={self.llm.provider})"
    def __repr__(self)->str:
        return self.__str__()
