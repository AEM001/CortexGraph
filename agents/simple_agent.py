"""Simple Agent Implementation - Based on OpenAI native API"""

from typing import Optional, Iterator

from ..core.agent import Agent
from ..core.llm import llm
from ..core.config import Config
from ..core.message import Message

class SimpleAgent(Agent):
    """Simple conversation Agent"""
    
    def __init__(
        self,
        name: str,
        llm: llm,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        super().__init__(name, llm, system_prompt, config)
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        Run simple Agent
        
        Args:
            input_text: User input
            **kwargs: Other parameters
            
        Returns:
            Agent response
        """
        # Build message list
        messages = []
        
        # Add system message
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add history messages
        for msg in self.history:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current user message
        messages.append({"role": "user", "content": input_text})
        
        # Call LLM
        response = self.llm.invoke(messages, **kwargs)
        
        # Save to history
        self.add_message(Message("user", input_text))
        self.add_message(Message("assistant", response))
        
        return response
    
    def stream_run(self, input_text: str, **kwargs):
        """
        Stream run Agent
        
        Args:
            input_text: User input
            **kwargs: Other parameters
            
        Yields:
            Agent response fragments
        """
        # Build message list
        messages = []
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        for msg in self.history:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": input_text})
        
        # Stream call LLM
        full_response = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk
        
        # Save complete conversation to history
        self.add_message(Message("user", input_text))
        self.add_message(Message("assistant", full_response))
