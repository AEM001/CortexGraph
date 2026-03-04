"""ReAct Agent Implementation - Reasoning and Acting Agent"""

import re
from typing import Optional, List, Dict, Any, Tuple
from ..core.agent import Agent
from ..core.llm import llm
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry

# Default ReAct prompt template
DEFAULT_REACT_PROMPT = """You are an AI assistant with reasoning and action capabilities. You can analyze problems by thinking, then call appropriate tools to get information, and finally provide accurate answers.

## Available Tools
{tools}

## Workflow
Please strictly follow the format below for each response, executing only one step at a time:

**Thought:** Analyze the current problem, think about what information you need or what action to take.
**Action:** Choose an action, format must be one of the following:
- `{{tool_name}}[{{tool_input}}]` - Call specified tool
- `Finish[final_answer]` - When you have enough information to provide the final answer

## Important Notes
1. Each response must include both Thought and Action parts
2. Tool call format must strictly follow: tool_name[parameter]
3. Only use Finish when you are confident you have enough information to answer the question
4. If the tool returns insufficient information, continue using other tools or different parameters for the same tool

## Current Task
**Question:** {question}

## Execution History
{history}

Now start your reasoning and action:"""

class ReActAgent(Agent):
    """
    ReAct (Reasoning and Acting) Agent
    
    An agent that combines reasoning and action, capable of:
    1. Analyzing problems and creating action plans
    2. Calling external tools to get information
    3. Reasoning based on observation results
    4. Iterating execution until reaching a final answer
    
    This is a classic agent paradigm, particularly suitable for tasks requiring external information.
    """
    
    def __init__(
        self,
        name: str,
        llm: llm,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None
    ):
        """
        Initialize ReActAgent

        Args:
            name: Agent name
            llm: LLM instance
            tool_registry: Tool registry
            system_prompt: System prompt
            config: Configuration object
            max_steps: Maximum execution steps
            custom_prompt: Custom prompt template
        """
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []

        # Set prompt template: user-defined takes priority, otherwise use default template
        self.prompt_template = custom_prompt if custom_prompt else DEFAULT_REACT_PROMPT
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        Run ReAct Agent
        
        Args:
            input_text: User question
            **kwargs: Other parameters
            
        Returns:
            Final answer
        """
        self.current_history = []
        current_step = 0
        
        print(f"\n🤖 {self.name} starts processing question: {input_text}")
        
        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- Step {current_step} ---")
            
            # Build prompt
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str
            )
            
            # Call LLM
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.invoke(messages, **kwargs)
            
            if not response_text:
                print("❌ Error: LLM failed to return valid response.")
                break
            
            # Parse output
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"🤔 Thought: {thought}")
            
            if not action:
                print("⚠️ Warning: Could not parse valid Action, process terminated.")
                break
            
            # Check if completed
            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                print(f"🎉 Final Answer: {final_answer}")
                
                # Save to history
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                
                return final_answer
            
            # Execute tool call
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or tool_input is None:
                self.current_history.append("Observation: Invalid Action format, please check.")
                continue
            
            print(f"🎬 Action: {tool_name}[{tool_input}]")
            
            # Call tool
            observation = self.tool_registry.execute_tool(tool_name, tool_input)
            print(f"👀 Observation: {observation}")
            
            # Update history
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")
        
        print("⏰ Maximum steps reached, process terminated.")
        final_answer = "Sorry, I was unable to complete this task within the step limit."
        
        # Save to history
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        
        return final_answer
    
    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse LLM output, extract thought and action"""
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        
        return thought, action
    
    def _parse_action(self, action_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse action text, extract tool name and input"""
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None
    
    def _parse_action_input(self, action_text: str) -> str:
        """Parse action input"""
        match = re.match(r"\w+\[(.*)\]", action_text)
        return match.group(1) if match else ""
