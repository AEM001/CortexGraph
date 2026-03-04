"""Reflection Agent Implementation - Self-reflection and Iterative Optimization Agent"""

from typing import Optional, List, Dict, Any
from ..core.agent import Agent
from ..core.llm import llm
from ..core.config import Config
from ..core.message import Message

# Default prompt templates
DEFAULT_PROMPTS = {
    "initial": """
Please complete the task according to the following requirements:

Task: {task}

Please provide a complete and accurate answer.
""",
    "reflect": """
Please carefully review the following answer and identify any potential issues or areas for improvement:

# Original task:
{task}

# Current answer:
{content}

Please analyze the quality of this answer, identify any shortcomings, and provide specific improvement suggestions.
If the answer is already good, please respond with "No improvements needed".
""",
    "refine": """
Please improve your answer based on the feedback:

# Original task:
{task}

# Last attempt:
{last_attempt}

# Feedback:
{feedback}

Please provide an improved answer.
"""
}

class Memory:
    """
    Simple short-term memory module for storing the agent's actions and reflections.
    """
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """Add a new record to memory"""
        self.records.append({"type": record_type, "content": content})
        print(f"📝 Memory updated, added a '{record_type}' record.")

    def get_trajectory(self) -> str:
        """Format all memory records into a coherent text string"""
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- Previous attempt (code) ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- Reviewer feedback ---\n{record['content']}\n\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        """Get the most recent execution result"""
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return ""

class ReflectionAgent(Agent):
    """
    Reflection Agent - Self-reflection and iterative optimization agent

    This Agent can:
    1. Execute initial tasks
    2. Reflect on results
    3. Optimize based on reflection
    4. Iterate improvements until satisfied

    Particularly suitable for code generation, document writing, analysis reports, and other tasks that require iterative optimization.

    Supports multiple professional field prompt templates, users can customize or use built-in templates.
    """

    def __init__(
        self,
        name: str,
        llm: llm,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        Initialize ReflectionAgent

        Args:
            name: Agent name
            llm: LLM instance
            system_prompt: System prompt
            config: Configuration object
            max_iterations: Maximum number of iterations
            custom_prompts: Custom prompt templates {"initial": "", "reflect": "", "refine": ""}
        """
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.memory = Memory()

        # Set prompt templates: user-defined takes priority, otherwise use default templates
        self.prompts = custom_prompts if custom_prompts else DEFAULT_PROMPTS
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        Run Reflection Agent

        Args:
            input_text: Task description
            **kwargs: Other parameters

        Returns:
            Final optimized result
        """
        print(f"\n🤖 {self.name} starts processing task: {input_text}")

        # Reset memory
        self.memory = Memory()

        # 1. Initial execution
        print("\n--- Performing initial attempt ---")
        initial_prompt = self.prompts["initial"].format(task=input_text)
        initial_result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution", initial_result)

        # 2. Iteration loop: reflection and optimization
        for i in range(self.max_iterations):
            print(f"\n--- Round {i+1}/{self.max_iterations} ---")

            # a. Reflect
            print("\n-> Reflecting...")
            last_result = self.memory.get_last_execution()
            reflect_prompt = self.prompts["reflect"].format(
                task=input_text,
                content=last_result
            )
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            # b. Check if we need to stop
            if "No improvements needed" in feedback or "no need for improvement" in feedback.lower():
                print("\n✅ Reflection indicates the result needs no improvement, task completed.")
                break

            # c. Optimize
            print("\n-> Optimizing...")
            refine_prompt = self.prompts["refine"].format(
                task=input_text,
                last_attempt=last_result,
                feedback=feedback
            )
            refined_result = self._get_llm_response(refine_prompt, **kwargs)
            self.memory.add_record("execution", refined_result)

        final_result = self.memory.get_last_execution()
        print(f"\n--- Task completed ---\nFinal result:\n{final_result}")

        # Save to history
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))

        return final_result
    
    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """Call LLM and get complete response"""
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs) or ""
