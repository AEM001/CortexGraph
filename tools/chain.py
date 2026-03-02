from typing import List, Dict, Any, Optional
from registry import ToolRegistry

class ToolChain:
    def __init__(self,name:str,description:str):
        self.name=name
        self.descriptino=description
        self.steps:List[Dict[str,Any]]=[]

    def add_step(self,tool_name:str,input_template:str,output_key:str=None):
        step={
            "tool_name":tool_name,
            "input_template":input_template,
            "output_key":output_key or f"step_{len(self.steps)}_result"
        }
        self.steps.append(step)
        print(f"Tool chain '{self.name}' added step: {tool_name}")

    def execute(self,toolregistry:ToolRegistry,input_data:str,context:Dict[str,Any]=None)->str:
        if self.steps==None:
            raise ValueError("Tool chain is empty")
        # for step in self.steps:
        #     tool=toolregistry.get_tool(step['tool_name'])
        #     if tool is None:
        #         raise ValueError(f"Tool '{step['tool_name']}' not found")
        #     try:
        #         result=tool.run({"input":step['input_template'].format(input=input_data)})
        #         input_data=result
        #     except Exception as e:
        #         raise ValueError(f"Error executing tool '{step['tool_name']}': {str(e)}")
        # return input_data
        if context is None:
            context={}
        context["input"]=input_data
        final_result=input_data
        for i, step in enumerate(self.steps):
            tool_name=step['tool_name']
            input_template=step["input_template"]
            output_key=step["output_key"]

            print(f"exectuting step{i+1}/{len(self.steps)}:{tool_name}")
            try:
                actual_input=input_template.format(**context)
                # I am very confused about this, the whole format thing
            except KeyError as e:
                return f"Template variable replacement failed:{e}"

            try:
                result=toolregistry.execute_tool(tool_name,actual_input)
                context[output_key]=result
                final_result=result
                print(f"Step {i+1} completed")
            except Exception as e:
                return f"Error executing tool '{tool_name}': {str(e)}"

        print(f"tool chain '{self.name}' executed successfully")
        return final_result


class ToolChainManager:
    """Tool Chain Manager"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: Dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain):
        """Register a tool chain"""
        self.chains[chain.name] = chain
        print(f"✅ Tool chain '{chain.name}' registered")

    def execute_chain(self, chain_name: str, input_data: str, context: Dict[str, Any] = None) -> str:
        """Execute the specified tool chain"""
        if chain_name not in self.chains:
            return f"❌ Tool chain '{chain_name}' does not exist"

        chain = self.chains[chain_name]
        return chain.execute(self.registry, input_data, context)

    def list_chains(self) -> List[str]:
        """List all registered tool chains"""
        return list(self.chains.keys())

    def get_chain_info(self, chain_name: str) -> Optional[Dict[str, Any]]:
        """Get tool chain information"""
        if chain_name not in self.chains:
            return None
        
        chain = self.chains[chain_name]
        return {
            "name": chain.name,
            "description": chain.description,
            "steps": len(chain.steps),
            "step_details": [
                {
                    "tool_name": step["tool_name"],
                    "input_template": step["input_template"],
                    "output_key": step["output_key"]
                }
                for step in chain.steps
            ]
        }
