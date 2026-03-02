from typing import Optional, Any, Callable
from core.exceptions import HelloAgentsException
from base import Tool

class ToolRegistry:
    def __init__(self):
        self.tools:dict[str,Tool]={}
        self.functions:dict[str,dict[str,Any]]={}

    def register_tool(self,tool:Tool):
        if tool.name in self.tools:
            print(f"Warning:Tool '{tool.name} already exists and wil be overwritten.")
        self.tools[tool.name]=tool
        print(f"Tool {tool.name} has been registered")

    def register_function(self,name:str,description:str,func:Callable[[str],str]):
        if name in self.functions:
            print(f"Warning, Tool '{name} already exists and will be overwirtten'")
        self.functions[name]={
            "description":description,
            "func":func
        }
        print(f"Tool '{name}'registered")

    def unregister(self,name:str):
        if name in self.tools:
            del self.tools[name]
            print(f"Tool '{name}' unregistered")
        elif name in self.functions:
            del self.functions[name]
            print(f"Tool '{name}' unregistered")
        else:
            print(f"Tool '{name} does not exist'")

    def get_tool(self,name:str)->Optional[Tool]:
        return self.tools.get(name)

    def get_function(self,name:str)->Optional[Callable]:
        func_info=self.functions.get(name)
        return func_info["func"] if func_info else None

    def execute_tool(self,name:str,input_text:str)->str:
        if name in self.tools:
            tool=self.tools[name]
            try:
                return tool.run({"input":input_text})
            except Exception as e:
                return f"Error: Exception occurred while executing tool '{name}': {str(e)}"

        elif name in self.functions:
            func=self.functions[name]["func"]
            try:
                return func(input_text)
            except Exception as e:
                return f"Error: Exception occurred while executing tool '{name}': {str(e)}"
        else:
            return f"Error: Tool named '{name}' not found."

    def get_tools_description(self)->str:
        descritption=[]
        for tool in self.tools.values():
            descritption.append(f"- {tool.name}: {tool.description}")

        for name,info in self.functions.items():
            descritption.append(f"- {name}: {info['description']}")
        return "\n".join(descritption) if descritption else "No available tools"

    def list_tools(self)->list[str]:
        return list(self.tools.keys())+list(self.funtions.key())

    def get_all_tools(self)->list[Tool]:
        return list(self.tools.values())

    def clear(self):
        self.tools.clear()
        self.functions.clear()
        print("All tools cleared.")

global_registry=ToolRegistry()        
        
    
