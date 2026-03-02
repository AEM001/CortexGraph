# base class for tools
from pydantic import BaseModel
from typing import Any,Dict,List
from abc import ABC,abstractmethod

class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None

class Tool(ABC):
    def __init__(self,name,description):
        self.name=name
        self.description=description

    @abstractmethod
    def run(self,parameters:Dict[str,Any])->str:
        pass
    @abstractmethod
    def get_parameters(self)->List[ToolParameter]:
        pass

    def validate_parameters(self,parameters:Dict[str,Any])->bool:
        required_params=[p.name for p in self.get_parameters() if p.required]
        return all(param in parameters for param in required_params)

    def to_dict(self)->Dict[str,Any]:
        return {
            "name":self.name,
            "description":self.description,
            "parameters":[param.dict() for param in self.get_parameters()]

        }

    def __str__(self)->str:
        return f"Tool(name={self.name})"

    def __repr__(self)->str:
        return self.__str__()

# # A minimal concrete tool for testing:
# class Calculate(Tool):
#     def __init__(self):
#         super().__init__(name="calculate",description="calculate two numbers")

#     def run(self,parameters:Dict[str,Any])->str:
#         a=parameters["a"]
#         b=parameters['b']
#         return str(a+b)

#     def get_parameters(self):
#         return [
#             ToolParameter(name="a",type="integer",description="first number"),
#             ToolParameter(name="b",type="integer",description='Seconde integer')
#         ]

# cal=Calculate()
# assert cal.run({"a":3,"b":4})=='7'
# assert cal.validate_parameters({"a":3,"b":4})==True
# assert cal.validate_parameters({"a":3})==False
# assert cal.validate_parameters({})==False
# assert cal.to_dict()["name"]=="calculate"
# print("tool base ok")