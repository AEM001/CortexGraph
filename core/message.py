from pydantic import BaseModel
from typing import Optional, Dict,Any, Literal
from datetime import datetime

MessageRole=Literal["user","system","tool","assistant"]
class Messages(BaseModel):
    role:MessageRole
    content:str
    timestamp:datetime=None
    metadata:Optional[Dict[str,Any]]=None

    def __init__(self,role:MessageRole,content:str,**kwargs):
        super().__init__(
            role=role,
            content=content,
            timestamp=kwargs.get('timestamp',datetime.now()),
            metadata=kwargs.get('metadata',{})
        )
    
    def to_dict(self)->Dict[str,Any]:
        return{
            "role":self.role,
            "content":self.content
        }

    def __str__(self)->str:
        return f"[{self.role}] {self.content}"

m=Messages(role="user",content="how to make a love")
assert m.role=='user'
assert m.content=="how to make a love"
assert m.timestamp is not None
assert m.to_dict()=={'role':'user','content':'how to make a love'}

try:
    Messages(content="what is titties",role="fucker")
except ValueError as e:
    print(f"correctly rejected:{e}")