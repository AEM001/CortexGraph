from pydantic import BaseModel
from typing import Optional,Dict,Any
import dotenv
import os

class Config(BaseModel):

    default_model:str="gpt-4.1-nano"
    default_provider:str="openai"
    default_model_base_url:str="https://api.apicore.ai/v1/"
    
    
    max_history_length:int=100
    temperature:float=0.7
    debug:bool=False
    log_level:str="INFO"

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    

    @classmethod
    def from_env(cls)->"Config":
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )

    def to_dict(self)->Dict[str,Any]:
        return self.model_dump()


# cfg=Config.from_env()
# assert cfg.temperature==0.7
# assert cfg.debug==False
# assert cfg.log_level=="INFO"