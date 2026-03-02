import asyncio
import concurrent.futures
from typing import Dict,Any,List,Callable,Optional
from registry import ToolRegistry

class AsyncToolExecutor:
    def __init__(self,registry:ToolRegistry, max_workers:int=4):
        self.registry=registry
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)