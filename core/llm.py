from exception import AgentFramworkError
from dotenv import load_dotenv
import os
from openai import OpenAI
from typing import Optional
import requests
import json

load_dotenv()

class llm:
    def __init__(
    self,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    # timeout: Optional[int] = None,
):
        self.provider=self.__auto_detect_provider()
        self.temperature=temperature or os.getenv("TEMPERATURE", "0.7")
        self.max_tokens=max_tokens or os.getenv("MAX_TOKENS")
        # self.timeout=os.getenv("TIMEOUT")
        self.api_key,self.base_url=self.__resolve_credentials()
        self.client=self._create_client()
        self.model=self._get_default_model()

    def __auto_detect_provider(self)->str:
        if os.getenv("OLLAMA_API_KEY"):
            return "ollama"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"

        
    def __resolve_credentials(self)->tuple[str,str]:
        if self.provider=="ollama":
            return os.getenv("OLLAMA_API_KEY"),os.getenv("OLLAMA_BASE_URL")
        elif self.provider=="openai":
            return os.getenv("OPENAI_API_KEY"),os.getenv("OPENAI_BASE_URL")
    
    def _create_client(self)->OpenAI:
        return OpenAI(api_key=self.api_key,
        base_url=self.base_url,
        )

    def _get_default_model(self)->str:
        if self.provider=="ollama":
            return "llama3.2:3b"
        elif self.provider=="openai":
            return "gpt-4.1-nano"

    def think(self, messages:list[dict[str,str]],tempereature:Optional[float]=None,max_tokens:Optional[int]=None):
        print(f"calling {self.provider} with {messages}")
        if self.provider == "ollama":
            return self._think_ollama(messages, tempereature, max_tokens)
        else:
            return self._think_openai(messages, tempereature, max_tokens)
    
    def _think_ollama(self, messages:list[dict[str,str]],tempereature:Optional[float]=None,max_tokens:Optional[int]=None):
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": tempereature or self.temperature,
            "stream": True
        }
        if max_tokens or self.max_tokens:
            data["max_tokens"] = max_tokens or self.max_tokens
        try:
            print("LLM response:")
            full_content = ""
            with requests.post(url, headers=headers, json=data, stream=True) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            json_str = line_text[6:]
                            if json_str == '[DONE]':
                                break
                            try:
                                chunk = json.loads(json_str)
                                content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                if content:
                                    print(content, end='', flush=True)
                                    full_content += content
                            except json.JSONDecodeError:
                                pass
            print()
            return full_content
        except Exception as e:
            print(f"LLM call failed: {e}")
            raise AgentFramworkError(f"LLM calling failed: {str(e)}")
    
    def _think_openai(self, messages:list[dict[str,str]],tempereature:Optional[float]=None,max_tokens:Optional[int]=None):
        try:
            response=self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=tempereature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True,
            )
            print("LLM response:")
            full_content = ""
            for chunk in response:
                content=chunk.choices[0].delta.content or ""
                if content:
                    print(content,end="",flush=True)
                    full_content += content
            print()
            return full_content
        except Exception as e:
            print(f"LLM call failed: {e}")
            raise AgentFramworkError(f"LLM calling failed: {str(e)}")

    def invoke(self,messages:list[dict[str,str]],tempereature:Optional[float]=None,max_tokens:Optional[int]=None)->str:
        try:
            response=self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=tempereature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM call failed: {e}")
            raise AgentFramworkError(f"LLM calling failed: {str(e)}")


llm=llm()
llm.think([
    {"role":"user","content":"how to make love"}
])