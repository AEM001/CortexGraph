from ..base import Tool,ToolParameter
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

# Load environment variables from core/.env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', 'core', '.env'))
class SearchTool(Tool):
    def __init__(self,backend:str="hybrid"):
        super().__init__(name="search",description="Search the web for information")
        self.backend=backend
        self.tavily_key=os.getenv("TAVILY_API_KEY")
        self.serpapi_key=os.getenv("SERPAPI_API_KEY")
        self.available_backends=[]
        self._setup_backends()
    
    def _setup_backends(self):
        if self.tavily_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=self.tavily_key)
                self.available_backends.append("tavily")
                print("✅ Tavily search engine has been initalized")
            except ImportError:
                print("⚠️ Tavily is not installed, cannot use Tavily search")
        if self.serpapi_key:
            try:
                import serpapi
                self.available_backends.append("serpapi")
                print("✅ SerpApi search engine has been initalized")
            except ImportError:
                print("⚠️ SerpApi is not installed, cannot use SerpApi search")

        if self.backend =="hybrid":
            if self.available_backends:
                print(f"🔧 Hybrid search mode enabled, available backends: {', '.join(self.available_backends)}")
            else:
                print("⚠️ No available search backends, please configure API keys")
        elif self.backend == "tavily" and "tavily" not in self.available_backends:
            print("⚠️ Tavily is not available, please check TAVILY_API_KEY configuration")
        elif self.backend == "serpapi" and "serpapi" not in self.available_backends:
            print("⚠️ SerpApi is not available, please check SERPAPI_API_KEY configuration")

    def run(self, parameters:Dict[str,Any])->str:
        query=parameters.get("input","").strip()
        if not query:
            return "wrong, can't search"
        try:
            if self.backend=="hybrid":
                return self._search_hybrid(query)
            elif self.backend=="tavily":
                if "tavily" not in self.available_backends:
                    return self._get_api_config_message()
                return self._search_tavily(query)
            elif self.backend=="serpapi":
                if "serpapi" not in self.available_backends:
                    return self._get_api_config_message()
                return self._search_serpapi(query)
            else:
                return self._get_api_config_message()
        except Exception as e:
            return f"search error: {str(e)}"

    def _search_hybrid(self,query:str)->str:
        if not self.available_backends:
            return self._get_api_config_message()

        if "tavily" in self.available_backends:
            try:
                print("using tavily")
                return self._search_tavily(query)
            except Exception as e:
                print(f"⚠️ Tavily search failed: {e}")
                if "serpapi" not in self.available_backends:
                    print("switch to api config message")
                    return self._search_serpapi(query)
        elif "serpapi" in self.available_backends:
            try:
                print("using serpapi")
                return self._search_serpapi(query)
            except Exception as e:
                print(f"⚠️ SerpApi search failed: {e}")
                print("switch to api config message")
                return self._get_api_config_message()
        return self._get_api_config_message()

    def _search_tavily(self, query: str) -> str:
        response = self.tavily_client.search(
            query=query,
            search_depth="basic",
            include_answer=True,
            max_results=3
        )

        result = f"🎯 Tavily AI search result: {response.get('answer', 'no direct answer')}\n\n"

        for i, item in enumerate(response.get('results', [])[:3], 1):
            result += f"[{i}] {item.get('title', '')}\n"
            result += f"    {item.get('content', '')[:200]}...\n"
            result += f"    source: {item.get('url', '')}\n\n"

        return result

    def _search_serpapi(self, query: str) -> str:
        """Search using SerpApi"""
        try:
            from serpapi import SerpApiClient
        except ImportError:
            return "Error: SerpApi not installed, please run pip install serpapi"

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
            "gl": "cn",
            "hl": "zh-cn",
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        result_text = "🔍 SerpApi Google search results:\n\n"

        # Smart parsing: prioritize finding the most direct answer
        if "answer_box" in results and "answer" in results["answer_box"]:
            result_text += f"💡 Direct answer: {results['answer_box']['answer']}\n\n"

        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            result_text += f"📖 Knowledge graph: {results['knowledge_graph']['description']}\n\n"

        if "organic_results" in results and results["organic_results"]:
            result_text += "🔗 Related results:\n"
            for i, res in enumerate(results["organic_results"][:3], 1):
                result_text += f"[{i}] {res.get('title', '')}\n"
                result_text += f"    {res.get('snippet', '')}\n"
                result_text += f"    Source: {res.get('link', '')}\n\n"
            return result_text

        return f"Sorry, no information found about '{query}'."

    def _get_api_config_message(self) -> str:
        """Get API configuration message"""
        tavily_key = os.getenv("TAVILY_API_KEY")
        serpapi_key = os.getenv("SERPAPI_API_KEY")

        message = "❌ No available search sources, please check the following configurations:\n\n"

        # Check Tavily
        message += "1. Tavily API:\n"
        if not tavily_key:
            message += "   ❌ Environment variable TAVILY_API_KEY not set\n"
            message += "   📝 Get it at: https://tavily.com/\n"
        else:
            try:
                import tavily
                message += "   ✅ API key configured, package installed\n"
            except ImportError:
                message += "   ❌ API key configured, but package needs to be installed: pip install tavily-python\n"

        message += "\n"

        # Check SerpAPI
        message += "2. SerpAPI:\n"
        if not serpapi_key:
            message += "   ❌ Environment variable SERPAPI_API_KEY not set\n"
            message += "   📝 Get it at: https://serpapi.com/\n"
        else:
            try:
                import serpapi
                message += "   ✅ API key configured, package installed\n"
            except ImportError:
                message += "   ❌ API key configured, but package needs to be installed: pip install google-search-results\n"

        message += "\nConfiguration methods:\n"
        message += "- Add to .env file: TAVILY_API_KEY=your_key_here\n"
        message += "- Or set in environment variable: export TAVILY_API_KEY=your_key_here\n"
        message += "\nRerun the program after configuration."

        return message

    def get_parameters(self) -> List[ToolParameter]:
        """Get tool parameter definitions"""
        return [
            ToolParameter(
                name="input",
                type="string",
                description="Search query keywords",
                required=True
            )
        ]

# Convenience functions
def search(query: str, backend: str = "hybrid") -> str:
    """
    Convenient search function

    Args:
        query: Search query keywords
        backend: Search backend ("hybrid", "tavily", "serpapi")

    Returns:
        Search results
    """
    tool = SearchTool(backend=backend)
    return tool.run({"input": query})

# Dedicated search functions
def search_tavily(query: str) -> str:
    """Use Tavily for AI-optimized search"""
    tool = SearchTool(backend="tavily")
    return tool.run({"input": query})

def search_serpapi(query: str) -> str:
    """Use SerpApi for Google search"""
    tool = SearchTool(backend="serpapi")
    return tool.run({"input": query})

def search_hybrid(query: str) -> str:
    """Smart hybrid search, automatically selects the best search source"""
    tool = SearchTool(backend="hybrid")
    return tool.run({"input": query})

if __name__=="__main__":
    tool=SearchTool()
    result=tool.run({"input":"what is claude code"})
    print(f"Search result (may be error if no keys): {result[:100]}")