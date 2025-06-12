import asyncio
import httpx
import platform
import socket
from langchain_mcp_adapters.client import MultiServerMCPClient
from recipe import Recipe
from trustcall import create_extractor
from dotenv import load_dotenv
from langchain_google_generativeai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()  # Load environment variables from .env file

async def get_tools():
    client = MultiServerMCPClient({
        "ChefTools MCP API Server": {
            "url": "http://localhost:8000/sse",
            "transport": "sse",
            # "headers": {"Authorization": "Bearer ..." }  # if you use auth
        }
    })
    tools = await client.get_tools()  # async fetch of tool definitions :contentReference[oaicite:2]{index=2}

    print("✅ Discovered tools:", [t.name for t in tools])
    return tools

def intialize_llm(recipe:Recipe,provider:str="google"):
    if provider == "google":
        # Google Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",max_retries=4,rate_limiter=rate_limiter ,temperature=0)
    elif provider == "groq":
        llm = ChatGroq(model="qwen-qwq-32b", temperature=0)
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")
    
    
    llm_with_tools = llm.bind_tools(get_tools())
    llm_with_structured = create_extractor(
        llm,
        tools=[recipe],
        tool_choice="recipe",
        enable_inserts=True,
    )
    return llm_with_tools,llm_with_structured
    
if __name__ == "__main__":
    asyncio.run(get_tools())
    asyncio.run(register_agent())