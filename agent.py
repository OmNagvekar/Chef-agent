import asyncio
import httpx
import platform
import socket
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.resources import get_mcp_resource
from mcp import ClientSession
from mcp.client.streamable_http  import streamablehttp_client
from recipe import Recipe
from trustcall import create_extractor
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph, MessagesState,END
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
import logging
import os
import time
import threading
import sys
from datetime import datetime, timedelta

load_dotenv()  # Load environment variables from .env file

# Set up logging
TODAY_DATE = datetime.now().strftime("%Y-%m-%d")
LOG_DIR='Logs'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"Client_logs_{TODAY_DATE}.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE,encoding="utf-8"),  # Save logs locally
        logging.StreamHandler(sys.stdout),  # Also print logs in console
    ],
)

# Initialize logger
logger = logging.getLogger(__name__)

# Function to delete old log files
def delete_old_logs():
    """Delete log files older than 1 day if they contain no errors (or only a specific memory error),
    and delete files older than 3 days regardless. Files in use are skipped.
    """
    while True:
        try:
            now = datetime.now()
            for filename in os.listdir(LOG_DIR):
                if filename.startswith("Client_logs_") and filename.endswith(".log"):
                    file_path = os.path.join(LOG_DIR, filename)

                    # Extract date from filename (assumes format "logs_YYYY-MM-DD.log")
                    file_date_str = filename[12:22]
                    try:
                        file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                    except Exception as e:
                        logging.error("Failed to parse date from filename %s: %s", filename, e)
                        continue

                    # Process files older than 1 day
                    if now - file_date > timedelta(days=1):
                        try:
                            with open(file_path, "r", encoding="utf-8") as log_file:
                                content = log_file.read()
                        except Exception as e:
                            logging.error("Error reading file %s: %s", filename, e)
                            continue

                        if "ERROR" not in content:
                            try:
                                os.remove(file_path)
                                logging.info("Deleted old log: %s", filename)
                            except Exception as e:
                                if hasattr(e, "winerror") and e.winerror == 32:
                                    logging.warning("File %s is in use; skipping deletion.", filename)
                                else:
                                    logging.error("Error deleting file %s: %s", filename, e)
                        else:
                            # Check if errors are only due to the memory issue
                            error_lines = [line for line in content.splitlines() if "ERROR" in line]
                            if error_lines and all("model requires more system memory" in line for line in error_lines):
                                try:
                                    os.remove(file_path)
                                    logging.info("Deleted old log (only memory error present): %s", filename)
                                except Exception as e:
                                    if hasattr(e, "winerror") and e.winerror == 32:
                                        logging.warning("File %s is in use; skipping deletion.", filename)
                                    else:
                                        logging.error("Error deleting file %s: %s", filename, e)

                    # Delete files older than 3 days regardless of content
                    if now - file_date > timedelta(days=3):
                        try:
                            os.remove(file_path)
                            logging.info("Deleted old log (older than 3 days): %s", filename)
                        except Exception as e:
                            if hasattr(e, "winerror") and e.winerror == 32:
                                logging.warning("File %s is in use; skipping deletion.", filename)
                            else:
                                logging.error("Error deleting file %s: %s", filename, e)

        except Exception as e:
            logging.error("Error in log cleanup: %s", e, exc_info=True)

        time.sleep(3600)  # Run every hour

# Start the cleanup thread to delete old logs
cleanup_thread = threading.Thread(target=delete_old_logs, daemon=True)
cleanup_thread.start()

async def get_tools(session:ClientSession):
    tools  = await load_mcp_tools(session)
    print("✅ Discovered tools:", [t.name for t in tools])
    print("Description of tools:", [t.description for t in tools])
    logger.info("✅ Discovered tools: %s", [t.name for t in tools])
    return tools

async def access_resource(session:ClientSession,query:str):
    prompt=f"{query}\nNote: Ignore case and punctuation"
    response = await session.read_resource(f"graph://query/{prompt}")
    print(f"✅ Response: {response}")
    logger.info("✅ Response: %s", response)
    return response

async def main():
    async with streamablehttp_client("http://127.0.0.1:8000/mcp") as (read, write, _):
        async with ClientSession(read,write) as session:
            # Initialize the connection
            await session.initialize()
            tools = await get_tools(session)
            resource_response = await access_resource(session,"what are the the ingredient are required to make butter chicken")


async def intialize_llm(recipe:Recipe,provider:str="google"):
    if provider == "google":
        # Google Gemini
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=5.0,     # maximum 5 requests per second
        )
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
    return llm,llm_with_tools,llm_with_structured

def build_graph():
    # Build the graph with the tools and LLMs
    recipe = Recipe()
    llm,llm_with_tools, llm_with_structured = await intialize_llm(recipe, provider="google")
    

    
    
if __name__ == "__main__":
    asyncio.run(main())