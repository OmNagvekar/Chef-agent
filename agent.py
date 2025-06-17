import asyncio
import httpx
import platform
import socket
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.resources import get_mcp_resource
from mcp import ClientSession
from mcp.client.streamable_http  import streamablehttp_client
from schemas import Profile,UpdateGraphDecision
from trustcall import create_extractor
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph, MessagesState,END
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.store.base import BaseStore
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.store.redis.aio import AsyncRedisStore
from langchain_core.runnables.config import RunnableConfig
import logging
import os
import time
import threading
import sys
from datetime import datetime, timedelta
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from IPython.display import Image, display
    

load_dotenv()  # Load environment variables from .env file

DB_URI = os.getenv("DB_URI")

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

# Load System prompt from a text file

with open("./prompts/SYSTEM_PROMPT.txt",'r', encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read()
    print("✅ System prompt loaded successfully.")

with open("./prompts/decision_prompt.txt",'r', encoding='utf-8') as f:
    decision_prompt = f.read()
    decision_prompt = SystemMessage(content=decision_prompt)
    print("✅ decision prompt loaded successfully.")

with open("./prompts/conversation_prompt.txt",'r', encoding='utf-8') as f:
    conversation_prompt = f.read()
    conversation_prompt = SystemMessage(content=conversation_prompt)
    # This is the conversation prompt that will be used in the workflow
    print("✅ Conversation prompt loaded successfully.")

with open("./prompts/summarization_prompt.txt",'r', encoding='utf-8') as f:
    summarization_prompt = f.read()
    summarization_prompt = SystemMessage(content=summarization_prompt)
    # This is the conversation prompt that will be used in the workflow
    print("✅ summarization prompt loaded successfully.")

class State(MessagesState):
    RecipeIntstruction: str
    AudioMessage:str
    has_final:bool

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

# async def access_resource(session:ClientSession,query:str):
#     prompt=f"{query}\nNote: Ignore case and punctuation"
#     response = await session.read_resource(f"graph://query/{prompt}")
#     print(f"✅ Response: {response}")
#     logger.info("✅ Response: %s", response)
#     resource_response = await access_resource(session,"what are the the ingredient are required to make butter chicken")
#     return response

async def main():
    async with (
        AsyncRedisStore.from_conn_string(DB_URI) as store,
        AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        async with streamablehttp_client("http://127.0.0.1:8000/mcp") as (read, write, _):
            async with ClientSession(read,write) as session:
                # Initialize the connection
                await session.initialize()
                tools = await get_tools(session)
                graph_png_path = "./graph.png"
                graph = await build_graph(tools,checkpointer,store)
                display(Image(graph.get_graph(xray=True).draw_mermaid_png(output_file_path=graph_png_path)))
            

async def intialize_llm(provider:str="google"):
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

    return llm

async def build_graph(tools,checkpointer,store: BaseStore):
    # Build the graph with the tools and LLMs
    llm = await intialize_llm(provider="google")
    llm_with_tools = llm.bind_tools(tools)
    decision_llm = create_extractor(
        llm,
        tools=[UpdateGraphDecision],
        tool_choice="UpdateGraphDecision",
    )
    profile_llm = create_extractor(
        llm,
        tools=[Profile],
        tool_choice="Profile",
    )
    def assistant(state:State,config: RunnableConfig,store: BaseStore):
        """The assistant node that generates a response based on the current state."""
        # Get the user ID from the config
        user_id = config["configurable"]["user_id"]

        # Retrieve memory from the store
        namespace = ("memory", user_id)
        existing_memory = store.get(namespace, "user_memory")

        # Format the memories for the system prompt
        if existing_memory and existing_memory.value:
            memory = existing_memory.value
        else:
            memory=""
        SYSTEM_PROMPT = SystemMessage(content=SYSTEM_PROMPT.format(memory=memory))
        messages = [SYSTEM_PROMPT] + state["messages"]
        
        # Use the LLM to generate a response
        response = llm_with_tools.invoke(messages)
        marker = "[FinalAnswer]:"
        answer = response.content
        idx = answer.find(marker)
        if idx != -1:
            # Extract text after the marker and strip whitespace
            parsed = answer[idx + len(marker):].strip()
            return {"messages":[response],"RecipeIntstruction":parsed,"has_final":True}
        else:
            # Fallback: no marker found, return full content stripped
            parsed = answer.strip()
        return {"messages":[response],"has_final":False}
    
    def update_graph(state:State):
        messages = [decision_prompt] + state["messages"]
        response = decision_llm.invoke({"messages":messages})
        result = response["responses"][0].model_dump()
        logger.info("Decision result: %s", result)
        followup_template = """You have a graph‑update decision to act on. Use the fields as follows:
        1. should_update (bool):
            - true  → proceed with an update.
            - false → skip any graph changes.

        2. tool_choice (one of):
            - graph_query        → call graph_query(reason).
            - ingest_url_to_graph → call ingest_url_to_graph(reason).
            - both               → first graph_query(reason), then ingest_url_to_graph(reason).
            - none               → do nothing.

        3. Available Tools:
            • graph_query(query_text: str): translate NL into Cypher to update existing nodes/relationships.  
            • ingest_url_to_graph(url: str): scrape and ingest a recipe URL into the graph, creating new nodes/relationships.

        Below is the decision you must act on:
        ```json
        {decision}
        ```"""

        if result["should_update"]:
            system_message = SystemMessage(content=followup_template.format(decision=result))
            tools_response = llm_with_tools.invoke(
                {"messages": [system_message]+ state["messages"]}
            )
            return {"messages":[tools_response]}
        else:
            return {"messages": state["messages"] + [HumanMessage(content="No graph update needed.")]}
    
    
    def finalize_answer(state: State):
        messages = [conversation_prompt]+state["RecipeIntstruction"]
        response = llm.invoke(messages)
        logger.info("Final answer generated: %s", response.content)
        
        return {"messages": state["messages"] + [response], "has_final": False,"AudioMessage":response.content}
    
    def write_memory(state: State, store: BaseStore, config: RunnableConfig):
        """Writes the current state to the memory store."""
        # Here you can implement logic to save the state to a redis database
        # Get the user ID from the config
        user_id = config["configurable"]["user_id"]

        # Retrieve existing memory from the store
        namespace = ("memory", user_id)
        existing_memory = store.get(namespace, "user_memory")
        # Get the profile as the value from the list, and convert it to a JSON doc
        existing_profile = {"UserProfile": existing_memory.value} if existing_memory else None
        CREATE_MEMORY_INSTRUCTION = """Create or update a user profile memory based on the user's chat history.
        This will be saved for long-term memory. If there is an existing memory, simply update it.
        Here is the existing memory (it may be empty): {memory}"""
        
        result = trustcall_extractor.invoke({"messages": [SystemMessage(content=CREATE_MEMORY_INSTRUCTION.format(memory=existing_profile))]+state["messages"]})
        
        # Get the updated profile as a JSON object
        updated_profile = result["responses"][0].model_dump()
        
        # Save the updated profile
        key = "user_memory"
        store.put(namespace, key, updated_profile)
        
        summary = llm.invoke({"messages": [summarization_prompt]+ state["messages"]})
        
        # Delete all but the 2 most recent messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        
        return {
            "messages": state["messages"] + [summary] + delete_messages,
        }


    workflow = StateGraph(State)
    
    workflow.add_node("assistant", assistant)
    workflow.add_node("tools",ToolNode(tools))
    workflow.add_node("update_graph", update_graph)
    workflow.add_node("finalize_answer", finalize_answer)
    workflow.add_node("write_memory", write_memory)
    
    workflow.add_edge(START, "assistant")
    workflow.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    workflow.add_conditional_edges(
        "assistant",
        lambda state: "update_graph" if state["has_final"] else "tools",
        {"update_graph": "update_graph", "tools": "tools"}
    )
    workflow.add_edge("tools", "assistant")
    workflow.add_edge("update_graph", "finalize_answer")
    workflow.add_edge("finalize_answer","write_memory")
    workflow.add_edge("write_memory", END)
    
    graph = workflow.compile(store=store, checkpointer=checkpointer)
    return graph

    
    
if __name__ == "__main__":
    asyncio.run(main())