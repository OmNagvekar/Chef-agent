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
# from langgraph.checkpoint.redis.aio import AsyncRedisSaver
# from langgraph.store.redis.aio import AsyncRedisStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables.config import RunnableConfig
import logging
import os
import time
from typing import Literal
import threading
import sys
from datetime import datetime, timedelta
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from IPython.display import Image, display
from langfuse.langchain import CallbackHandler
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
import json
import re
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
def load_promt(template:Literal["SYSTEM_PROMPT","decision_prompt","conversation_prompt","summarization_prompt"]=None):
    if template == "SYSTEM_PROMPT":
        with open("./prompts/SYSTEM_PROMPT.txt",'r', encoding='utf-8') as f:
            SYSTEM_PROMPT_msg = f.read()
            print("✅ System prompt loaded successfully.")
        return SYSTEM_PROMPT_msg

    elif template == "decision_prompt":
        with open("./prompts/decision_prompt.txt",'r', encoding='utf-8') as f:
            decision_prompt_msg = f.read()
            decision_prompt_msg = SystemMessage(content=decision_prompt_msg)
            print("✅ decision prompt loaded successfully.")
        return decision_prompt_msg
    
    elif template == "conversation_prompt":

        with open("./prompts/conversation_prompt.txt",'r', encoding='utf-8') as f:
            conversation_prompt_msg = f.read()
            conversation_prompt_msg = SystemMessage(content=conversation_prompt_msg)
            # This is the conversation prompt that will be used in the workflow
            print("✅ Conversation prompt loaded successfully.")
        return conversation_prompt_msg
    
    elif template == "summarization_prompt":
        with open("./prompts/summarization_prompt.txt",'r', encoding='utf-8') as f:
            summarization_prompt_msg = f.read()
            summarization_prompt_msg = SystemMessage(content=summarization_prompt_msg)
            # This is the conversation prompt that will be used in the workflow
            print("✅ summarization prompt loaded successfully.")
        return summarization_prompt_msg
    else:
        print("Invalid template name. Please choose from 'SYSTEM_PROMPT', 'decision_prompt', 'conversation_prompt', or 'summarization_prompt'.")

class State(MessagesState):
    RecipeIntstruction: str
    AudioMessage:str
    has_final:bool
    summary:str

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
    async with streamablehttp_client("http://127.0.0.1:8000/mcp") as (read, write, _):
        async with ClientSession(read,write) as session:
            # Initialize the connection
            await session.initialize()
            tools = await get_tools(session)
            graph_png_path = "./graph.png"
            graph,memStore = build_graph(tools)
            langfuse_handler = CallbackHandler()
            # display(Image(graph.get_graph(xray=True).draw_mermaid_png(output_file_path=graph_png_path)))
            config = {"configurable": {"thread_id": "1", "user_id": "1"},"callbacks": [langfuse_handler]}
            input_message = HumanMessage(content="how to make idli ?")
            async for chunk in graph.astream({"messages": input_message}, config, stream_mode="values"):
                chunk["messages"][-1].pretty_print()
            namespace =("memory","1")
            existing_memory = memStore.get(namespace, "user_memory")
            print(f"\n\n## memory {existing_memory}\n\n")
            print("#####"*30,"\n\n")
            thread = {"configurable": {"thread_id": "1"}}
            temp = graph.get_state(thread)
            stetes = temp.values.get("messages")
            for m in stetes:
                m.pretty_print()
            temp = graph.get_state(thread)
            stetes = temp.values.get("summary")
            print(stetes)
            

def intialize_llm(provider:str="google"):
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

def build_graph(tools):
    # Build the graph with the tools and LLMs
    llm = intialize_llm(provider="groq")
    llm_with_tools = llm.bind_tools(tools)
    decision_llm = create_extractor(
        llm,
        tools=[UpdateGraphDecision],
        tool_choice="UpdateGraphDecision",
        enable_inserts=True,
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
        SYSTEM_PROMPT = load_promt("SYSTEM_PROMPT")
        SYSTEM_PROMPT = SystemMessage(content=SYSTEM_PROMPT.format(memory=memory))
        messages = [SYSTEM_PROMPT] + state["messages"]
        
        # Use the LLM to generate a response
        response = llm_with_tools.invoke(messages)
        marker = "[FinalAnswer]:"
        answer = response.content
        idx = answer.find(marker)
        pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
        def strip_think(text: str) -> str:
            return pattern.sub("", text)
        if idx != -1:
            # Extract text after the marker and strip whitespace
            parsed = answer[idx + len(marker):].strip()
            return {"messages":[response],"RecipeIntstruction":strip_think(parsed),"has_final":True}
        else:
            # Fallback: no marker found, return full content stripped
            parsed = answer.strip()
        return {"messages":[response],"has_final":False}
    
    def update_graph(state:State):
        decision_prompt = load_promt("decision_prompt")
        messages = [decision_prompt] + state["messages"]
        response = decision_llm.invoke({"messages":messages})
        result = response["responses"][0].model_dump()
        logger.info("Decision result: %s", result)
        followup_template = """You have a graph‑update decision to act on. Use the fields as follows:
        1. should_update (bool):
            - true  → proceed with an update.
            - false → skip any graph changes.

        2. tool_choice (one of):
            - graph_query        → call graph_query().
            - ingest_url_to_graph → call ingest_url_to_graph().
            - both               → first graph_query(), then ingest_url_to_graph().
            - none               → do nothing.

        3. Available Tools:
            • graph_query(query_text: str): translate NL into Cypher to update existing nodes/relationships.  
            • ingest_url_to_graph(url: str): scrape and ingest a recipe URL into the graph, creating new nodes/relationships.

        Below is the decision you must act on:
        ```json
        {decision}
        ```"""

        if result["should_update"]:
            print("✅"*4,"\n\n")
            system_message = SystemMessage(content=followup_template.format(decision=json.dumps(result, indent=4)))
            tools_response = llm_with_tools.invoke(
                [system_message]
            )
            return {"messages":[tools_response]}
        else:
            return {"messages": state["messages"] + [HumanMessage(content="No graph update needed.")]}
    
    
    def finalize_answer(state: State):
        conversation_prompt = load_promt("conversation_prompt")
        messages = [conversation_prompt]+[state["RecipeIntstruction"]]
        response = llm.invoke(messages)
        logger.info("Final answer generated: %s", response.content)
        pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

        def strip_think(text: str) -> str:
            return pattern.sub("", text)
        
        return {"messages": state["messages"] + [response], "has_final": False,"AudioMessage":strip_think(response.content)}
    
    def write_memory(state: State, store: BaseStore, config: RunnableConfig):
        """Writes the current state to the memory store."""
        # Here you can implement logic to save the state to a redis database
        # Get the user ID from the config
        user_id = config["configurable"]["user_id"]

        # Retrieve existing memory from the store
        namespace = ("memory", user_id)
        existing_memory = store.get(namespace, "user_memory")
        # Get the profile as the value from the list, and convert it to a JSON doc
        existing_profile = {"UserProfile": existing_memory.value} if existing_memory else "None"
        CREATE_MEMORY_INSTRUCTION = """Create or update a user profile memory based on the user's chat history.
        This will be saved for long-term memory. If there is an existing memory, simply update it.
        Here is the existing memory (it may be empty): {memory}"""
        
        result = profile_llm.invoke([SystemMessage(content=CREATE_MEMORY_INSTRUCTION.format(memory=existing_profile))]+state["messages"])
        
        # Get the updated profile as a JSON object
        updated_profile = result["responses"][0].model_dump()
        
        # Save the updated profile
        key = "user_memory"
        store.put(namespace, key, updated_profile)
    
    summarization_node = SummarizationNode(
        token_counter=count_tokens_approximately,
        model=llm,
        max_tokens=512,
        max_tokens_before_summary=512,
        max_summary_tokens=128,
        output_messages_key="messages"
    )

    workflow = StateGraph(State)
    
    workflow.add_node("assistant", assistant)
    workflow.add_node("tools",ToolNode(tools))
    workflow.add_node("update_graph", update_graph)
    workflow.add_node("finalize_answer", finalize_answer)
    workflow.add_node("write_memory", write_memory)
    workflow.add_node("summarization_node",summarization_node)
    
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
    workflow.add_edge("write_memory","summarization_node")
    workflow.add_edge("summarization_node", END)
    
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    graph = workflow.compile(store=store, checkpointer=checkpointer)
    return graph,store
    
    
if __name__ == "__main__":
    asyncio.run(main())