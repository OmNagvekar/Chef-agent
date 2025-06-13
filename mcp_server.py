from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
# from fastapi_mcp import FastApiMCP
import uvicorn
import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults
from e2b_code_interpreter import Sandbox 
from firecrawl import AsyncFirecrawlApp
from typing import Dict,List,Any,Optional
from scrapper import WebScraper
from fastapi.encoders import jsonable_encoder
import os
import time
import sys
from datetime import datetime, timedelta
import threading
import asyncio
from dotenv import load_dotenv

load_dotenv()

TODAY_DATE = datetime.now().strftime("%Y-%m-%d")
LOG_DIR='Logs'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"MCP_logs_{TODAY_DATE}.log")
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
def delete_old_logs():
    """Delete log files older than 1 day if they contain no errors (or only a specific memory error),
    and delete files older than 3 days regardless. Files in use are skipped.
    """
    while True:
        try:
            now = datetime.now()
            for filename in os.listdir(LOG_DIR):
                if filename.startswith("MCP_logs_") and filename.endswith(".log"):
                    file_path = os.path.join(LOG_DIR, filename)

                    # Extract date from filename (assumes format "logs_YYYY-MM-DD.log")
                    file_date_str = filename[5:15]
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

cleanup_thread = threading.Thread(target=delete_old_logs, daemon=True)
cleanup_thread.start()


app = FastAPI()


mcp = FastMCP("ChefTools MCP API Server")


# Mount MCP Server on FastAPI Server
app.mount("/", mcp.sse_app())

# fastmcp = FastApiMCP(
#     app,
#     name="ChefTools MCP API Server",
#     description="A modular control plane API server for managing and integrating ChefTools components, services, and workflows.",
# )



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@mcp.tool()
async def web_search(query: str) -> Dict[str,str]:
    """Search Tavily for a query and return maximum 3 results.
    as fallback, uses DuckDuckGo if Tavily fails.
    
    Args:
        query: The search query."""
    try:
        search_docs = await TavilySearchResults(max_results=3).ainvoke({"query":query})
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.get("url","")}"/>\n{doc.get("content","")}\n</Document>'
                for doc in search_docs
            ])
        print("✅ Tavily search successful.")
    except Exception as e:
        print("❌ Tavily search unsuccessful.")
        logger.error(f"Error during web search: {e}")
        print("Using DuckDuckGo as fallback.")
        # Fallback to DuckDuckGo search if Tavily fails
        search_docs = await DuckDuckGoSearchResults(max_results=4,output_format='list').ainvoke({"query":query})
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.get("link","")}" title="{doc.get("title", "")}"/>\n{doc.get("snippet","")}\n</Document>'
                for doc in search_docs
            ]
        )
        print("✅ DuckDuckGo search successful.")
    return {"web_results": formatted_search_docs}

@mcp.tool()
async def execute_python(code: str) -> Any:
    """Execute Python code and return the result.
    Note: Strip backticks in code blocks.
    
    Args:
        code: The Python code to execute."""
    try:
        with Sandbox() as sandbox:
            execution = sandbox.run_code(code)
            result = jsonable_encoder(execution)
            return result
    except Exception as e:
        logger.error("There was an error executing the Python code: %s", e)
        print("❌ There was an error executing the Python code: ",str(e))
        return {"error": str(e)}
    
@mcp.tool()
async def web_scraper(url: str) -> Dict[str, str]:
    """
    Scrape the content of a webpage, preferring FireCrawl but falling
    back to a simple HTTP+BeautifulSoup scraper when FireCrawl runs out
    of quota or otherwise fails.
    
    Args:
        url: The URL of the webpage to scrape.
    Returns:
        A dict with either 'content' or 'error'.
    """
    # 1) Try the FireCrawlLoader first
    try:
        app = AsyncFirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
        result = await app.scrape_url(
            url=url,
            formats=["markdown"],               # desired output formats
            only_main_content=True              # clean page body
        )
        # Determine how to extract docs:
        if hasattr(result, "__iter__") and not hasattr(result, "metadata"):
            # It's iterable (tuple/list) of docs
            docs = list(result)
        else:
            # Single doc object
            docs = [result]
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{url}" metadata="{getattr(doc, "metadata", {})}"/>\n{getattr(doc, "markdown", "")}\n</Document>'
                for doc in docs
            ]
        )
        print("✅ FireCrawl scraping successful.")
        logger.info("FireCrawl scraping successful.")
        return {"content": formatted_search_docs}
    
    # 2) Catch any other FireCrawl errors that should trigger fallback
    except Exception as e:
        logger.error(f"FireCrawl failed ({e}), trying WebScraper fallback.")
        print(f"❌ FireCrawl failed ({e}), trying WebScraper fallback.")

    # 3) Fallback: use your lightweight WebScraper
    try:
        scraper = WebScraper()
        # If you want the whole page content:
        html = await scraper.load_html_to_md(url)
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document URL="{url}"source="{doc.metadata.get("source","")}" metadata="{doc.metadata}"/>\n{doc.page_content}\n</Document>'
                for doc in html
            ]
        )
        print("✅ lightweight WebScraper scraping successful.")
        logger.info("lightweight WebScraper scraping successful.")
        return {"content": formatted_search_docs}

        # — or, if you want structured fields, use scrape_data:
        # mapping = {
        #     "title": "head > title",
        #     "paragraphs": "article p"
        # }
        # data = await scraper.scrape_data(url, mapping)
        # return {"content": data}

    except Exception as e:
        logger.error(f"❌ WebScraper fallback failed: {e}")
        print(f"❌ WebScraper fallback failed: {e}")
        return {"error": "Both FireCrawl and fallback scraping failed."}


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    # fastmcp.mount()
    # mcp.run(transport='sse')
    uvicorn.run(app, host="127.0.0.1", port=8000)