````markdown
# Chef‑Agent Knowledge‑Graph Cooking Assistant

A streaming AI “Chef” agent that uses LangGraph workflows, MCP tools, and a Neo4j‑backed recipe knowledge graph to answer cooking queries, update or ingest recipes, and remember user preferences.

---

## 🚀 Features

- **Interactive streaming** conversation via FastMCP + FastAPI  
- **Graph‑driven** recipe storage & updates (Neo4j + `langchain_neo4j` + `LLMGraphTransformer`)  
- **Tool support** for:
  - `web_search` (Tavily/DuckDuckGo)
  - `web_scraper` (FireCrawl + BeautifulSoup fallback)
  - `execute_python` sandboxed code
  - `graph_query` (natural‑language → Cypher)
  - `ingest_url_to_graph` (scrape & ingest new recipes)
- **Memory** via in‑process store (or Redis) to personalize sessions  
- **Auto‑summarization** of long chats with a short‑term summarizer  

---

## 📦 Prerequisites

- Python 3.10+  
- Neo4j 4.4+ (standalone or Docker)  
- Redis Stack (if using RedisStore/checkpointer)  

### Environment Variables

Create a `.env` file at project root and set all of the following:

```ini
# Multi‑provider LLM keys
GOOGLE_API_KEY=
GROQ_API_KEY=
CEREBRAS_API_KEY=

# Search & scraping
TAVILY_API_KEY=
E2B_API_KEY=
FIRECRAWL_API_KEY=

# Langfuse observability
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=

# Neo4j connection
NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
NEO4J_DATABASE=

# Redis (optional)
DB_URI=redis://localhost:6379/0
````

---

## 🔧 Installation

1. **Clone repo**

   ```bash
   git clone https://github.com/your-org/chef-agent.git
   cd chef-agent
   ```

2. **Create & activate** a virtual env

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set your** `.env` as above.

---

## ⚙️ Running the MCP Server

```bash
python mcp_server.py
```

* Exposes MCP tools at `http://127.0.0.1:8000/mcp`
* Health check: `GET /health`

---

## ⚙️ Running the Agent

```bash
python agent.py
```

* Connects to MCP server
* Builds a LangGraph `StateGraph` workflow:

  1. **assistant**: generates initial answer, sets `has_final` once `[FinalAnswer]:` appears
  2. **tools**: invokes any needed tools (web\_search, graph\_query, etc.)
  3. **update\_graph** → **graph\_update\_tool\_calling**: decides & applies graph updates
  4. **finalize\_answer**: produces final user‑facing recipe plan
  5. **write\_memory** → **summarization\_node**: saves memory & summarizes
* Streaming output: prints incremental responses

---

## 📂 Code Structure

```
.
├── agent.py            # Main agent orchestration & graph workflow
├── mcp_server.py       # FastAPI + FastMCP tool definitions
├── graphDB.py          # GraphDB wrapper (Neo4j + LLMGraphTransformer)
├── schemas.py          # Pydantic models: Recipe, Profile, UpdateGraphDecision
├── scrapper.py         # Web scraper & Markdown converter
├── prompts/
│   ├── SYSTEM_PROMPT.txt
│   ├── decision_prompt.txt
│   ├── decision_prompt_2.txt
│   ├── conversation_prompt.txt
│   └── summarization_prompt.txt
├── requirements.txt
├── .env
└── README.md
```

---

## 🛠️ Customization

* **Switch LLM**: in `agent.py` change `provider="google"` to `"groq"` or another supported model.
* **Enable Redis** for persistence: swap `InMemoryStore/Saver` with `AsyncRedisStore/Saver` and set `DB_URI`.
* **Extend tools**: add new `@mcp.tool()` functions in `mcp_server.py`.

---

## 🐞 Troubleshooting

* **Graph connectivity**: confirm Neo4j credentials & network reachability.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch.
3. Submit a pull request with your changes.

---

## Contact

For any questions or suggestions, feel free to contact on below Contact details:

- Om Nagvekar Portfolio Website, Email: https://omnagvekar.github.io/ , omnagvekar29@gmail.com
- GitHub Profile:
   - Om Nagvekar: https://github.com/OmNagvekar

---

## 📜 License

This project is licensed under the [GPL-3.0 license](LICENSE).
