from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import GraphDocument
from scrapper import WebScraper
from typing import List, Any, Literal, Dict
from langchain_core.documents import Document
import logging
from tqdm import tqdm
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_neo4j import GraphCypherQAChain


logger = logging.getLogger(__name__)
load_dotenv()

class GraphDB:
    def __init__(self, llm, refresh_schema: bool = False):
        """ Initializes the GraphDB with a Neo4jGraph instance.

        Args:
            llm (LangChain): The language model to be used for the graph operations.
            refresh_schema (bool, optional): Defaults to True.
        """

        self.graph = Neo4jGraph(refresh_schema=refresh_schema)
        self.llm = llm
        self.transformer = llm_transformer_filtered = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=["RecipeName", "Veg_nonveg", "Ingredient", "NutritionFact", "Cuisine", "Category", "InstructionSteps"],
            allowed_relationships=[
                "HAS_INGREDIENT",
                "HAS_NUTRITION",
                "BELONGS_TO_CUISINE",
                "BELONGS_TO_CATEGORY",
                "HAS_STEP",
                "BELONGS_TO_VEG_NONVEG"
            ],
            node_properties=["step_number", "text"],
            relationship_properties=False,
            strict_mode=False
        )
        
    async def pypdf_loader(self):
        """sumary_line
        Simple and Fast document loader
        """
        def load_pdf_files(dirpath:str="./pdf_files")-> list:
            """sumary_line
            Return: return list of path of pdf's in specific directory
            """
            logger.info("Loading PDF files from directory...")
            pdf_files = []
            for filename in tqdm(os.listdir(dirpath)):
                if filename.endswith(".pdf"):
                    filepath = os.path.join(dirpath, filename)
                    pdf_files.append(os.path.abspath(filepath))
            pdf_files = [p for p in pdf_files if os.path.isfile(p)]  # Validate paths
            logger.info(f"Found {len(pdf_files)} PDF files.")
            return pdf_files
        logger.info("Loading PDFs using PyPDFLoader...")
        document=[]

        file_path = load_pdf_files()
        for path in tqdm(file_path):
            try:
                loader = PyPDFLoader(file_path=path)
                combined_content =[]
                metadata=None
                pages = await loader.aload()
                for page in pages:
                    if metadata is None:
                        metadata=page.metadata
                        del metadata['page'],metadata['page_label']
                        metadata.update({'source':os.path.basename(metadata['source'])})
                    combined_content.append(page.page_content)

                combined_content = "\n\n".join(combined_content)
                document.append(Document(page_content=combined_content,metadata=metadata))
                logging.info(f"Processed file: {path}")
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")

        logger.info(f"Total documents loaded: {len(document)}")      
        return document
    
    async def document_loader(self,url: str, web_scraper: bool=True) -> List[Document]:
        """
        Loads documents from a URL, either by scraping the web or using a PDF loader.

        Args:
            url (str): The URL to load documents from.
            web_scraper (bool, optional): If True, uses a web scraper to load HTML content and convert it to Markdown. Defaults to True.

        Returns:
            List[Document]: A list of Document objects containing the loaded content.
        """
        if web_scraper:
            web_scraper = WebScraper()
            markdown_docs = await web_scraper.load_html_to_md(url)
            logger.info(f"Loaded {len(markdown_docs)} Markdown documents from {url}.")
            if not markdown_docs:
                logger.warning(f"No documents found at {url}.")
                return []
            return markdown_docs
        else:
            documents = await self.pypdf_loader()
            return documents

    async def transform_documents(self, documents: List[Document]) -> List[GraphDocument]:
        graph_documents_filtered = await self.transformer.aconvert_to_graph_documents(
            documents
        )
        logger.info(f"Transformed {len(documents)} documents into graph documents.")
        return graph_documents_filtered

    async def add_documents_to_graph(self, documents: List[GraphDocument]) -> None:
        """Adds a list of GraphDocument objects to the Neo4j graph.

        Args:
            documents (List[GraphDocument]): A list of GraphDocument objects to be added to the graph.
        """
        if not documents:
            logger.warning("No documents to add to the graph.")
            return
        
        try:
            self.graph.add_graph_documents(documents)
            logger.info(f"Added {len(documents)} documents to the graph.")
        except RuntimeError as e:
            logger.error(f"Explicitly Neo4j driver connection is closeed: {e}")
            raise
    
    async def run(self,url:str) -> bool:
        """ Runs the GraphDB by loading documents from a URL and transforming them into graph documents,
        and then adding the graph documents to the Neo4j graph.
        """
        try:
            documents = await self.document_loader(url=url)
            if not documents:
                logger.warning("No documents loaded. Exiting.")
                raise ValueError("No documents loaded from the provided URL.")
                return False
            logger.info(f"Loaded {len(documents)} documents from {url}.")
            graph_documents = await self.transform_documents(documents)
            await self.add_documents_to_graph(graph_documents)
            logger.info(f"Successfully added {len(graph_documents)} graph documents to the Neo4j graph.")
            logger.info("GraphDB run completed successfully.")
            return True
        except Exception as e:
            logger.error(f"Error in GraphDB run: {e}")
            return False
    
    async def Cypher_query(self, query: str) -> List[Dict[str,Any]]:
        """Queries the Neo4j graph with a Cypher query.

        Args:
            query (str): The Cypher query to execute.

        Returns:
            List[Dict[str,Any]]: A list of dictionaries containing the results of the query.
        """
        try:
            results = self.graph.query(query)
            logger.info(f"Executed query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    async def query(self,query:str)-> Dict[str,Any]:
        """ Queries the Neo4j graph using a GraphCypherQAChain.
        This method allows you to ask questions about the graph data using natural language queries.

        Args:
            query (str): The natural language query to ask about the graph data.

        Returns:
            Dict[str,Any]: A dictionary containing the response from the graph, which may include the answer to the query and any relevant context or metadata.
        """
        try:
            chain = GraphCypherQAChain.from_llm(
                graph=self.graph, llm=self.llm, verbose=True,allow_dangerous_requests=True
            )
            response = await chain.ainvoke(query)
            print(response)
            logger.info(f"Query response: {response}")
        except Exception as e:
            logger.error(f"Error in query execution: {e}")
            response = {"error": str(e)}
        return response
        
if __name__ == "__main__":
    import asyncio
    from langchain_groq import ChatGroq
    from langchain_core.rate_limiters import InMemoryRateLimiter

    # Initialize the LLM
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=5.0,  # maximum 5 requests per second
    )
    llm= ChatGroq(model="qwen-qwq-32b", temperature=0)
    # Create an instance of GraphDB
    graph_db = GraphDB(llm=llm, refresh_schema=True)
    
    
    asyncio.run(graph_db.run("https://www.allrecipes.com/recipe/141169/easy-indian-butter-chicken/"))