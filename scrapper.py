import httpx
from bs4 import BeautifulSoup
from typing import List, Optional, Dict
import tempfile
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_core.documents import Document
import os
import logging

logger = logging.getLogger(__name__)
temp_dir = os.path.join(os.getcwd(), "temp_html")  # Create a folder in your current dir
os.makedirs(temp_dir, exist_ok=True)


class WebScraper:
    """
    A simple async web scraping tool using httpx and BeautifulSoup.
    Use this to fetch HTML content and extract data via CSS selectors.

    Example:
        scraper = WebScraper()
        html = await scraper.fetch_html("https://example.com")
        titles = scraper.get_text_by_selector(html, "h1.title")
    """

    def __init__(self, timeout: float = 10.0, headers: Optional[Dict[str, str]] = None):
        self.timeout = timeout
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (compatible; ChefAgent/1.0; +https://yourdomain.com)"
        }

    async def fetch_html(self, url: str) -> str:
        """
        Fetches HTML content of the given URL asynchronously.

        Args:
            url: The target webpage URL.
        Returns:
            The raw HTML content as a string.
        Raises:
            httpx.HTTPError on network issues.
        """
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    def parse_soup(self, html: str) -> BeautifulSoup:
        """
        Parses HTML with BeautifulSoup.

        Args:
            html: Raw HTML string.
        Returns:
            BeautifulSoup object for querying.
        """
        return BeautifulSoup(html, "lxml")

    def get_text_by_selector(self, html: str, selector: str) -> List[str]:
        """
        Extracts text from elements matching a CSS selector.

        Args:
            html: Raw HTML string.
            selector: CSS selector string (e.g., "div.content > p").
        Returns:
            List of text contents for matched elements.
        """
        soup = self.parse_soup(html)
        elements = soup.select(selector)
        return [el.get_text(strip=True) for el in elements]

    def get_links(self, html: str, selector: Optional[str] = None) -> List[str]:
        """
        Extracts href attributes from <a> tags. Optionally filter by CSS selector.

        Args:
            html: Raw HTML string.
            selector: Optional CSS selector to narrow down <a> tags.
        Returns:
            List of URLs as strings.
        """
        soup = self.parse_soup(html)
        if selector:
            anchors = soup.select(selector)
        else:
            anchors = soup.find_all("a", href=True)
        return [a["href"] for a in anchors]

    def get_attributes_by_selector(self, html: str, selector: str, attr: str) -> List[str]:
        """
        Extracts specified attribute values from elements matching selector.

        Args:
            html: Raw HTML string.
            selector: CSS selector for elements.
            attr: The attribute name (e.g., 'src', 'data-id').
        Returns:
            List of attribute values.
        """
        soup = self.parse_soup(html)
        elements = soup.select(selector)
        return [el.get(attr) for el in elements if el.has_attr(attr)]

    async def scrape_data(self, url: str, mapping: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Fetches page and extracts multiple fields based on a mapping of keys to selectors.

        Args:
            url: Page URL.
            mapping: Dict where keys are field names and values are CSS selectors.
        Returns:
            Dict mapping field names to lists of extracted text.

        Example:
            mapping = {
                'title': 'h1.title',
                'ingredients': 'ul.ingredients li',
                'steps': 'ol.steps li'
            }
        """
        html = await self.fetch_html(url)
        result: Dict[str, List[str]] = {}
        for field, sel in mapping.items():
            result[field] = self.get_text_by_selector(html, sel)
        return result

    async def load_html_to_md(self,url: str)->List[Document]:
        """ Loads HTML content from a URL and converts it to Markdown format.
        This method fetches the HTML content, saves it to a temporary file,

        Args:
            url (str): The URL of the webpage to scrape.

        Returns:
            List[Document]: A list of Document objects containing the Markdown content.
        """
        try:
            html = await self.fetch_html(url)
            # Create a temporary file to store the HTML content
            with tempfile.NamedTemporaryFile(dir=temp_dir,suffix=".html", delete=False) as tmp_file:
                tmp_file.write(html.encode("utf-8"))
                tmp_file.flush()
                tmp_file.close()
                
                loader = DoclingLoader(file_path=tmp_file.name, export_type=ExportType.MARKDOWN)
                docs =  await loader.aload()
                os.remove(tmp_file.name) # Clean up the temporary file
        except Exception as e:
            logger.error(f"Error converting HTML to Markdown: {e}")
            print(f"❌ Error converting HTML to Markdown: {e}")
            return []
        return docs
        