'''
Web crawler/scraper for a specific website, designed to extract text content
and store for AI applications.

Copyright (C) 2025 Andrea Marson

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.
'''

'''
Main entry point for the web crawler/scraper application.
'''

#import streamlit as st
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
import time
from typing import Set, List, Dict
import config
import hashlib
import warnings

#from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.vectorstores import Qdrant

import cheshire_cat_api as ccat
from cheshire_cat_api.models.body_upload_url import BodyUploadUrl

SKIP_VECTOR_STORING = False  # Set to True to skip vector store operations
QUERY_DELAY = 1.0  # Default delay between requests



#embeddings = OllamaEmbeddings(model="llama3.2")
#vector_store = InMemoryVectorStore(embedding=embeddings)  # Fixed parameter name


class WebCrawler:
    def __init__(self, base_url: str, max_depth: int = 3, delay: float = 1.0):
        self.base_url = base_url
        self.max_depth = max_depth
        self.delay = delay
        self.visited: Set[str] = set()
        #self.embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
        # Get the embedding dimension

        #self.vector_store = InMemoryVectorStore(embedding=self.embeddings)  # Fixed parameter name

        if (not SKIP_VECTOR_STORING):
            match config.SCRAPING_MODE:
                case config.SCRAPING_MODE_COLEMAN_LOCAL_AI_PACKAGED:
                    self.embeddings_model = config.EMBEDDINGS_MODEL  # Default model, can be changed later
                    self.embeddings = OllamaEmbeddings(model=self.embeddings_model)  # Initialize embeddings with the model
                    config.logging.info(f"Using model '{self.embeddings_model}' for embeddings")
                    # Initialize Qdrant client
                    self.qdrant_client = QdrantClient("localhost", port=6333)  # Use ":memory:" for in-memory or "localhost" for local server
                    #self.collection_name = "web_crawl"
                    self.collection_name = config.COLLECTION_NAME
                    config.logging.info(f"Using collection '{self.collection_name}' for vector storage")
                    # Detect embedding dimension
                    test_embedding = self.embeddings.embed_query("test")
                    embedding_dimension = len(test_embedding)
                    config.logging.info(f"Detected embedding dimension: {embedding_dimension}")
                    # Check if collection exists and create if it doesn't
                    collections = self.qdrant_client.get_collections()
                    collection_exists = any(collection.name == self.collection_name for collection in collections.collections)

                    if not collection_exists:
                        config.logging.info(f"Creating collection '{self.collection_name}'...")
                        self.qdrant_client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=models.VectorParams(
                                size=embedding_dimension,  # Use detected dimension
                                distance=models.Distance.COSINE
                            )
                        )
                    else:
                        config.logging.info(f"Collection '{self.collection_name}' already exists")        

                    self.vector_store = Qdrant(
                        client=self.qdrant_client,
                        collection_name=self.collection_name,
                        embeddings=self.embeddings
                    )
                case config.SCRAPING_MODE_CHESHIRE_CAT:
                    config.logging.info("Connecting to Cheshire Cat AI instance ...")
                    # For more details, please see https://github.com/cheshire-cat-ai/api-client-py
                    self.ccat_config = ccat.Config(user_id="admin")
                    # Connect to the API
                    self.ccat_client = ccat.CatClient(
                        config=self.ccat_config
                    )
                    

        # Set up headers for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; MyBot/1.0; +http://mybot.com)'
        }

    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to the same domain."""
        return urlparse(url).netloc == urlparse(self.base_url).netloc
    
    def contains_any(self, text: str, patterns: List[str]) -> bool:
        """
        Check if any string pattern from the list is contained in the text.
        
        Args:
            text (str): The text to search in
            patterns (List[str]): List of patterns to search for
        
        Returns:
            bool: True if any pattern is found in text, False otherwise
        """
        return any(pattern in text for pattern in patterns)
    
    def is_unwanted_url(self, url: str) -> bool:
        """Check if URL contains any unwanted patterns."""
        # Use lowercase strings only
        unwanted_patterns = ["edit",
                             "oldid",
                             "action=",
                             "diff=",
                             "oldid=",
                             "jpg",
                             "jpeg",
                             "png",
                             "svg",
                             "gif",
                             "special:",
                             "?",
                             "javascript",
                             "#",
                             "entities",
                             "zip"
                             ]
        return self.contains_any(url.lower(), unwanted_patterns)

    def extract_links(self, html: str, current_url: str) -> Set[str]:
        """Extract all valid links from a page."""
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        for a in soup.find_all('a', href=True):
            url = urljoin(current_url, a['href'])
            if self.is_valid_url(url):
                links.add(url)
        return links

    def extract_text(self, html: str) -> str:
        """Extract clean text content from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        for script in soup(['script', 'style', 'header', 'footer', 'nav']):
            script.decompose()
        return soup.get_text(separator=' ', strip=True)

    def process_page(self, url: str, depth: int) -> List[Dict]:
        """Process a single page and return its content and links."""
        if depth > self.max_depth or url in self.visited:
            return []
        
        try:
            time.sleep(self.delay)  # Rate limiting
            response = requests.get(url, headers=self.headers, timeout=10)
            self.visited.add(url)
            
            if response.status_code == 200:
                html = response.text
                text = self.extract_text(html)
                return [{"content": text, "url": url, "depth": depth}]
            
        except Exception as e:
            config.logging.error(f"Error processing {url}: {str(e)}")
        
        return []

    def crawl(self):
        """Crawl the website iteratively."""
        to_visit = [(self.base_url, 0)]  # (url, depth)
        #documents = []

        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                add_start_index=True
            )

        while to_visit:
            url, depth = to_visit.pop(0)
            
            # Skip unwanted URLs
            if self.is_unwanted_url(url):
                config.logging.info(f"Skipping {url} at depth {depth} due to unwanted patterns.")
                continue
            config.logging.info(f"Visiting {url} at depth {depth}")
            
            # Process the page if it hasn't been visited
            if url not in self.visited:
                page_docs = self.process_page(url, depth)

                # Process the page documents
                if len(page_docs) > 0:
                    config.logging.debug(f"Processing {page_docs[0]}")
                    if (not SKIP_VECTOR_STORING):
                        match config.SCRAPING_MODE:
                            case config.SCRAPING_MODE_COLEMAN_LOCAL_AI_PACKAGED:
                                # Create text chunks
                                chunks = text_splitter.split_text(page_docs[0]["content"])
                                config.logging.debug(f"Created {len(chunks)} chunks from {page_docs[0]['url']}")
                            
                                # Process and store documents
                                for chunk in chunks:
                                    # Create a unique ID for the chunk based on its content
                                    chunk_id = hashlib.md5(chunk.encode()).hexdigest()
                                    
                                    # Check if this chunk already exists
                                    search_result = self.qdrant_client.scroll(
                                        collection_name=self.collection_name,
                                        limit=1,
                                        scroll_filter=models.Filter(
                                            must=[
                                                models.FieldCondition(
                                                    key="metadata.chunk_id",
                                                    match=models.MatchValue(value=chunk_id)
                                                )
                                            ]
                                        )
                                    )

                                    # Only add if the chunk doesn't exist
                                    if not search_result[0]:
                                        config.logging.debug(f"Adding chunk {chunk_id} to vector store ...")
                                        try:
                                            self.vector_store.add_texts(
                                                texts=[chunk],
                                                metadatas=[{
                                                    "url": page_docs[0]["url"],
                                                    "depth": page_docs[0]["depth"],
                                                    "chunk_id": chunk_id
                                                }]
                                            )
                                            config.logging.info(f"Added chunk {chunk_id} from {page_docs[0]['url']}")
                                        except Exception as e:
                                            config.logging.error(f"Error adding chunk to vector store: {str(e)}")
                                    else:
                                        config.logging.debug(f"Chunk {chunk_id} already exists in vector store, skipping ...")
                                    config.logging.info(f"Processed {len(chunks)} chunks from {page_docs[0]['url']}")
                            case config.SCRAPING_MODE_CHESHIRE_CAT:
                                body_upload_url = BodyUploadUrl(
                                    url=url
                                )
                                try:
                                    response = self.ccat_client.rabbit_hole.upload_url(body_upload_url)
                                except Exception as e:
                                    config.logging.error("Error detected: " + e)
                                    # Handle HTTP errors
                                    if isinstance(e, HTTPError):
                                        match e.response.status_code:
                                            case HTTPStatus.BAD_REQUEST:
                                                pass       

                                    else:
                                        config.logging.info(f"Skipping vector store operations for {page_docs[0]['url']} ...")

                #documents.extend(page_docs)
                
                if depth < self.max_depth:
                    try:
                        response = requests.get(url, headers=self.headers)
                        new_links = self.extract_links(response.text, url)
                        new_links = [(link, depth + 1) for link in new_links 
                                   if link not in self.visited]
                        to_visit.extend(new_links)
                    except Exception as e:
                        config.logging.error(f"Error extracting links from {url}: {str(e)}")

        # return documents

    def process_and_store(self):
        """Process crawled content and store in vector database."""
        #documents = self.crawl()
        self.crawl()
    
        #for doc in documents:

if __name__ == "__main__":
    warnings.simplefilter('always')  # or 'always'

    config.logging.info(f"### Starting web crawler version {config.VERSION} (mode = {config.SCRAPING_MODE}) ...")

    # Sanity checks
    if (not config.SCRAPING_MODE in config.SCRAPING_MODES):
        config.logging.error(f"Invalid scraping mode: {config.SCRAPING_MODE}. Available modes: {config.SCRAPING_MODES}")
        exit(1)

    crawler = WebCrawler(
        base_url="https://" + config.BASE_URL,
        max_depth=config.MAX_DEPTH,
        delay=QUERY_DELAY
    )
  
    crawler.process_and_store()