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
Global configuration for the web crawler/scraper application.
'''

import logging

VERSION = "0.3.1"

BASE_URL = "wiki.dave.eu"
CHAT_MODEL = "llama3.2:3b"
EMBEDDINGS_MODEL = "llama3.2:3b"
#COLLECTION_NAME = "web_crawl_4096"
#COLLECTION_NAME = f"{BASE_URL}_{EMBEDDINGS_MODEL}_3072"
COLLECTION_NAME = "www_dot_dave_eu_llama3_dot_2_3b_3072"
MAX_DEPTH = 10  # Default maximum depth for crawling

SCRAPING_MODE_COLEMAN_LOCAL_AI_PACKAGED = "coleman_local_ai_packaged"
SCRAPING_MODE_CHESHIRE_CAT = "cheshire_cat"
SCRAPING_MODES = [SCRAPING_MODE_COLEMAN_LOCAL_AI_PACKAGED, SCRAPING_MODE_CHESHIRE_CAT]
SCRAPING_MODE = SCRAPING_MODE_CHESHIRE_CAT

MAX_EMBEDDING_ATTEMPTS = 3  # Maximum attempts to embed a document


# Configure logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='[%Y/%m/%d %I:%M:%S %p]')
logging.root.setLevel(logging.NOTSET)
