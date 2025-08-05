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
Helper application for testing/debugging purposes.
'''

import streamlit as st
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import logging
import config


# Initialize Qdrant client and models
@st.cache_resource
def init_qdrant():
    embeddings = OllamaEmbeddings(model=config.EMBEDDINGS_MODEL)
    qdrant_client = QdrantClient("localhost", port=6333)
    vector_store = Qdrant(
        client=qdrant_client,
        collection_name=config.COLLECTION_NAME,
        embeddings=embeddings
    )
    return vector_store

@st.cache_resource
def init_llm():
    return OllamaLLM(model=config.CHAT_MODEL)

# Create the RAG prompt template
prompt_template = """You are a helpful AI assistant that answers questions based on the provided context.
Context: {context}

Question: {question}

Please provide a clear and concise answer based on the context provided. If you cannot find the answer in the context, say so.

Answer:"""

def get_response(query: str, vector_store, llm) -> str:
    # Search for relevant documents
    docs = vector_store.similarity_search(query, k=3)
    
    # Prepare context from documents
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Format prompt with context and query
    formatted_prompt = prompt.format(context=context, question=query)
    
    # Get response from LLM
    response = llm.invoke(formatted_prompt)
    
    return response

def main():
    config.logging.info("Starting Knowledge Base Chat application ...")
    st.title("Knowledge Base Chat")
    st.write("Ask questions about the content from the scraped website!")
    
    # Initialize vector store and LLM
    try:
        vector_store = init_qdrant()
        llm = init_llm()
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        return
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            # Get AI response
            with st.chat_message("assistant"):
                response = get_response(prompt, vector_store, llm)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()