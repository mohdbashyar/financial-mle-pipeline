import os
import logging
import chromadb
from typing import List, Dict, Optional
from google import genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize ChromaDB Persistent Client
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_data")
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    # The default embedding function is all-MiniLM-L6-v2
    collection = chroma_client.get_or_create_collection(name="financial_news")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    collection = None

def store_news_in_chroma(ticker: str, news_items: List[Dict]):
    """Embeds and stores fetched financial news into ChromaDB."""
    if not collection or not news_items:
        return
    
    documents = []
    metadatas = []
    ids = []
    
    for i, item in enumerate(news_items):
        title = item.get("title") or item.get("content", {}).get("title", "")
        if not title:
            continue
            
        # Enhance RAG context with summary
        summary = item.get("summary") or item.get("content", {}).get("summary", "")
        full_text = f"Title: {title}\nSummary: {summary}" if summary else title
            
        publisher = item.get("publisher") or item.get("content", {}).get("provider", {}).get("displayName", "Unknown")
        link = item.get("link") or item.get("content", {}).get("canonicalUrl", {}).get("url", "")
        
        # yfinance news uses uuid or id
        doc_id = item.get("uuid") or item.get("id") or f"{ticker}_news_{i}"
        
        # We store the full_text as the document to embed
        documents.append(full_text)
        metadatas.append({
            "ticker": ticker,
            "publisher": publisher,
            "link": link
        })
        ids.append(doc_id)
        
    if documents:
        try:
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Upserted {len(documents)} news items for {ticker} into ChromaDB.")
        except Exception as e:
            logger.error(f"Error upserting into ChromaDB: {e}")

def query_rag(query: str, ticker: Optional[str] = None) -> Dict:
    """Queries ChromaDB for relevant news and asks Gemini to generate an answer."""
    if not collection:
        return {"answer": "ChromaDB is not initialized.", "sources": []}
        
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"answer": "GEMINI_API_KEY environment variable is not set.", "sources": []}
        
    # Build filter if ticker is provided
    where_filter = {"ticker": ticker.upper()} if ticker else None
    
    # Retrieve top 10 most relevant documents for richer context
    try:
        results = collection.query(
            query_texts=[query],
            n_results=10,
            where=where_filter
        )
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        return {"answer": f"Error querying vector database: {e}", "sources": []}
        
    if not results or not results["documents"] or not results["documents"][0]:
        return {"answer": f"No relevant news context found in the database for query: '{query}'.", "sources": []}
        
    retrieved_docs = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]
    
    # Construct context string
    context_text = "\n".join([f"- {doc} (Source: {meta.get('publisher', 'Unknown')})" 
                              for doc, meta in zip(retrieved_docs, retrieved_metadatas)])
                              
    sources = [meta.get("link", "") for meta in retrieved_metadatas if meta.get("link")]
    
    # Initialize Gemini client
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""You are a professional financial AI assistant. 
Use ONLY the following retrieved news headlines to answer the user's question. 
If the retrieved news does not contain the answer, say "I don't have enough recent news data to answer this."

When answering:
1. Be highly detailed and comprehensive.
2. Explicitly mention the publisher/source for the news you are referencing in your answer text.
3. Format your answer with clear bullet points if summarizing multiple news items or events.

Context News:
{context_text}

User Question: {query}
"""
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        answer = response.text
    except Exception as e:
        logger.error(f"Error calling Gemini LLM: {e}")
        answer = f"Error generating answer with Gemini: {e}"
        
    return {
        "answer": answer,
        "sources": list(set(sources)) # Unique sources
    }
