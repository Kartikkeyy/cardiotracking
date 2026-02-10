from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from pydantic import Field
from typing import List, Optional
from logger import logger
import os

router=APIRouter()

@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        # Embed model + Pinecone setup
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        # Embed the question
        embedded_query = embed_model.embed_query(question)
        logger.debug(f"Query embedded successfully")
        
        # Query Pinecone - get top 5 most relevant chunks
        res = index.query(vector=embedded_query, top_k=5, include_metadata=True)
        logger.debug(f"Pinecone returned {len(res.get('matches', []))} matches")

        # Create documents from results WITH similarity scores
        docs = []
        retrieval_details = []
        
        for match in res["matches"]:
            # Create document
            doc = Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            )
            docs.append(doc)
            
            # Store retrieval details for response
            retrieval_details.append({
                "id": match["id"],
                "score": match["score"],
                "text_preview": match["metadata"].get("text", "")[:200] + "...",
                "source": match["metadata"].get("source", "Unknown"),
                "page": match["metadata"].get("page", "N/A")
            })
        
        # Log if no text was retrieved
        if all(not doc.page_content for doc in docs):
            logger.warning("No text content found in retrieved documents. Check if 'text' field exists in Pinecone metadata.")
            return JSONResponse(
                status_code=404,
                content={
                    "error": "No text content found in database",
                    "suggestion": "Please re-upload your documents with the fixed load_vectorstore.py"
                }
            )

        class SimpleRetriever(BaseRetriever):
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)

            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs

        retriever = SimpleRetriever(docs)
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)
        
        # Add retrieval details to the response
        result["retrieval_details"] = retrieval_details

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})