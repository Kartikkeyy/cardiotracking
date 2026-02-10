from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_llm_chain(retriever):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are **MediBot**, an AI assistant helping users understand medical documents.

Answer ONLY from the provided context.

Context:
{context}

User Question:
{question}

Answer:
- If answer not found in context, say:
"I'm sorry, but I couldn't find relevant information in the provided documents."
- Do NOT make up facts.
- Do NOT give medical advice.
"""
    )

    # Format docs into text
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # This step retrieves documents first
    retrieve_docs = RunnableLambda(
        lambda question: {
            "question": question,
            "docs": retriever.invoke(question)
        }
    )

    # Build full pipeline
    rag_chain = (
        retrieve_docs
        | {
            "answer": (
                {
                    "context": lambda x: format_docs(x["docs"]),
                    "question": lambda x: x["question"],
                }
                | prompt
                | llm
                | StrOutputParser()
            ),
            "source_documents": lambda x: x["docs"]
        }
    )

    return rag_chain
