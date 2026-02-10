from logger import logger

def query_chain(chain, user_input: str):
    try:
        logger.debug(f"Running chain for input: {user_input}")

        result = chain.invoke(user_input)

        # Extract detailed information from source documents
        source_docs = []
        
        print("\n" + "="*80)
        print("SOURCE DOCUMENTS RETRIEVED")
        print("="*80 + "\n")
        
        for idx, doc in enumerate(result["source_documents"], 1):
            source_info = {
                "content": doc.page_content,  # The actual text chunk
                "metadata": doc.metadata,      # All metadata (page, source file, etc.)
                "source_file": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
            }
            source_docs.append(source_info)
            
            # Log each source document to terminal
            print(f"{'─'*80}")
            print(f"SOURCE DOCUMENT #{idx}")
            print(f"{'─'*80}")
            print(f"Source File: {source_info['source_file']}")
            print(f"Page:        {source_info['page']}")
            print(f"\nCONTENT:")
            print(f"{source_info['content']}")
            print(f"\n{'─'*80}\n")

        response = {
            "response": result["answer"],
            "source_documents": source_docs,
            "num_sources": len(source_docs)
        }

        print("="*80)
        print(f"TOTAL SOURCE DOCUMENTS: {len(source_docs)}")
        print("="*80 + "\n")
        
        logger.debug(f"Chain response with {len(source_docs)} source documents")
        return response

    except Exception as e:
        logger.exception("Error on query chain")
        raise