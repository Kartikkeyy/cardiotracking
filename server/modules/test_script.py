"""
Test script to verify the enhanced query response
This will help you see what data is being retrieved from Pinecone
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

load_dotenv()

def test_retrieval(question: str):
    
    # Setup
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "medicalindex1"))
    embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    print(f"\n{'='*80}")
    print(f"QUERY: {question}")
    print(f"{'='*80}\n")
    
    # Embed query
    print("1. Embedding query...")
    embedded_query = embed_model.embed_query(question)
    print(f"   ✓ Query embedded (dimension: {len(embedded_query)})\n")
    
    # Query Pinecone
    print("2. Querying Pinecone...")
    res = index.query(vector=embedded_query, top_k=3, include_metadata=True)
    print(f"   ✓ Retrieved {len(res['matches'])} matches\n")
    
    # Display results
    print("3. Retrieved Documents:\n")
    
    for i, match in enumerate(res['matches'], 1):
        print(f"   {'─'*76}")
        print(f"   DOCUMENT #{i}")
        print(f"   {'─'*76}")
        print(f"   ID:           {match['id']}")
        print(f"   Score:        {match['score']:.4f}")
        print(f"   Source:       {match['metadata'].get('source', 'N/A')}")
        print(f"   Page:         {match['metadata'].get('page', 'N/A')}")
        print(f"   Has Text:     {'text' in match['metadata']}")
        
        if 'text' in match['metadata']:
            text = match['metadata']['text']
            preview = text[:300] + "..." if len(text) > 300 else text
            print(f"\n   TEXT PREVIEW:")
            print(f"   {preview}\n")
        else:
            print(f"   ⚠️  WARNING: No 'text' field in metadata!")
            print(f"   Available metadata keys: {list(match['metadata'].keys())}\n")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total documents retrieved: {len(res['matches'])}")
    docs_with_text = sum(1 for m in res['matches'] if 'text' in m['metadata'])
    print(f"Documents with text:       {docs_with_text}")
    print(f"Documents without text:    {len(res['matches']) - docs_with_text}")
    
    if docs_with_text == 0:
        print("\n⚠️  PROBLEM DETECTED:")
        print("   No documents have 'text' in metadata!")
        print("   → You need to re-upload documents with the fixed load_vectorstore.py")
    elif docs_with_text < len(res['matches']):
        print("\n⚠️  PARTIAL PROBLEM:")
        print("   Some documents missing 'text' in metadata")
        print("   → Consider re-uploading all documents")
    else:
        print("\n✅ All documents have text content - retrieval should work!")
    
    print(f"{'='*80}\n")
    
    return res

if __name__ == "__main__":
    # Test with a sample medical query
    test_questions = [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?",
        "What are the side effects of aspirin?"
    ]
    
    print("\n" + "="*80)
    print("PINECONE RETRIEVAL TEST")
    print("="*80)
    
    # You can test with one question or multiple
    for question in test_questions[:1]:  # Change [:1] to test all questions
        test_retrieval(question)
        
    print("\nTest complete! Check the output above to verify:")
    print("1. Documents are being retrieved")
    print("2. Each document has a 'text' field in metadata")
    print("3. The text content is relevant to your query")