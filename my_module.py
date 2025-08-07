import os
from pinecone import Pinecone

def search_pinecone_memory_offline(namespace: str, index_name="hackrx-index-2"):
    """
    Fetches all question-answer pairs from a Pinecone namespace without needing an online embedding model.
    It works by fetching all vectors and checking their metadata.
    """
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    
    qa_cache = {}

    try:
        # Fetch all vector IDs from the namespace. This is more efficient than a broad query.
        # Note: list() paginates, so for very large namespaces (>1000s of Qs), you might need to handle pagination.
        id_list_response = index.list(namespace=namespace, limit=1000) # Adjust limit as needed
        vector_ids = [r['id'] for r in id_list_response.vectors]

        if not vector_ids:
            return {}

        # Fetch the actual vectors and their metadata by ID
        fetch_response = index.fetch(ids=vector_ids, namespace=namespace)
        
        for vec_id, vector_data in fetch_response.vectors.items():
            question = vector_data.metadata.get("question")
            answer = vector_data.metadata.get("answer")
            if question and answer:
                qa_cache[question] = answer
                
    except Exception as e:
        # If the namespace doesn't exist or there's an error, return an empty cache
        print(f"Could not fetch from Pinecone namespace '{namespace}': {e}")
        return {}
        
    return qa_cache