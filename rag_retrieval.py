from sentence_transformers import SentenceTransformer
from vector_db_client import get_collection
from config import EMBED_MODEL_NAME

model = SentenceTransformer(EMBED_MODEL_NAME)
collection = get_collection()

def retrieve_support_chunks(query: str, top_k: int = 5):
    """
    Retrieves top-k most relevant support document chunks for a given query.
    Args:
        query (str): User support query
        top_k (int): Number of chunks to retrieve
    Returns:
        List[dict]: Each dict contains chunk_id, content, metadata, and similarity score
    """
    # TODO: Implement (1) query embedding, (2) top-k cosine-similarity retrieval using Chroma, (3) result assembly
    # Do not modify collection or external files – focus only on retrieval logic
    raise NotImplementedError("Complete the retrieval logic here.")
