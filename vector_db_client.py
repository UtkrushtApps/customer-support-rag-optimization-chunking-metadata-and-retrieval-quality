import chromadb
from config import CHROMA_COLLECTION, VECTOR_DB_HOST, VECTOR_DB_PORT

_client = None
_collection = None

def get_chroma_client():
    global _client
    if _client is None:
        _client = chromadb.Client(host=VECTOR_DB_HOST, port=VECTOR_DB_PORT)
    return _client

def get_collection():
    global _collection
    client = get_chroma_client()
    if _collection is None:
        _collection = client.get_or_create_collection(CHROMA_COLLECTION)
    return _collection

def validate_db_ready():
    coll = get_collection()
    n = coll.count()
    assert n > 0, "Support chunks not loaded. Initialization required."
    print(f"[DB VALIDATION] {n} support chunks indexed in collection '{CHROMA_COLLECTION}'.")
