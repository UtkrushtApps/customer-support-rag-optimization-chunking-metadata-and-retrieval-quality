import os
import re
from sentence_transformers import SentenceTransformer
from vector_db_client import get_collection
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL_NAME, CHROMA_COLLECTION
import nltk
nltk.download('punkt', quiet=True)

RAW_DOC_PATH = 'data/full_doc.txt'
MODEL = SentenceTransformer(EMBED_MODEL_NAME)

def extract_metadata(chunk_text):
    cat_match = re.search(r'\[CATEGORY:\s*(.*?)\]', chunk_text)
    prio_match = re.search(r'\[PRIORITY:\s*(.*?)\]', chunk_text)
    date_match = re.search(r'\[DATE:\s*(.*?)\]', chunk_text)
    return {
        "category": cat_match.group(1).strip() if cat_match else None,
        "priority": prio_match.group(1).strip() if prio_match else None,
        "date": date_match.group(1).strip() if date_match else None
    }

def tokenize_text(text):
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)

def chunk_document(text, size, overlap):
    tokens = tokenize_text(text)
    n_tokens = len(tokens)
    chunks = []
    idx = 0
    for start in range(0, n_tokens, size - overlap):
        end = min(start + size, n_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = ' '.join(chunk_tokens)
        meta = extract_metadata(chunk_text)
        chunks.append({
            'chunk_id': f'supportdoc_{idx}',
            'doc_id': 'support_corpus',
            'chunk_index': idx,
            'category': meta['category'],
            'priority': meta['priority'],
            'date': meta['date'],
            'content': chunk_text,
            'token_count': len(chunk_tokens),
            'start_position': start
        })
        idx += 1
    return chunks

def main():
    print("[INIT] Loading customer support corpus...")
    with open(RAW_DOC_PATH, 'r', encoding='utf-8') as f:
        data = f.read()
    print("[INIT] Chunking data...")
    chunks = chunk_document(data, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"[INIT] {len(chunks)} chunks produced.")

    print("[INIT] Embedding chunks...")
    texts = [c['content'] for c in chunks]
    embeddings = MODEL.encode(texts, batch_size=16, show_progress_bar=True)
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i].tolist()

    collection = get_collection()
    print("[INIT] Inserting chunks to Chroma...")
    collection.add(
        ids=[c['chunk_id'] for c in chunks],
        embeddings=[c['embedding'] for c in chunks],
        documents=[c['content'] for c in chunks],
        metadatas=[{
            k: v for k, v in c.items() if k not in ["embedding", "content"]
        } for c in chunks]
    )
    print(f"[INIT] Successfully inserted {len(chunks)} support chunks.")

if __name__ == '__main__':
    try:
        main()
        print("[COMPLETE] Chroma vector database is initialized and ready.")
    except Exception as e:
        print(f"[ERROR] {e}")
        exit(1)
