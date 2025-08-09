# Customer Support RAG: Retrieval & Chunking Optimization Task

## Task Overview
You are improving a semantic search feature for a customer support assistant. The support database has 8,000 documents in Chroma using Sentence-Transformers embeddings, but the retrieval results are often irrelevant due to ineffective chunking and missing metadata. Your goal: update the chunking approach, attach key metadata, configure the collection for cosine similarity, and implement a reliable top-5 support document retrieval logic.

## Guidance
- The current pipeline splits documents into large, context-diluting 2,000-token chunks and omits overlap/metadata; this impacts recall and precision.
- Focus on implementing a chunking approach using 200-token window and 50-token overlap. Attach metadata such as `category`, `priority`, and `date` to each chunk.
- You only need to complete the marked retrieval logic and chunking sections in the Python scripts. Infrastructure, embedding, and database setup are fully provided.
- Retrieval must use cosine similarity on the Chroma collection, returning the top-5 relevant support chunks for each query.
- Evaluate your fix by running sample queries and reviewing improvements in recall@k versus previous baseline results.

## Database Access
- **Vector DB Host:** `<DROPLET_IP>`
- **Port:** 8000 (Chroma default API)
- **Collection Name:** `support_docs_chunks`
- **Vector Dimension:** 384 (all-MiniLM-L6-v2)
- **Chunk Metadata:** `chunk_id`, `doc_id`, `chunk_index`, `category`, `priority`, `date`, `content`, `embedding`, `token_count`, `start_position`
- Use any Python Chroma client for exploration; embeddings and chunks will be available after running the init script.

## Objectives
- Implement correct chunking (200 tokens, 50 overlap), metadata extraction/storage, and embedding logic in the provided scripts.
- Complete the marked retrieval function to fetch top-5 relevant support chunks for user queries based on cosine similarity.
- Improve search quality for common support tasks and meet recall@5 targets (see sample queries).

## How to Verify
- Run test queries from `sample_queries.txt` through your completed retrieval function.
- Spot-check that returned results match the correct support topics, as indicated in chunk `category`/`priority`.
- Calculate recall@5 by seeing if the relevant support documents appear in the returned results across the sample queries. Mark improvements over baseline.

---
