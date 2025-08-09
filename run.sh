#!/bin/bash
set -e

echo "[1/3] Starting Chroma vector database..."
docker-compose up -d
sleep 4

echo "[2/3] Initializing collection with processed support document chunks..."
python3 init_vector_db.py

echo "[3/3] Validating embedding and chunk readiness..."
python3 -c "from vector_db_client import validate_db_ready; validate_db_ready()"

echo "Environment and index setup complete. Ready for retrieval logic testing."
