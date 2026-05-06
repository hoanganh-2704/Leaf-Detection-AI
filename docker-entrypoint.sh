#!/bin/bash
# docker-entrypoint.sh
# Runs once when the container starts.
# Initialises the ChromaDB knowledge base (needs GEMINI_API_KEY at runtime),
# then hands off to whatever CMD was passed (backend or frontend).

set -e

# Initialise the vector KB only if it has not been built yet
# (the data/ volume is shared, so this runs once across restarts)
if [ ! -d "/app/data/processed/chroma_db" ]; then
    echo "🧠 First-run: initialising ChromaDB knowledge base..."
    python -m src.core.knowledge_setup
    echo "✅ Knowledge base ready."
else
    echo "✅ ChromaDB knowledge base already present – skipping init."
fi

# Execute the CMD supplied by docker-compose (uvicorn or streamlit)
exec "$@"
