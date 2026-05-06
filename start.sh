#!/bin/bash

# Stop script on errors
set -e

echo "======================================"
echo "    🌱 Rice Disease Detection AI - 1-Click Start         "
echo "======================================"

# 1. Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created a default .env file from .env.example."
        echo "❌ ACTION REQUIRED: Please open the .env file and add your GEMINI_API_KEY."
        exit 1
    else
        echo "❌ ERROR: No .env and no .env.example found. Unable to proceed."
        exit 1
    fi
fi

# 2. Setup isolated Virtual Environment
if [ ! -d "venv" ]; then
    echo "📦 Creating fresh Python virtual environment..."
    python3 -m venv venv
fi

# Activate environment
source venv/bin/activate

# 3. Install requirements silently to avoid wall of text
echo "📥 Installing dependencies (this might take a minute on the first run)..."
pip install -r requirements.txt -q

# 4. Initialize ChromaDB Vector Database
echo "🧠 Initializing Knowledge Base (Vector DB)..."
python -m src.core.knowledge_setup

# 5. Start FastAPI Backend
echo "🚀 Starting AI Backend API (FastAPI) on port 8000..."
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
BACKEND_PID=$!

# Wait briefly to ensure backend has started before launching the frontend
sleep 3

# Function to cleanly kill the backend when you exit the script
cleanup() {
    echo ""
    echo "🛑 Stopping FastAPI Backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null || true
    echo "✅ Rice Disease Detection AI shut down. Goodbye!"
    exit 0
}
# Catch Ctrl+C and exit gracefully
trap cleanup SIGINT SIGTERM

# 6. Start Streamlit Frontend
echo "🎨 Starting Streamlit User Interface on port 8506..."
echo ""
echo "  ✅ App ready at: http://localhost:8506"
echo "  📡 API docs at:  http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop all services."
echo ""
streamlit run src/ui/streamlit_app.py --server.port=8506
