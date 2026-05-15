@echo off
setlocal EnableDelayedExpansion

echo ==========================================
echo   Rice Disease Detection AI - Start
echo ==========================================

:: ── 1. Check .env ──────────────────────────
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo [WARNING] Created .env from .env.example.
        echo [ACTION]  Open .env and set your GEMINI_API_KEY, then re-run this script.
        pause
        exit /b 1
    ) else (
        echo [ERROR] No .env or .env.example found. Cannot continue.
        pause
        exit /b 1
    )
)

:: ── 2. Create virtual environment ──────────
if not exist "venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to create venv. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)

:: ── 3. Activate and install dependencies ───
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Installing dependencies (may take a minute on first run)...
pip install -r requirements.txt -q
if !ERRORLEVEL! NEQ 0 (
    echo [ERROR] pip install failed. Check requirements.txt and your internet connection.
    pause
    exit /b 1
)

:: ── 4. Initialise ChromaDB ─────────────────
echo [INFO] Initialising Knowledge Base (ChromaDB)...
python -m src.core.knowledge_setup
if !ERRORLEVEL! NEQ 0 (
    echo [ERROR] Knowledge base setup failed. Try deleting data\processed\chroma_db and rerun.
    pause
    exit /b 1
)

:: ── 5. Start FastAPI backend in background ─
echo [INFO] Starting FastAPI backend on port 8000...
start "RiceAI Backend" /MIN cmd /c "call venv\Scripts\activate.bat && uvicorn src.api.app:app --host 0.0.0.0 --port 8000"

:: Give the backend a moment to start
timeout /t 4 /nobreak >nul

echo.
echo   App ready at:  http://localhost:8506
echo   API docs at:   http://localhost:8000/docs
echo.
echo   Close this window or press Ctrl+C to stop the frontend.
echo   The backend runs in a separate minimised window - close that too to stop it.
echo.

:: ── 6. Start Streamlit frontend ────────────
streamlit run src/ui/streamlit_app.py --server.port=8506

endlocal
