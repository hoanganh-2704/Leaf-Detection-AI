# 🌿 Rice Disease Detection AI

A multi-agent AI system for diagnosing rice leaf diseases using **Vision Transformer (ViT)**, **Google Gemini**, and **ChromaDB RAG**.

---

## 🤖 AI Multi-Agent Architecture

Five specialised agents collaborate to produce a complete diagnosis:

| Agent | Role |
|---|---|
| **Coordinator** | Orchestrates the pipeline and synthesises the final report |
| **Preprocessing** | Removes background (via `rembg`) and normalises the image |
| **Classification** | Runs ViT (`prithivMLmods/Rice-Leaf-Disease`) → disease label + confidence |
| **Morphology** | Describes visual symptoms using **Gemini Vision** multimodal analysis |
| **Retrieval** | Queries **ChromaDB** (RAG) for evidence-based treatment information |

Supported diseases: **Blast (Đạo ôn)** · **Bacterial Blight (Bạc lá)** · **Brown Spot (Đốm nâu)** · **Tungro (Vàng lùn)**

---

## 📋 Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | Required for local / venv methods. [Download](https://www.python.org/downloads/) |
| **Docker Desktop** | Required for the Docker method. [Download](https://www.docker.com/products/docker-desktop/) |
| **GEMINI_API_KEY** | Get one free at [Google AI Studio](https://aistudio.google.com/app/apikey) |

> **Windows users:** Make sure Python is added to `PATH` during installation (check the box on the Python installer). After installing Docker Desktop, ensure it is running before using any `docker compose` command.

---

## ⚙️ Environment Setup (all methods)

### Linux / macOS
```bash
cp .env.example .env
```

### Windows (Command Prompt)
```cmd
copy .env.example .env
```

### Windows (PowerShell)
```powershell
Copy-Item .env.example .env
```

Then open `.env` in any text editor and set:
```
GEMINI_API_KEY=your_actual_key_here
```

---

## 🚀 Method 1 — 1-Click Script (Recommended for Development)

### Linux / macOS

```bash
chmod +x start.sh
./start.sh
```

### Windows (Double-click or Command Prompt)

```cmd
start.bat
```

Or run from Command Prompt:
```cmd
cd "path\to\Leaf Detection"
start.bat
```

The script automatically:
- Creates a Python virtual environment
- Installs all dependencies from `requirements.txt`
- Initialises the ChromaDB vector knowledge base (~20 s, needs internet + API key)
- Starts **FastAPI backend** on port **8000**
- Starts **Streamlit frontend** on port **8506**

**Access the app:**

| Service | URL |
|---|---|
| 🌿 Streamlit UI | http://localhost:8506 |
| 📡 FastAPI Docs | http://localhost:8000/docs |

Press `Ctrl+C` (or close the windows on Windows) to stop all services.

---

## 🐳 Method 2 — Docker Compose (Recommended for Production)

> Works identically on **Linux, macOS, and Windows** (Docker Desktop required on Windows/macOS).

### Step 1 — Build and start

```bash
docker compose up --build
```

Or in detached (background) mode:

```bash
docker compose up --build -d
```

### Step 2 — Wait for startup

The **backend** initialises ChromaDB and loads the ViT model on the first run (~60 s).  
The **frontend** waits for the backend healthcheck to pass before starting.

Follow logs in real time:

```bash
docker compose logs -f
```

### Step 3 — Access the app

| Service | URL |
|---|---|
| 🌿 Streamlit UI | http://localhost:8506 |
| 📡 FastAPI Docs | http://localhost:8000/docs |

### Useful commands

```bash
# Stop containers (data volume is preserved)
docker compose down

# Stop and delete the ChromaDB data volume (forces re-init next time)
docker compose down -v

# Rebuild after code changes
docker compose up --build

# Check container status
docker compose ps

# View logs for one service
docker compose logs -f backend
docker compose logs -f frontend
```

> **First-run note:** On the very first `docker compose up`, the backend runs `knowledge_setup.py` which calls the Gemini API to build the ChromaDB index. This takes ~30–60 s. The `./data/processed/chroma_db/` directory is persisted as a bind-mount volume so subsequent restarts skip this step and start in seconds.

> **Windows note:** Run all `docker compose` commands in **PowerShell** or **Command Prompt**. Docker Desktop must be running. If you see a `WSL 2` error, enable it in Docker Desktop settings.

---

## 🔧 Method 3 — Manual Step-by-Step

### 1. Create and activate a virtual environment

**Linux / macOS**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt)**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> **Windows PowerShell tip:** If you see `cannot be loaded because running scripts is disabled`, run this once as Administrator:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

**Linux / macOS**
```bash
cp .env.example .env
```

**Windows (Command Prompt)**
```cmd
copy .env.example .env
```

Open `.env` and add your `GEMINI_API_KEY`.

### 4. Initialise the ChromaDB knowledge base

```bash
python -m src.core.knowledge_setup
```

> Requires `GEMINI_API_KEY` in `.env`. Takes ~20 seconds on first run.

### 5. Start the FastAPI backend

Open a terminal (with the venv activated):

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

API available at `http://localhost:8000` · Docs at `http://localhost:8000/docs`

### 6. Start the Streamlit frontend

Open a **second terminal** (activate the venv again):

**Linux / macOS**
```bash
source venv/bin/activate
streamlit run src/ui/streamlit_app.py --server.port 8506
```

**Windows**
```cmd
venv\Scripts\activate.bat
streamlit run src/ui/streamlit_app.py --server.port 8506
```

Open `http://localhost:8506` in your browser.

---

## 🗂️ Project Structure

```
Leaf Detection/
├── src/
│   ├── agents/               # Multi-agent pipeline
│   │   ├── coordinator.py
│   │   ├── preprocessing.py
│   │   ├── classification.py
│   │   ├── morphology.py
│   │   └── retrieval.py
│   ├── api/                  # FastAPI REST backend
│   │   └── app.py
│   ├── core/                 # Shared utilities
│   │   ├── config.py
│   │   ├── workflow.py
│   │   └── knowledge_setup.py
│   └── ui/                   # Streamlit frontend
│       └── streamlit_app.py
├── data/
│   └── processed/
│       └── chroma_db/        # Vector DB (auto-created on first run)
├── docs/                     # Thesis, diagrams, screenshots
├── tests/                    # pytest test suite
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh      # Container startup script
├── requirements.txt
├── start.sh                  # 1-click launcher (Linux/macOS)
├── start.bat                 # 1-click launcher (Windows)
└── .env.example
```

---

## 🧪 Running Tests

**Linux / macOS**
```bash
source venv/bin/activate
pytest tests/ -v
```

**Windows**
```cmd
venv\Scripts\activate.bat
pytest tests/ -v
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key — get one free at [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `HF_TOKEN` | ⬜ Optional | HuggingFace token (only if the ViT model requires authentication) |

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `python` not found on Windows | Use `python3` or ensure Python is added to `PATH` during installation |
| PowerShell script execution blocked | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` as Administrator |
| `GEMINI_API_KEY` error | Ensure the key is set in `.env` (not just exported in your shell) |
| Docker: `port is already allocated` | Another service is using port 8000 or 8506. Stop it, or change the port mapping in `docker-compose.yml` |
| Docker: frontend starts before backend is ready | This is handled by the healthcheck. If it persists, increase `start_period` in `docker-compose.yml` |
| ChromaDB not found error | Run `python -m src.core.knowledge_setup` to (re)create the vector DB |
| `rembg` install fails on Windows | Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first |
