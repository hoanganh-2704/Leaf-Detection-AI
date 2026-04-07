# Leaf Detection System (Multi-Agent framework)

## Overview
This project develops an AI Agent system for rice leaf disease detection. It utilizes a Multi-Agent architecture incorporating a Vision Transformer (ViT) model, an LLM, and a Vector Database for diagnosis and explanation.

## AI Multi-Agent Architecture
The system consists of specialized agents coordinated to perform diagnosis seamlessly:
1. **Coordinator Agent (`src.agents.coordinator`)**: Orchestrates the workflow among other agents and uses Gemini to synthesize the final robust agricultural report based on their combined inputs.
2. **Preprocessing Agent (`src.agents.preprocessing`)**: Processes raw user images to remove background noise (using `rembg`), leaving only the necessary leaf shape for better model focus.
3. **Classification Agent (`src.agents.classification`)**: Feeds the image into a Hugging Face Vision Transformer (`prithivMLmods/Rice-Leaf-Disease`) to extract a primary disease label and prediction confidence score.
4. **Morphology Agent (`src.agents.morphology`)**: Evaluates the preprocessed image using Google's Gemini Vision to describe the morphological specifics and visual symptoms the input exhibits.
5. **Retrieval Agent (`src.agents.retrieval`)**: Queries the locally-stored ChromaDB vector database (Retrieval-Augmented Generation) to extract scientific knowledge, characteristics, and prevention strategies for the detected disease.

## Directory Structure
- `src/`: Source code for the application.
  - `agents/`: Multi-Agent system components, including the pipeline orchestrator.
  - `api/`: API endpoints via **FastAPI** (`app.py`).
  - `core/`: Core utilities, environment setup (`config.py`), and RAG vector DB generation scripts (`knowledge_setup.py`).
  - `models/`: Machine learning models setup and inferences.
  - `ui/`: User Interface handled by **Streamlit** (`streamlit_app.py`).
- `data/`: Datasets for training, validation, and testing.
  - `raw/`: Raw image data.
  - `processed/`: Processed data and ChromaDB vector embeddings.
- `notebooks/`: Jupyter notebooks for explorations and data preparation.
- `docs/`: Documents, reports, and diagrams.
- `references/`: Reference papers and academic materials.

---

## 🚀 Quick Start / How to Run

There are two primary ways to run this project depending on your environment needs.

### Method 1: Local 1-Click Start (Recommended for Development)
If you are running on macOS or Linux and want the easiest possible local setup:
1. Copy the environment variables template:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and add your `GEMINI_API_KEY`.
3. Run the automated starter script:
   ```bash
   ./start.sh
   ```
*This script will automatically create a virtual environment, install dependencies, initialize the Vector DB, start the FastAPI server in the background, and launch the Streamlit frontend locally!*

---

### Method 2: Docker / Docker Compose (Recommended for Production)
The exact entire infrastructure is fully containerized. You do not need Python or PyTorch installed locally.
1. Add your `GEMINI_API_KEY` to the `.env` file.
2. Build and launch the cluster using Docker:
   ```bash
   docker-compose up --build -d
   ```
*Docker will handle setting up the appropriate System-level C bindings, fetching isolated Python images, caching the ML arrays, and launching both the UI (Port 8506) and the API Base (Port 8000).*

---

### Method 3: Manual Deployment (Step-by-Step)
If you prefer to manually configure the environment and run the services individually without Docker or the starter script:

#### 1. Setup Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/MacOS
source venv/bin/activate
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Environment Variables
Copy the template to establish your `.env` file and insert your `GEMINI_API_KEY`:
```bash
cp .env.example .env
```

#### 4. Initialize the Vector Knowledge Base (ChromaDB)
```bash
python -m src.core.knowledge_setup
```

#### 5. Run the API (Backend)
In your terminal, boot up the FastAPI framework:
```bash
uvicorn src.api.app:app --reload
```
The API serves at `http://127.0.0.1:8000`.

#### 6. Run the UI (Frontend Application)
In a **new terminal tab** (with the virtual environment freshly activated), host the interface:
```bash
streamlit run src/ui/streamlit_app.py
```
This automatically opens the web console interface on `http://localhost:8501`.
