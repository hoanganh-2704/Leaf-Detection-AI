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

## 🚀 How to Run the Project (Step-by-Step)

### 1. Requirements and Setup
Ensure you have Python 3.9 or higher. 
First, it is highly recommended to set up a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/MacOS
source venv/bin/activate
```

Install the required project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
You need a Google API Key to use Gemini and create Generative Embeddings.
- Copy `.env.example` to `.env` (or create a `.env` file):
  ```bash
  cp .env.example .env
  ```
- Open `.env` and add your Google Gemini API key:
  ```ini
  GEMINI_API_KEY=your_actual_gemini_api_key_here
  ```

### 3. Initialize the Vector Knowledge Base (ChromaDB)
To set up the Retrieval Agent correctly, you must generate the necessary ChromaDB instances comprising the disease reference documents. Run:
```bash
python -m src.core.knowledge_setup
```
This builds and saves the `chroma_db` into `data/processed`.

### 4. Run the API (Backend)
The system leverages FastAPI to expose the Multi-Agent diagnoses endpoint (`/diagnose`).
To start the backend, run:
```bash
uvicorn src.api.app:app --reload
```
Once this launches, the API is running locally (usually on `http://127.0.0.1:8000`). You can view auto-generated documentation via `http://127.0.0.1:8000/docs`.

### 5. Run the UI (Frontend Application)
In a **new terminal tab** (with the virtual environment activated), start the user experience layer powered by Streamlit:
```bash
streamlit run src/ui/streamlit_app.py
```
This will open the web console interface on `http://localhost:8501`. Upload any image of a rice leaf here, and follow along as the Multi-Agent system works through each process layer to achieve a diagnosis!
