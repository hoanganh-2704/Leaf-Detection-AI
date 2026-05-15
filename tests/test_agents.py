"""
tests/test_agents.py — Unit and integration tests for the multi-agent pipeline.
Usage: pytest tests/ -v
"""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

# ── Helper fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture
def sample_image():
    """Creates a simple 224x224 green-ish test image resembling a leaf."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :, 1] = 120   # Strong green channel
    arr[:, :, 0] = 40    # Some red
    arr[:, :, 2] = 30    # Some blue
    return Image.fromarray(arr, mode="RGB")

# ── Phase 1: Config ──────────────────────────────────────────────────────────────
def test_config_loads():
    from src.core.config import settings
    assert settings.VIT_MODEL_NAME == "prithivMLmods/Rice-Leaf-Disease"
    assert settings.VECTOR_DB_PATH is not None

# ── Phase 2: Preprocessing Agent ────────────────────────────────────────────────
def test_preprocessing_output_size(sample_image):
    """Preprocessed image must always be 224x224 RGB."""
    from src.agents.preprocessing import PreprocessingAgent
    agent = PreprocessingAgent()
    result = agent.process(sample_image)
    
    assert result.size == (224, 224), f"Expected (224, 224) but got {result.size}"
    assert result.mode == "RGB"

def test_preprocessing_output_is_image(sample_image):
    from src.agents.preprocessing import PreprocessingAgent
    agent = PreprocessingAgent()
    result = agent.process(sample_image)
    assert isinstance(result, Image.Image)

# ── Phase 2: Classification Agent ───────────────────────────────────────────────
def test_classification_returns_correct_keys(sample_image):
    """Classification result must contain disease_label, confidence, all_probabilities."""
    from src.agents.classification import ClassificationAgent
    
    # Mock the huggingface model loading and inference
    with patch("src.agents.classification.AutoImageProcessor.from_pretrained") as mock_proc, \
         patch("src.agents.classification.SiglipForImageClassification.from_pretrained") as mock_model:
        
        mock_model_inst = MagicMock()
        mock_model_inst.config.id2label = {
            0: "Bacterialblight",
            1: "Blast",
            2: "Brownspot",
            3: "Healthy",
            4: "Tungro",
        }
        
        import torch
        # Simulate logits
        logits = torch.tensor([[2.0, 0.5, 0.1, 0.1, 0.0]])
        mock_outputs = MagicMock()
        mock_outputs.logits = logits
        mock_model_inst.return_value = mock_outputs
        mock_model.return_value = mock_model_inst
        mock_proc.return_value = MagicMock(return_value={"pixel_values": torch.zeros(1, 3, 224, 224)})
        
        agent = ClassificationAgent()
        result = agent.classify(sample_image)
    
    assert result["disease_key"] == "Bacterialblight"
    assert result["disease_label"] == "Bacterial Blight"
    assert "disease_label" in result
    assert "confidence" in result
    assert "all_probabilities" in result
    assert "raw_probabilities" in result
    assert isinstance(result["confidence"], float)

# ── Phase 2: Retrieval Agent ─────────────────────────────────────────────────────
def test_retrieval_returns_string_when_no_db():
    """If Vector DB path does not exist, retrieval should return a warning string."""
    with patch("src.agents.retrieval.os.path.exists", return_value=False):
        from src.agents.retrieval import RetrievalAgent
        agent = RetrievalAgent()
        result = agent.retrieve_info("Blast")
    assert isinstance(result, str)
    assert len(result) > 0

def test_retrieval_handles_healthy_without_db():
    """Healthy classification should not require ChromaDB or embeddings."""
    with patch("src.agents.retrieval.os.path.exists", return_value=False), \
         patch("src.agents.retrieval.LocalHashEmbeddings") as mock_embeddings:
        from src.agents.retrieval import RetrievalAgent
        agent = RetrievalAgent()
        result = agent.retrieve_info("Healthy")
    assert "Không phát hiện bệnh" in result
    mock_embeddings.assert_not_called()

# ── Phase 3: End-to-End Workflow ─────────────────────────────────────────────────
def test_full_pipeline_returns_expected_keys(sample_image):
    """Integration test: the pipeline dict must contain all expected keys."""
    with patch("src.agents.classification.AutoImageProcessor.from_pretrained") as p, \
         patch("src.agents.classification.SiglipForImageClassification.from_pretrained") as m, \
         patch("src.agents.morphology.ChatGoogleGenerativeAI") as mock_morph_llm, \
         patch("src.agents.retrieval.os.path.exists", return_value=False), \
         patch("src.agents.coordinator.ChatGoogleGenerativeAI") as mock_coord_llm:
        
        import torch
        mock_model_inst = MagicMock()
        mock_model_inst.config.id2label = {0: "Blast"}
        logits = torch.tensor([[2.0]])
        mock_outputs = MagicMock()
        mock_outputs.logits = logits
        mock_model_inst.return_value = mock_outputs
        m.return_value = mock_model_inst
        p.return_value = MagicMock(return_value={"pixel_values": torch.zeros(1, 3, 224, 224)})
        
        mock_morph_llm.return_value.invoke.return_value = MagicMock(content="Vết bệnh màu nâu hình thoi.")
        mock_coord_llm.return_value.invoke.return_value = MagicMock(content="## KẾT LUẬN: Đạo ôn\n...")
        
        from src.core.workflow import run_diagnosis
        result = run_diagnosis(sample_image)
    
    for key in ["preprocessed_image", "classification", "visual_verification", "diagnosis", "morphology", "knowledge", "final_report"]:
        assert key in result, f"Missing key: {key}"

def test_coordinator_visual_override():
    """A high-confidence Gemini disagreement should replace the raw model label."""
    from src.agents.coordinator import CoordinatorAgent

    coordinator = CoordinatorAgent.__new__(CoordinatorAgent)
    classification = {
        "disease_key": "Bacterialblight",
        "disease_label": "Bacterial Blight",
        "confidence": 99.0,
    }
    visual = {
        "suggested_key": "Blast",
        "suggested_label": "Rice Blast",
        "confidence": 82.0,
    }

    diagnosis = coordinator._select_diagnosis(classification, visual)

    assert diagnosis["disease_key"] == "Blast"
    assert diagnosis["source"] == "visual_verification_override"
