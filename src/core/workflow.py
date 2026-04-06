"""
workflow.py — Entry point for the Multi-Agent pipeline.
This module provides a simple function-based interface to run the coordinator,
so the API and UI layers stay decoupled from the agent internals.
"""
from PIL import Image
from src.agents.coordinator import CoordinatorAgent

# Singleton — only load models once per process
_coordinator_instance: CoordinatorAgent | None = None

def get_coordinator() -> CoordinatorAgent:
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = CoordinatorAgent()
    return _coordinator_instance

def run_diagnosis(image: Image.Image, progress_callback=None) -> dict:
    """
    Chạy toàn bộ quy trình chẩn đoán bệnh lúa từ một hình ảnh đầu vào.
    
    Args:
        image: PIL Image của lá lúa cần chẩn đoán.
        progress_callback: Hàm callback nhận chuỗi mô tả bước hiện tại (dùng cho UI).
    
    Returns:
        dict chứa: preprocessed_image, classification, morphology, knowledge, final_report
    """
    coordinator = get_coordinator()
    return coordinator.run(image, progress_callback=progress_callback)
