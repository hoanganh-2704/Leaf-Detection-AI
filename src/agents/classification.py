import torch
from transformers import AutoProcessor, SiglipForImageClassification
from PIL import Image
from src.core.config import settings

class ClassificationAgent:
    def __init__(self):
        self.model_name = settings.VIT_MODEL_NAME
        print(f"Loading Classification Model: {self.model_name}")
        auth_token = settings.HF_TOKEN if settings.HF_TOKEN else None
        
        # Use SiglipForImageClassification explicitly — this model is based on
        # google/siglip2-base-patch16-224, not a generic ViT. Using Auto may
        # load a mismatched architecture and silently produce wrong predictions.
        self.processor = AutoProcessor.from_pretrained(self.model_name, token=auth_token)
        self.model = SiglipForImageClassification.from_pretrained(self.model_name, token=auth_token)
        self.model.eval()
        
    def classify(self, image: Image.Image) -> dict:
        """
        Nhận diện bệnh từ hình ảnh đã qua tiền xử lý bằng mô hình SigLIP2.
        Trả về dictionary chứa tên bệnh và độ tin cậy.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        
        # Lấy nhãn và tính toán score
        predicted_label = self.model.config.id2label[predicted_class_idx]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        confidence = probabilities[predicted_class_idx].item()
        
        return {
            "disease_label": predicted_label,
            "confidence": round(confidence * 100, 2),
            "all_probabilities": {self.model.config.id2label[i]: round(p.item() * 100, 2) for i, p in enumerate(probabilities)}
        }
