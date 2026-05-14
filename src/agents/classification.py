import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
from src.core.config import settings


LABEL_METADATA = {
    "Bacterialblight": {
        "display": "Bacterial Blight",
        "vietnamese": "Bạc lá",
    },
    "Blast": {
        "display": "Rice Blast",
        "vietnamese": "Đạo ôn",
    },
    "Brownspot": {
        "display": "Brown Spot",
        "vietnamese": "Đốm nâu",
    },
    "Healthy": {
        "display": "Healthy",
        "vietnamese": "Lá khỏe",
    },
    "Tungro": {
        "display": "Tungro",
        "vietnamese": "Vàng lùn",
    },
}

LABEL_ALIASES = {
    "bacterialblight": "Bacterialblight",
    "bacterial blight": "Bacterialblight",
    "bacterial leaf blight": "Bacterialblight",
    "bac la": "Bacterialblight",
    "bạc lá": "Bacterialblight",
    "blast": "Blast",
    "rice blast": "Blast",
    "dao on": "Blast",
    "đạo ôn": "Blast",
    "brownspot": "Brownspot",
    "brown spot": "Brownspot",
    "dom nau": "Brownspot",
    "đốm nâu": "Brownspot",
    "healthy": "Healthy",
    "la khoe": "Healthy",
    "lá khỏe": "Healthy",
    "tungro": "Tungro",
    "vang lun": "Tungro",
    "vàng lùn": "Tungro",
}


def label_display_name(raw_label: str) -> str:
    return LABEL_METADATA.get(raw_label, {}).get("display", raw_label)


def label_vietnamese_name(raw_label: str) -> str:
    return LABEL_METADATA.get(raw_label, {}).get("vietnamese", label_display_name(raw_label))


def normalize_label_key(label: str | None) -> str | None:
    if not label:
        return None
    cleaned = str(label).strip()
    if cleaned in LABEL_METADATA:
        return cleaned
    return LABEL_ALIASES.get(cleaned.lower())


def load_hf_component(component_cls, model_name: str, auth_token: str | None):
    kwargs = {"token": auth_token} if auth_token else {}
    try:
        return component_cls.from_pretrained(
            model_name,
            local_files_only=True,
            **kwargs,
        )
    except Exception:
        return component_cls.from_pretrained(model_name, **kwargs)


class ClassificationAgent:
    def __init__(self):
        self.model_name = settings.VIT_MODEL_NAME
        print(f"Loading Classification Model: {self.model_name}")
        auth_token = settings.HF_TOKEN if settings.HF_TOKEN else None
        
        # This checkpoint is image-classification only. AutoProcessor tries to
        # build a SigLIP tokenizer as well and fails before inference starts.
        self.processor = load_hf_component(AutoImageProcessor, self.model_name, auth_token)
        self.model = load_hf_component(SiglipForImageClassification, self.model_name, auth_token)
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
        
        # Lấy nhãn và tính toán score. Keep the raw model key for exact
        # metadata lookup, and expose readable names to API/UI/report layers.
        raw_label = self.model.config.id2label[predicted_class_idx]
        display_label = label_display_name(raw_label)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        confidence = probabilities[predicted_class_idx].item()
        
        return {
            "disease_key": raw_label,
            "disease_label": display_label,
            "disease_label_vi": LABEL_METADATA.get(raw_label, {}).get("vietnamese", display_label),
            "confidence": round(confidence * 100, 2),
            "all_probabilities": {
                label_display_name(self.model.config.id2label[i]): round(p.item() * 100, 2)
                for i, p in enumerate(probabilities)
            },
            "raw_probabilities": {
                self.model.config.id2label[i]: round(p.item() * 100, 2)
                for i, p in enumerate(probabilities)
            },
        }
