import base64
import io
import json
import re
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from src.core.config import settings
from src.agents.classification import label_display_name, normalize_label_key

class MorphologyAgent:
    def __init__(self):
        print("Initializing Morphology Agent (Gemini Vision)...")
        # We use a vision-capable Gemini model to analyze the image
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.2
        )

    def _image_to_data_url(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def _extract_json(self, content: str) -> dict:
        text = content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
        return json.loads(text)

    def _parse_bool(self, value, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1", "đúng", "co", "có"}:
                return True
            if lowered in {"false", "no", "0", "sai", "khong", "không"}:
                return False
        return default

    def verify(self, image: Image.Image, model_prediction: str, model_probabilities: dict | None = None) -> dict:
        """
        Ask Gemini Vision for an independent diagnosis and symptom check.
        The classifier prediction is provided as context, but the prompt tells
        the model not to assume it is correct.
        """
        probabilities_text = "Không có"
        if model_probabilities:
            probabilities_text = "\n".join(
                f"- {label}: {score}%"
                for label, score in sorted(model_probabilities.items(), key=lambda item: item[1], reverse=True)
            )

        prompt = f"""Bạn là chuyên gia bệnh học cây lúa. Hãy đọc ảnh lá lúa và đưa ra đánh giá độc lập; KHÔNG mặc định mô hình phân loại là đúng.

Các nhãn hợp lệ:
- Bacterialblight: bạc lá / cháy bìa lá do vi khuẩn, vệt dài từ chóp hoặc mép lá, màu vàng rơm đến trắng bạc, mép gợn sóng.
- Blast: đạo ôn, vết hình thoi hoặc mắt én, tâm xám/trắng, viền nâu, có thể rải rác trên phiến lá.
- Brownspot: đốm nâu, đốm tròn hoặc bầu dục màu nâu, thường có viền đậm/vầng vàng.
- Tungro: vàng lùn, lá vàng/cam lan từ chóp, cây lùn hoặc vàng toàn bộ, ít thấy vết hoại tử riêng lẻ.
- Healthy: lá khỏe, không thấy triệu chứng bệnh rõ ràng.
- Unknown: ảnh mờ, không phải lá lúa, hoặc không đủ bằng chứng.

Kết quả mô hình hiện tại: {model_prediction}
Xác suất mô hình:
{probabilities_text}

Chỉ trả về JSON hợp lệ, không thêm markdown:
{{
  "suggested_label": "Bacterialblight | Blast | Brownspot | Tungro | Healthy | Unknown",
  "confidence": 0-100,
  "agrees_with_model": true/false,
  "symptoms": "mô tả ngắn các đặc điểm quan sát được",
  "reasoning": "giải thích ngắn vì sao chọn nhãn này"
}}
        """
        
        messages = [
            HumanMessage(
                content=[
                    {
                        "type": "text", 
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._image_to_data_url(image)
                        }
                    }
                ]
            )
        ]
        
        try:
            response = self.llm.invoke(messages)
        except Exception:
            message = "Không thể phân tích triệu chứng chi tiết ở thời điểm hiện tại."
            return {
                "suggested_key": None,
                "suggested_label": "Unknown",
                "confidence": 0,
                "agrees_with_model": False,
                "symptoms": message,
                "reasoning": "Sử dụng kết quả mô hình phân loại làm dự phòng.",
                "analysis": message,
                "raw_response": message,
            }

        content = response.content if isinstance(response.content, str) else str(response.content)

        try:
            parsed = self._extract_json(content)
        except Exception:
            parsed = {
                "suggested_label": "Unknown",
                "confidence": 0,
                "agrees_with_model": False,
                "symptoms": content,
                "reasoning": "Không đọc được JSON từ phản hồi Gemini.",
            }

        suggested_key = normalize_label_key(parsed.get("suggested_label"))
        suggested_label = label_display_name(suggested_key) if suggested_key else "Unknown"
        confidence = parsed.get("confidence", 0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0
        confidence = max(0, min(confidence, 100))

        symptoms = str(parsed.get("symptoms", "")).strip()
        reasoning = str(parsed.get("reasoning", "")).strip()
        agrees_with_model = self._parse_bool(
            parsed.get("agrees_with_model"),
            suggested_key == normalize_label_key(model_prediction),
        )
        analysis = (
            f"**Kết quả quan sát:** {suggested_label} ({confidence:.0f}%).\n\n"
            f"**Triệu chứng:** {symptoms}\n\n"
            f"**Nhận định:** {reasoning}"
        )

        return {
            "suggested_key": suggested_key,
            "suggested_label": suggested_label,
            "confidence": round(confidence, 2),
            "agrees_with_model": agrees_with_model,
            "symptoms": symptoms,
            "reasoning": reasoning,
            "analysis": analysis,
            "raw_response": content,
        }

    def analyze(self, image: Image.Image, predicted_disease: str) -> str:
        """
        Phân tích hình thái của lá lúa và đối chiếu với kết quả dự đoán của mô hình phân loại.
        """
        return self.verify(image, predicted_disease)["analysis"]
