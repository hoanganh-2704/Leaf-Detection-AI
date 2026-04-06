import base64
import io
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from src.core.config import settings

class MorphologyAgent:
    def __init__(self):
        print("Initializing Morphology Agent (Gemini Vision)...")
        # We use a vision-capable Gemini model to analyze the image
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.2
        )
        
    def analyze(self, image: Image.Image, predicted_disease: str) -> str:
        """
        Phân tích hình thái của lá lúa và đối chiếu với kết quả dự đoán của mô hình phân loại.
        """
        # Convert PIL Image to Base64 to pass to Gemini
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        prompt = f"""Bạn là một chuyên gia về bệnh học thực vật. Hãy phân tích hình ảnh lá lúa được cung cấp và kiểm tra xem nó có khớp với chẩn đoán sơ bộ là '{predicted_disease}' hay không.
        
        Vui lòng tập trung mô tả các đặc trưng thị giác sau:
        1. Màu sắc của vết bệnh (vàng, nâu, xám, trắng...).
        2. Hình dạng của vết bệnh (hình thoi, vệt dài, chấm tròn, viền mờ...).
        3. Sự phân bố của vết bệnh trên lá.
        
        Sau khi mô tả, hãy kết luận logic xem các đặc điểm này có thực sự đặc trưng cho bệnh '{predicted_disease}' hay không.
        Lưu ý: Không cần đưa ra cách phòng trị (sẽ có Agent khác lo), chỉ tập trung vào phân tích hình thái.
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
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    }
                ]
            )
        ]
        
        response = self.llm.invoke(messages)
        return response.content
