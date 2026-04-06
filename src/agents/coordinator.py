import asyncio
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from src.core.config import settings
from src.agents.preprocessing import PreprocessingAgent
from src.agents.classification import ClassificationAgent
from src.agents.morphology import MorphologyAgent
from src.agents.retrieval import RetrievalAgent

class CoordinatorAgent:
    def __init__(self):
        print("Initializing Coordinator Agent...")
        # Initialize all sub-agents
        self.preprocessor = PreprocessingAgent()
        self.classifier = ClassificationAgent()
        self.morphologist = MorphologyAgent()
        self.retriever = RetrievalAgent()
        
        # LLM for final report synthesis
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.4
        )

    def run(self, image: Image.Image, progress_callback=None) -> dict:
        """
        Điều phối toàn bộ quy trình chẩn đoán:
        1. Tiền xử lý ảnh
        2. Phân loại bệnh + Phân tích hình thái (song song)
        3. Tra cứu kiến thức bệnh học
        4. Tổng hợp báo cáo
        """
        results = {}

        # --- BƯỚC 1: Tiền xử lý ---
        if progress_callback: progress_callback("🔧 Đang tiền xử lý hình ảnh...")
        preprocessed_image = self.preprocessor.process(image)
        results["preprocessed_image"] = preprocessed_image

        # --- BƯỚC 2: Phân loại bệnh (ViT) ---
        if progress_callback: progress_callback("🔬 Đang phân tích bệnh bằng mô hình ViT...")
        classification_result = self.classifier.classify(preprocessed_image)
        results["classification"] = classification_result
        predicted_disease = classification_result["disease_label"]

        # --- BƯỚC 3: Phân tích hình thái (song song với bước trên logic-wise, sequential here) ---
        if progress_callback: progress_callback("🧬 Đang phân tích hình thái lá lúa bằng Gemini Vision...")
        morphology_analysis = self.morphologist.analyze(preprocessed_image, predicted_disease)
        results["morphology"] = morphology_analysis

        # --- BƯỚC 4: Tra cứu tri thức (RAG) ---
        if progress_callback: progress_callback("📚 Đang tra cứu cơ sở dữ liệu bệnh học...")
        knowledge_info = self.retriever.retrieve_info(predicted_disease)
        results["knowledge"] = knowledge_info

        # --- BƯỚC 5: Tổng hợp báo cáo cuối bằng LLM ---
        if progress_callback: progress_callback("📝 Đang tổng hợp báo cáo chẩn đoán cuối cùng...")
        final_report = self._synthesize_report(
            disease=predicted_disease,
            confidence=classification_result["confidence"],
            morphology=morphology_analysis,
            knowledge=knowledge_info
        )
        results["final_report"] = final_report

        return results

    def _synthesize_report(self, disease: str, confidence: float, morphology: str, knowledge: str) -> str:
        prompt = f"""Bạn là một hệ thống AI chuyên gia nông nghiệp. Dựa trên dữ liệu đầu vào từ các Agent khác, 
hãy tổng hợp một BÁO CÁO CHẨN ĐOÁN BỆNH LÚA hoàn chỉnh, rõ ràng và chuyên nghiệp bằng tiếng Việt.

**DỮ LIỆU ĐẦU VÀO:**
- Kết quả mô hình phân loại ViT: {disease} (Độ tin cậy: {confidence}%)
- Phân tích hình thái từ Gemini Vision:
{morphology}
- Thông tin bệnh học từ cơ sở dữ liệu:
{knowledge}

**YÊU CẦU BÁO CÁO:**
Hãy viết báo cáo theo cấu trúc sau:
## KẾT LUẬN CHẨN ĐOÁN
(Tên bệnh, mức độ tin cậy, xác nhận từ phân tích hình thái)

## ĐẶC ĐIỂM NHẬN DẠNG QUAN SÁT ĐƯỢC
(Tóm tắt các đặc điểm thị giác đã quan sát)

## NGUYÊN NHÂN & ĐIỀU KIỆN GÂY BỆNH
(Tác nhân gây bệnh, điều kiện thời tiết/môi trường thuận lợi)

## KHUYẾN NGHỊ PHÒNG TRỊ SINH HỌC
(Các biện pháp cụ thể, ưu tiên phương pháp sinh học an toàn)

Hãy viết súc tích, dễ hiểu, phù hợp cho cả nông dân lẫn chuyên gia nông nghiệp.
"""
        from langchain_core.messages import HumanMessage
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
