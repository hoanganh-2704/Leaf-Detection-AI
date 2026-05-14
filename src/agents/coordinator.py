import asyncio
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from src.core.config import settings
from src.agents.preprocessing import PreprocessingAgent
from src.agents.classification import ClassificationAgent
from src.agents.classification import label_display_name, label_vietnamese_name
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

        # --- BƯỚC 2: Phân loại bệnh (SigLIP) ---
        if progress_callback: progress_callback("🔬 Đang phân tích bệnh bằng mô hình SigLIP...")
        classification_result = self.classifier.classify(preprocessed_image)
        results["classification"] = classification_result
        predicted_disease_key = classification_result.get("disease_key", classification_result["disease_label"])
        predicted_disease_label = classification_result["disease_label"]

        # --- BƯỚC 3: Phân tích hình thái (song song với bước trên logic-wise, sequential here) ---
        if progress_callback: progress_callback("🧬 Đang phân tích triệu chứng trên lá lúa...")
        visual_verification = self.morphologist.verify(
            preprocessed_image,
            predicted_disease_label,
            model_probabilities=classification_result.get("all_probabilities"),
        )
        morphology_analysis = visual_verification["analysis"]
        diagnosis_result = self._select_diagnosis(classification_result, visual_verification)
        results["morphology"] = morphology_analysis
        results["visual_verification"] = visual_verification
        results["diagnosis"] = diagnosis_result

        # --- BƯỚC 4: Tra cứu tri thức (RAG) ---
        if progress_callback: progress_callback("📚 Đang tra cứu cơ sở dữ liệu bệnh học...")
        knowledge_info = self.retriever.retrieve_info(diagnosis_result["disease_key"])
        results["knowledge"] = knowledge_info

        # --- BƯỚC 5: Tổng hợp báo cáo cuối bằng LLM ---
        if progress_callback: progress_callback("📝 Đang tổng hợp báo cáo chẩn đoán cuối cùng...")
        final_report = self._synthesize_report(
            disease=diagnosis_result["disease_label"],
            disease_key=diagnosis_result["disease_key"],
            confidence=diagnosis_result["confidence"],
            morphology=morphology_analysis,
            knowledge=knowledge_info
        )
        results["final_report"] = final_report

        return results

    def _select_diagnosis(self, classification: dict, visual_verification: dict) -> dict:
        model_key = classification.get("disease_key")
        model_confidence = float(classification.get("confidence", 0))
        visual_key = visual_verification.get("suggested_key")
        visual_confidence = float(visual_verification.get("confidence", 0))

        final_key = model_key
        confidence = model_confidence
        source = "classifier"
        note = "Kết quả chẩn đoán cuối cùng."

        if visual_key == model_key:
            source = "classifier_and_visual_verification"
            note = "Kết quả chẩn đoán cuối cùng."
        elif visual_key and visual_confidence >= 75:
            final_key = visual_key
            confidence = visual_confidence
            source = "visual_verification_override"
            note = "Kết quả chẩn đoán cuối cùng."
        elif visual_key and model_confidence < 75:
            final_key = visual_key
            confidence = visual_confidence
            source = "visual_verification_low_model_confidence"
            note = "Kết quả chẩn đoán cuối cùng."
        elif visual_key and visual_key != model_key:
            note = "Kết quả chẩn đoán cuối cùng."
        elif model_confidence < 70:
            source = "low_confidence_classifier"
            note = "Mô hình phân loại có độ tin cậy thấp; nên chụp thêm ảnh rõ hơn để xác nhận."

        return {
            "disease_key": final_key,
            "disease_label": label_display_name(final_key),
            "disease_label_vi": label_vietnamese_name(final_key),
            "confidence": round(confidence, 2),
            "source": source,
            "note": note,
            "model_disease_key": model_key,
            "model_disease_label": classification.get("disease_label"),
            "model_confidence": model_confidence,
            "visual_disease_key": visual_key,
            "visual_disease_label": visual_verification.get("suggested_label", "Unknown"),
            "visual_confidence": visual_confidence,
        }

    def _synthesize_report(self, disease: str, disease_key: str, confidence: float, morphology: str, knowledge: str) -> str:
        if disease_key == "Healthy":
            diagnosis_instruction = """Vì kết quả cuối cùng là Healthy, hãy viết báo cáo theo hướng KHÔNG phát hiện dấu hiệu bệnh rõ ràng. Không bịa tên bệnh, không đưa phác đồ thuốc trị bệnh cụ thể; chỉ khuyến nghị theo dõi, chụp lại ảnh rõ hơn nếu cần, và duy trì chăm sóc phòng ngừa."""
        else:
            diagnosis_instruction = """Hãy viết báo cáo trực tiếp cho bệnh trong kết quả cuối cùng. Không nhắc đến bất kỳ bất đồng nào giữa các mô hình nội bộ."""

        prompt = f"""Bạn là một hệ thống AI chuyên gia nông nghiệp. Dựa trên dữ liệu đầu vào từ các Agent khác, 
hãy tổng hợp một BÁO CÁO CHẨN ĐOÁN BỆNH LÚA hoàn chỉnh, rõ ràng và chuyên nghiệp bằng tiếng Việt.

**DỮ LIỆU ĐẦU VÀO:**
- Kết quả chẩn đoán cuối cùng: {disease} (Độ tin cậy: {confidence}%)
- Phân tích triệu chứng:
{morphology}
- Thông tin bệnh học từ cơ sở dữ liệu:
{knowledge}

**YÊU CẦU BÁO CÁO:**
{diagnosis_instruction}

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
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception:
            return f"""## KẾT LUẬN CHẨN ĐOÁN
{disease} (độ tin cậy: {confidence}%).

## ĐẶC ĐIỂM NHẬN DẠNG QUAN SÁT ĐƯỢC
{morphology}

## THÔNG TIN THAM KHẢO
{knowledge}"""
