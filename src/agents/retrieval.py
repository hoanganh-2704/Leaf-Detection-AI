import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.core.config import settings


HEALTHY_MESSAGE = """Không phát hiện bệnh trên lá lúa trong kết quả phân loại.

Khuyến nghị:
- Tiếp tục theo dõi ruộng định kỳ, đặc biệt sau mưa, sương mù hoặc giai đoạn bón phân.
- Duy trì bón phân cân đối, quản lý nước ổn định và vệ sinh đồng ruộng.
- Nếu ảnh có ánh sáng yếu, bị mờ, hoặc lá bị che khuất, nên chụp lại ảnh rõ hơn để xác nhận."""


class RetrievalAgent:
    def __init__(self):
        print("Initializing Retrieval Agent and loading Vector DB...")
        self.embeddings = None
        
        # Load ChromaDB if it exists
        if os.path.exists(settings.VECTOR_DB_PATH):
            self.vectorstore = Chroma(
                persist_directory=settings.VECTOR_DB_PATH,
            )
        else:
            self.vectorstore = None
            print("Warning: Vector database not found. Please run knowledge_setup.py first.")

    def _get_embeddings(self):
        if self.embeddings is None:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-2",
                google_api_key=settings.GEMINI_API_KEY,
            )
        return self.embeddings
            
    def retrieve_info(self, disease_query: str) -> str:
        """
        Tra cứu thông tin bệnh học (tác nhân, điều kiện phát triển, biện pháp phòng ngừa)
        từ Vector Database dựa trên tên bệnh được nhận diện.
        """
        if disease_query == "Healthy":
            return HEALTHY_MESSAGE

        if not self.vectorstore:
            return "Không tìm thấy cơ sở dữ liệu bệnh học. Vui lòng chạy `src/core/knowledge_setup.py` để khởi tạo Vector DB."

        # Prefer exact metadata lookup. With only a few disease documents,
        # embedding search can pick the wrong disease when labels are compact
        # model keys such as "Brownspot" or "Bacterialblight".
        try:
            matches = self.vectorstore.get(
                where={"disease": disease_query},
                include=["documents", "metadatas"],
                limit=1,
            )
            documents = matches.get("documents", [])
            if documents:
                return documents[0]
        except Exception as exc:
            print(f"Warning: Exact Chroma metadata lookup failed: {exc}")

        # Fallback for older/corrupt indexes that do not support metadata get.
        similarity_store = Chroma(
            persist_directory=settings.VECTOR_DB_PATH,
            embedding_function=self._get_embeddings(),
        )
        docs = similarity_store.similarity_search(disease_query, k=1)
        
        if docs:
            return docs[0].page_content
        else:
            return f"Không tìm thấy tài liệu tham khảo cho: {disease_query}"
