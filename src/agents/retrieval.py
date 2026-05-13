import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.core.config import settings

class RetrievalAgent:
    def __init__(self):
        print("Initializing Retrieval Agent and loading Vector DB...")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2", 
            google_api_key=settings.GEMINI_API_KEY
        )
        
        # Load ChromaDB if it exists
        if os.path.exists(settings.VECTOR_DB_PATH):
            self.vectorstore = Chroma(
                persist_directory=settings.VECTOR_DB_PATH, 
                embedding_function=self.embeddings
            )
        else:
            self.vectorstore = None
            print("Warning: Vector database not found. Please run knowledge_setup.py first.")
            
    def retrieve_info(self, disease_query: str) -> str:
        """
        Tra cứu thông tin bệnh học (tác nhân, điều kiện phát triển, biện pháp phòng ngừa)
        từ Vector Database dựa trên tên bệnh được nhận diện.
        """
        if not self.vectorstore:
            return "Không tìm thấy cơ sở dữ liệu bệnh học. Vui lòng chạy `src/core/knowledge_setup.py` để khởi tạo Vector DB."
            
        # Tìm kiếm document liên quan nhất
        docs = self.vectorstore.similarity_search(disease_query, k=1)
        
        if docs:
            return docs[0].page_content
        else:
            return f"Không tìm thấy tài liệu tham khảo cho: {disease_query}"
