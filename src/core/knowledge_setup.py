import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from src.core.config import settings

def setup_knowledge_base():
    """
    Tạo và lưu trữ cơ sở dữ liệu Vector (ChromaDB) chứa thông tin về 4 loại bệnh:
    Đạo ôn, Bạc lá, Đốm nâu, và Vàng lùn.
    """
    
    # 1. Khởi tạo Embeddings
    print("Initializing Google Generative AI Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=settings.GEMINI_API_KEY)
    
    # 2. Chuẩn bị dữ liệu bệnh học (Knowledge Base)
    documents = [
        Document(
            page_content="""Bệnh Đạo ôn (Rice Blast):
Tác nhân: Nấm Pyricularia oryzae.
Đặc điểm nhận dạng: Vết bệnh ban đầu là những chấm nhỏ màu xanh lục sẫm, sau lớn dần thành hình thoi, tâm màu xám nhạt, viền màu nâu đậm. Thường xuất hiện trên lá, cổ bông.
Điều kiện phát triển: Độ ẩm cao, trời âm u, sương mù, nhiệt độ từ 20-30°C.
Biện pháp phòng ngừa: Sử dụng giống chống chịu, bón phân cân đối (không bón thừa đạm), giữ mực nước ruộng phù hợp, sử dụng thuốc bảo vệ thực vật sinh học khi bệnh chớm xuất hiện.""",
            metadata={"disease": "Blast", "name_vn": "Đạo ôn"}
        ),
        Document(
            page_content="""Bệnh Bạc lá (Bacterial Leaf Blight):
Tác nhân: Vi khuẩn Xanthomonas oryzae.
Đặc điểm nhận dạng: Các dải bạc hoặc vàng nhạt chạy dọc theo mép lá. Vết bệnh có thể bắt đầu từ chóp lá lan dần xuống dưới. Buổi sáng sớm có thể thấy giọt dịch vi khuẩn màu đục đọng ở mép lá.
Điều kiện phát triển: Nhiệt độ cao (28-30°C), độ ẩm cao, mưa to gió lớn làm lây lan vi khuẩn.
Biện pháp phòng ngừa: Không bón thừa đạm vãi muộn, bón bổ sung kali để tăng khả năng chống chịu. Rút nước phơi ruộng khi bệnh chớm xuất hiện. Phun các loại thuốc đặc trị vi khuẩn.""",
            metadata={"disease": "Blight", "name_vn": "Bạc lá"}
        ),
        Document(
            page_content="""Bệnh Đốm nâu (Brown Spot):
Tác nhân: Nấm Bipolaris oryzae (hoặc Helminthosporium oryzae).
Đặc điểm nhận dạng: Vết bệnh màu nâu, có hình tròn hoặc bầu dục. Kích thước vết bệnh thường bằng hạt vừng. Tâm vết bệnh thường có màu xám hoặc trắng ở giai đoạn muộn.
Điều kiện phát triển: Ruộng thiếu dinh dưỡng (đặc biệt thiếu Kali, Silic), đất phèn, ngộ độc hữu cơ, nhiệt độ từ 25-30°C.
Biện pháp phòng ngừa: Cải tạo đất, bón phân cân đối và đầy đủ dinh dưỡng hữu cơ, giữ nước ruộng ổn định để tránh xì phèn.""",
            metadata={"disease": "Brown Spot", "name_vn": "Đốm nâu"}
        ),
        Document(
            page_content="""Bệnh Vàng lùn (Rice Tungro):
Tác nhân: Virus (RTSV và RTBV) do rầy xanh đuôi đen lây truyền.
Đặc điểm nhận dạng: Cây lúa bị lùn lại, lá có màu vàng, biến dạng. Khác với thiếu dinh dưỡng, màu vàng thường xuất hiện ở các lá già phía dưới rồi lan dần lên trên đỉnh.
Điều kiện phát triển: Phụ thuộc vào mật độ rầy xanh đuôi đen mang mầm bệnh. Thời tiết nắng nóng xem kẽ mưa dông thuận lợi cho rầy phát triển.
Biện pháp phòng ngừa: Diệt trừ rầy xanh đuôi đen bảo vệ thiên địch, gieo sạ né rầy, vệ sinh đồng ruộng sạch sẽ, sử dụng giống kháng rầy.""",
            metadata={"disease": "Tungro", "name_vn": "Vàng lùn"}
        )
    ]
    
    # 3. Tạo ChromaDB
    print(f"Creating ChromaDB at: {settings.VECTOR_DB_PATH}")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=settings.VECTOR_DB_PATH
    )
    
    # Persist the DB (automatic in newer versions of Chroma)
    print("Vector database created and persisted successfully!")

if __name__ == "__main__":
    setup_knowledge_base()
