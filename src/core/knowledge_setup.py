import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from src.core.config import settings

def setup_knowledge_base():
    """
    Tạo và lưu trữ cơ sở dữ liệu Vector (ChromaDB) chứa thông tin về 4 loại bệnh lúa:
    Đạo ôn, Bạc lá, Đốm nâu, và Vàng lùn.

    Nguồn tài liệu:
    - IRRI Rice Knowledge Bank: https://www.knowledgebank.irri.org/
    - Ou, S.H. (1985). Rice Diseases. 2nd ed. Commonwealth Agricultural Bureaux, UK.
    - Mew, T.W. & Misra, J.K. (1994). A Manual of Rice Seed Health Testing. IRRI, Philippines.
    - Savary, S. et al. (2019). The global burden of pathogens and pests on major food crops.
      Nature Ecology & Evolution, 3(3), 430–439. https://doi.org/10.1038/s41559-018-0793-y
    - Ou, S.H. (1972). Rice Diseases. Commonwealth Mycological Institute, UK.
    - Gnanamanickam, S.S. (2009). Biological Control of Rice Diseases. Springer.
    """

    # 1. Khởi tạo Embeddings
    print("Initializing Google Generative AI Embeddings (text-embedding-004)...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.GEMINI_API_KEY
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize embeddings: {e}")
        return

    # 2. Cơ sở tri thức bệnh học — trích xuất từ tài liệu khoa học công bố
    documents = [

        # ── BỆNH ĐẠO ÔN ──────────────────────────────────────────────────────
        # Nguồn: IRRI Rice Knowledge Bank; Ou (1985) Rice Diseases 2nd ed.;
        #        Savary et al. (2019) Nature Ecology & Evolution 3:430-439.
        Document(
            page_content="""Bệnh Đạo ôn lúa (Rice Blast)
Tác nhân gây bệnh: Nấm Magnaporthe oryzae (B.C. Couch & L.M. Kohn, 2002), anamorph Pyricularia oryzae Cavara (1891). Đây là một trong những bệnh phá hoại nghiêm trọng nhất trên lúa toàn cầu, được ghi nhận gây thiệt hại đáng kể ở hơn 85 quốc gia (Savary et al., 2019).

Triệu chứng nhận dạng:
- Trên lá (Leaf Blast): Vết bệnh ban đầu là những chấm nhỏ xanh xám, sau phát triển thành dạng hình thoi (spindle-shaped) hoặc hình thoi mở rộng. Tâm vết bệnh có màu xám nhạt, viền nâu đậm đặc trưng, xung quanh đôi khi có vầng vàng nhạt (yellow halo). Vết bệnh nặng liên kết với nhau làm cháy toàn bộ lá (Ou, 1985).
- Trên cổ bông (Neck Blast): Vết bệnh xuất hiện tại đốt cổ bông, gây thối và gãy cổ bông, dẫn đến lép hạt hoàn toàn (white ear) — đây là dạng nguy hiểm nhất về kinh tế (IRRI Rice Knowledge Bank).
- Trên đốt thân (Node Blast): Đốt thân chuyển màu nâu đen, cây dễ gãy.

Điều kiện phát dịch:
- Nhiệt độ tối ưu: 20–28°C, đặc biệt nguy hiểm khi biên độ nhiệt ngày-đêm lớn.
- Ẩm độ không khí > 90% và thời gian ướt lá (leaf wetness) kéo dài từ 10–16 giờ liên tục (IRRI Rice Knowledge Bank).
- Bón thừa đạm (nitrogen) làm tăng đáng kể mức độ nhiễm bệnh.
- Sương mù và ánh sáng yếu tạo điều kiện thuận lợi cho bào tử nảy mầm.

Biện pháp phòng trị (theo khuyến cáo của IRRI):
1. Sử dụng giống kháng bệnh — biện pháp hiệu quả và bền vững nhất.
2. Bón phân cân đối, tránh bón thừa đạm, chia nhỏ lần bón.
3. Giữ mực nước ruộng ổn định, tránh để ruộng khô hạn đột ngột.
4. Tỉa cây thưa thoáng để tăng lưu thông không khí.
5. Phun thuốc nấm (triazole, strobilurin) khi bệnh chớm xuất hiện tại cổ bông; ưu tiên chế phẩm sinh học.
6. Vệ sinh đồng ruộng, tiêu hủy tàn dư cây bệnh sau thu hoạch.""",
            metadata={
                "disease": "Blast",
                "name_vn": "Đạo ôn",
                "pathogen_latin": "Magnaporthe oryzae",
                "sources": "IRRI Rice Knowledge Bank; Ou (1985); Savary et al. (2019) Nature Ecol Evol 3:430-439"
            }
        ),

        # ── BỆNH BẠC LÁ ───────────────────────────────────────────────────────
        # Nguồn: IRRI Rice Knowledge Bank; Mew & Misra (1994) IRRI;
        #        Niño-Liu et al. (2006) Molecular Plant Pathology 7(5):303-324.
        Document(
            page_content="""Bệnh Bạc lá lúa (Bacterial Leaf Blight — BLB)
Tác nhân gây bệnh: Vi khuẩn Xanthomonas oryzae pv. oryzae (Ishiyama, 1922; Swings et al., 1990). Đây là bệnh vi khuẩn quan trọng nhất trên lúa, phổ biến ở châu Á và châu Phi; trong điều kiện dịch nặng có thể gây thiệt hại năng suất từ 20–30%, thậm chí 50–75% ở giai đoạn mạ (IRRI Rice Knowledge Bank).

Triệu chứng nhận dạng:
- Dạng Bạc lá điển hình (Leaf Blight): Vết bệnh bắt đầu là sọc thấm nước ở mép lá hoặc chóp lá, lan dần thành dải màu vàng đến vàng rơm, sau chuyển trắng bạc. Vết bệnh có đường viền gợn sóng đặc trưng (wavy margin). Buổi sáng sớm, trên vết bệnh non xuất hiện giọt dịch vi khuẩn màu đục (bacterial ooze) — đây là đặc điểm chẩn đoán quan trọng nhất (IRRI Rice Knowledge Bank; Niño-Liu et al., 2006).
- Dạng Héo xanh (Kresek): Thường xảy ra ở giai đoạn mạ đến đẻ nhánh sớm tại vùng nhiệt đới. Lá cuộn lại, chuyển xanh xám và héo rũ; cả cây có thể chết trong vòng 1–2 tuần.

Dịch tễ học:
- Mầm bệnh lưu tồn trong gốc rạ, cây lúa chét, và các loài cỏ ký chủ (Leersia spp., Zizania spp.).
- Lây lan qua nước tưới, mưa bắn, gió, và tiếp xúc cây-cây trong quá trình cấy.
- Nhiệt độ thuận lợi: 25–34°C; ẩm độ > 70%; ngập lụt và mưa bão tạo điều kiện bùng phát dịch.
- Bón thừa đạm và ngập sâu làm tăng mức độ nhiễm bệnh nghiêm trọng.

Biện pháp phòng trị (theo khuyến cáo của IRRI):
1. Trồng giống kháng — biện pháp chủ lực và bền vững nhất; cần theo dõi chủng vi khuẩn địa phương.
2. Tránh bón thừa đạm, đặc biệt không vãi đạm muộn giai đoạn đứng cái.
3. Rút nước phơi ruộng khi bệnh chớm xuất hiện; duy trì mực nước nông.
4. Cày vùi tàn dư và gốc rạ sau thu hoạch; loại trừ cỏ ký chủ xung quanh ruộng.
5. Xử lý hạt giống bằng nhiệt hoặc hóa chất trước khi gieo; không lấy giống từ ruộng bệnh.
6. Không có thuốc hóa học đặc hiệu cao; biện pháp canh tác và giống kháng là chủ đạo.""",
            metadata={
                "disease": "Bacterialblight",
                "name_vn": "Bạc lá",
                "pathogen_latin": "Xanthomonas oryzae pv. oryzae",
                "sources": "IRRI Rice Knowledge Bank; Niño-Liu et al. (2006) Mol Plant Pathol 7(5):303-324; Mew & Misra (1994) IRRI"
            }
        ),

        # ── BỆNH ĐỐM NÂU ──────────────────────────────────────────────────────
        # Nguồn: IRRI Rice Knowledge Bank; Ou (1985) Rice Diseases 2nd ed.;
        #        Barnwal et al. (2013) Annual Review of Phytopathology 51:49-70.
        Document(
            page_content="""Bệnh Đốm nâu lúa (Brown Spot)
Tác nhân gây bệnh: Nấm Bipolaris oryzae (Breda de Haan, 1900) Shoemaker (1959), teleomorph Cochliobolus miyabeanus (Ito & Kuribayashi, 1931). Bệnh được ghi nhận lịch sử là tác nhân góp phần gây nạn đói Bengal năm 1943, phá hủy 50–90% sản lượng lúa (Barnwal et al., 2013). Hiện phổ biến toàn cầu, đặc biệt trên đất nghèo dinh dưỡng và đất phèn.

Triệu chứng nhận dạng:
- Trên lá: Vết bệnh ban đầu là những chấm nhỏ màu nâu tím; phát triển thành dạng hình tròn hoặc bầu dục (oval). Tâm vết bệnh màu nâu nhạt đến xám, viền nâu đậm hoặc tím nâu, thường có vầng vàng (yellow halo) bao quanh — đây là đặc điểm phân biệt với bệnh khác (Ou, 1985; IRRI Rice Knowledge Bank).
- Trên hạt (Grain Discoloration): Nấm tấn công vỏ trấu và nội nhũ, tạo ra hạt lem lép (pecky rice) — hạt đổi màu, giảm trọng lượng và chất lượng xay xát.
- Trên mầm (Seedling Blight): Nấm lây qua hạt giống, có thể gây thối mầm, giảm tỷ lệ nảy mầm.

Dịch tễ học:
- Nấm tồn tại trong hạt giống trên 4 năm và trong tàn dư cây trồng (rơm rạ).
- Điều kiện thuận lợi: Ẩm độ 86–100%, nhiệt độ 16–36°C (tối ưu 25–30°C), thời gian ướt lá từ 8–24 giờ.
- Bệnh nặng hơn trên đất nghèo dinh dưỡng (đặc biệt thiếu Kali và đạm), đất chua phèn, đất ngập mặn, hoặc khi cây bị stress hạn.
- Bào tử phát tán qua không khí, lây nhiễm rất nhanh trong giai đoạn trổ bông.

Biện pháp phòng trị (theo khuyến cáo của IRRI và Barnwal et al., 2013):
1. Sử dụng hạt giống sạch bệnh, được kiểm định chứng nhận; xử lý hạt bằng thuốc trừ nấm trước khi gieo.
2. Bón phân cân đối, đặc biệt đảm bảo đủ Kali và Đạm; cải thiện dinh dưỡng đất trên ruộng phèn.
3. Cải tạo đất, cải thiện thoát nước; san phẳng mặt ruộng để tránh ngập cục bộ.
4. Luân canh với cây trồng khác lúa; tiêu hủy tàn dư sau thu hoạch để giảm nguồn lây.
5. Phun thuốc nấm (mancozeb, propiconazole) khi bệnh xuất hiện ở giai đoạn đòng — trổ.
6. Trồng giống chống chịu (không có giống hoàn toàn miễn dịch, nhưng có giống chịu được).""",
            metadata={
                "disease": "Brownspot",
                "name_vn": "Đốm nâu",
                "pathogen_latin": "Bipolaris oryzae",
                "sources": "IRRI Rice Knowledge Bank; Barnwal et al. (2013) Annu Rev Phytopathol 51:49-70; Ou (1985)"
            }
        ),

        # ── BỆNH VÀNG LÙN (TUNGRO) ───────────────────────────────────────────
        # Nguồn: IRRI Rice Knowledge Bank; Azzam & Chancellor (2002)
        #        Plant Disease 86(2):88-100; Hibino (1996) Annu Rev Phytopathol 34:249-274.
        Document(
            page_content="""Bệnh Vàng lùn lúa (Rice Tungro Disease — RTD)
Tác nhân gây bệnh: Phức hợp hai virus khác nhau:
- Rice Tungro Bacilliform Virus (RTBV): DNA virus nhóm Badnavirus, là tác nhân gây triệu chứng chính (lùn, vàng lá).
- Rice Tungro Spherical Virus (RTSV): RNA virus nhóm Waikavirus, thường không gây triệu chứng khi nhiễm đơn độc, nhưng bắt buộc phải có mặt để RTBV lây truyền qua môi giới.
(Hibino, 1996; Azzam & Chancellor, 2002)

Vector truyền bệnh:
- Môi giới chính: Rầy xanh đuôi đen (Green Leafhopper — GLH), chủ yếu là Nephotettix virescens (Distant), cùng N. nigropictus, Recilia dorsalis. Cách truyền: bán bền vững (semi-persistent) — rầy có thể truyền bệnh ngay sau thời gian chích hút ngắn (15–30 phút) và mất khả năng truyền bệnh sau vài giờ đến vài ngày nếu không tiếp tục hút cây bệnh (IRRI Rice Knowledge Bank).

Triệu chứng nhận dạng:
- Cây lùn rõ rệt, số nhánh đẻ giảm, hệ rễ kém phát triển.
- Lá chuyển vàng đến vàng cam, bắt đầu từ chóp lá và mép lá trên các lá non-trung, không phải từ lá già lên như thiếu dinh dưỡng — đây là điểm phân biệt quan trọng (Azzam & Chancellor, 2002).
- Trổ bông muộn, bông nhỏ, hạt lép nhiều hoặc không trổ được.
- Bệnh không lây qua tiếp xúc cây-cây hay hạt giống; chỉ lây qua rầy mang virus.

Dịch tễ học:
- Nguồn bệnh: Lúa chét, lúa hoang dại và một số cỏ dại (Leersia, Commelina) là ổ chứa virus.
- Gieo cấy không đồng loạt (asynchronous planting) tạo điều kiện cho rầy di chuyển liên tục giữa các lứa lúa khác tuổi, làm bùng phát dịch nhanh.
- Không có biện pháp chữa trị khi cây đã nhiễm bệnh — phòng ngừa là chủ đạo.

Biện pháp phòng trị (theo khuyến cáo của IRRI và Azzam & Chancellor, 2002):
1. Trồng giống kháng rầy và/hoặc kháng virus — biện pháp bền vững và kinh tế nhất.
2. Gieo cấy đồng loạt trong cùng vùng (synchronous planting) để cắt đứt vòng lây lan của rầy.
3. Để ruộng trống (fallow) ít nhất 30 ngày sau thu hoạch để giảm mật độ rầy và nguồn bệnh.
4. Vệ sinh đồng ruộng: cày vùi gốc rạ, tiêu diệt lúa chét và cỏ ký chủ ngay sau thu hoạch.
5. Điều chỉnh lịch gieo sạ né cao điểm rầy theo dự báo của cơ quan bảo vệ thực vật địa phương.
6. Thuốc trừ sâu có thể kiểm soát mật độ rầy nhưng kém hiệu quả do rầy truyền bệnh rất nhanh; cần kết hợp với biện pháp canh tác cộng đồng.""",
            metadata={
                "disease": "Tungro",
                "name_vn": "Vàng lùn",
                "pathogen_latin": "RTBV (Badnavirus) + RTSV (Waikavirus)",
                "sources": "IRRI Rice Knowledge Bank; Azzam & Chancellor (2002) Plant Dis 86(2):88-100; Hibino (1996) Annu Rev Phytopathol 34:249-274"
            }
        ),
    ]

    # 3. Tạo ChromaDB
    print(f"Creating ChromaDB at: {settings.VECTOR_DB_PATH}")
    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=settings.VECTOR_DB_PATH
        )
        print("Vector database created and persisted successfully!")
        print(f"Total documents indexed: {len(documents)}")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Could not create Vector DB: {e}")
        if "403" in str(e):
            print("\n!!! PHÁT HIỆN LỖI PHÂN QUYỀN (403 PERMISSION DENIED) !!!")
            print("Vui lòng kiểm tra API Key tại: https://aistudio.google.com/app/apikey")
            print("Chạy script chẩn đoán: `python src/core/diagnose_api.py`")

if __name__ == "__main__":
    setup_knowledge_base()
