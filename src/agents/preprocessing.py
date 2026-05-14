from PIL import Image, ImageEnhance
import io

class PreprocessingAgent:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def process(self, image: Image.Image) -> Image.Image:
        """
        Thực hiện tiền xử lý ảnh:
        1. Chuyển về RGB (đảm bảo định dạng chuẩn)
        2. Điều chỉnh kích thước về 224x224 (phù hợp với SigLIP2)
        3. Tăng nhẹ độ tương phản để làm nổi rõ vết bệnh

        Lưu ý: KHÔNG tách nền. Mô hình SigLIP2 được huấn luyện trên
        ảnh lá tự nhiên có nền. Việc tách nền và thay bằng màu trắng
        tạo ra phân phối đầu vào khác với dữ liệu huấn luyện,
        dẫn đến kết quả sai (thường phân loại thành Healthy).
        """
        # 1. Đảm bảo ảnh là RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 2. Tăng nhẹ độ tương phản (10%) để làm nổi vết bệnh
        enhancer = ImageEnhance.Contrast(image)
        image_enhanced = enhancer.enhance(1.1)

        # 3. Resize về 224x224 (kích thước đầu vào của SigLIP2-base-patch16-224)
        image_resized = image_enhanced.resize(self.target_size, Image.Resampling.LANCZOS)

        return image_resized

    def process_from_bytes(self, image_bytes: bytes) -> Image.Image:
        image = Image.open(io.BytesIO(image_bytes))
        return self.process(image)
