from PIL import Image, ImageEnhance
from rembg import remove
import io

class PreprocessingAgent:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def process(self, image: Image.Image) -> Image.Image:
        """
        Thực hiện tiền xử lý ảnh:
        1. Tách nền (loại bỏ tay người, đất, cỏ dại)
        2. Điều chỉnh kích thước về 224x224
        3. Tăng cường độ tương phản (tuỳ chọn)
        """
        # 1. Tách nền (Background Removal)
        # remove() expects bytes or PIL Image. We pass PIL Image and get PIL Image back.
        image_no_bg = remove(image)
        
        # Chuyển đổi về RGB nếu ảnh kết quả là RGBA (trong suốt nền)
        if image_no_bg.mode in ('RGBA', 'LA') or (image_no_bg.mode == 'P' and 'transparency' in image_no_bg.info):
            background = Image.new('RGB', image_no_bg.size, (255, 255, 255))
            background.paste(image_no_bg, mask=image_no_bg.split()[3]) # 3 is the alpha channel
            image_no_bg = background
            
        # 2. Tăng cường độ tương phản tự động
        enhancer = ImageEnhance.Contrast(image_no_bg)
        image_enhanced = enhancer.enhance(1.2) # Tăng 20% tương phản
        
        # 3. Thay đổi kích thước
        image_resized = image_enhanced.resize(self.target_size, Image.Resampling.LANCZOS)
        
        return image_resized

    def process_from_bytes(self, image_bytes: bytes) -> Image.Image:
        image = Image.open(io.BytesIO(image_bytes))
        return self.process(image)
