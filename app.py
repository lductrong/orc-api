from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import tempfile
from werkzeug.utils import secure_filename
import imghdr
from flask_cors import CORS 
import re

app = Flask(__name__)

CORS(app)  # Cho phép tất cả domains

# Security Configuration
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_MIME_TYPES = {'image/png', 'image/jpeg', 'image/webp'}

# Gemini Configuration
API_KEY = "AIzaSyC21O0_WJ2Lzn-KSZu7OOBR3eJV9rDCaqQ"
if not API_KEY:
    raise RuntimeError("GEMINI_AI_API_KEY environment variable not set")
genai.configure(api_key=API_KEY)

# Prompt không thay đổi
PROMPT = """
Extract the text in the image and return in the following format:

1. Text: [the original text from the image]
2. Pronunciation: [Phonetic of the text]
3. Translation: [Translate into Vietnamese naturally, matching native Vietnamese style and context, avoiding word-for-word translation]

Only return these three items in plain text. Do not explain anything else.
"""

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file_stream):
    """Validate actual image type"""
    file_stream.seek(0)
    header = file_stream.read(32)
    file_stream.seek(0)
    
    ext = imghdr.what(None, header)
    return ext in ALLOWED_EXTENSIONS

def prep_image(image_path):
    """Upload image to Gemini"""
    try:
        mime_type = f"image/{imghdr.what(image_path)}"
        if mime_type not in ALLOWED_MIME_TYPES:
            raise ValueError("Unsupported image type")

        return genai.upload_file(
            path=image_path,
            display_name=secure_filename(image_path),
            mime_type=mime_type
        )
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        return None

def parse_gemini_response(response_text):
    """Parse Gemini response with robust handling of different formats"""
    
    # Chuẩn hóa đầu vào: loại bỏ khoảng trắng thừa và ký tự đặc biệt
    normalized_text = response_text.strip()
    
    # Pattern 1: Định dạng chuẩn với số thứ tự rõ ràng (1. 2. 3.)
    pattern1 = re.compile(
        r"1\.\s*Text:\s*(.*?)(?=\n\s*2\.\s*Pronunciation:|\Z)" +
        r"\s*2\.\s*Pronunciation:\s*(.*?)(?=\n\s*3\.\s*Translation:|\Z)" +
        r"\s*3\.\s*Translation:\s*(.*)",
        re.DOTALL | re.IGNORECASE
    )
    
    # Pattern 2: Định dạng không có số thứ tự, chỉ có tiêu đề
    pattern2 = re.compile(
        r"Text:\s*(.*?)(?=\n\s*Pronunciation:|\Z)" +
        r"\s*Pronunciation:\s*(.*?)(?=\n\s*Translation:|\Z)" +
        r"\s*Translation:\s*(.*)",
        re.DOTALL | re.IGNORECASE
    )
    
    # Pattern 3: Chỉ có các dòng riêng biệt với số thứ tự
    pattern3 = re.compile(
        r"1\.\s*(.*?)\n\s*2\.\s*(.*?)\n\s*3\.\s*(.*)",
        re.DOTALL
    )
    
    # Thử các pattern theo thứ tự
    for pattern in [pattern1, pattern2, pattern3]:
        match = pattern.search(normalized_text)
        if match:
            groups = [g.strip() for g in match.groups()]
            # Kiểm tra xem có đủ 3 phần không
            if len(groups) >= 3 and all(groups[:3]):
                return {
                    "text": groups[0],
                    "pronunciation": groups[1],
                    "translation": groups[2]
                }
    
    # Fallback: Xử lý từng dòng nếu không match pattern nào
    lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    
    # Nếu có ít nhất 3 dòng
    if len(lines) >= 3:
        # Dòng đầu tiên là text
        text = lines[0]
        
        # Tìm pronunciation (có thể chứa ký tự phiên âm)
        pronunciation = ""
        translation_start = 1
        for i in range(1, len(lines)):
            if any(char in lines[i] for char in ["/", "[", "]", "ˈ", "ː"]):  # Các ký tự phiên âm phổ biến
                pronunciation = lines[i]
                translation_start = i + 1
                break
        
        # Phần còn lại là translation
        translation = " ".join(lines[translation_start:]) if translation_start < len(lines) else ""
        
        return {
            "text": text,
            "pronunciation": pronunciation,
            "translation": translation
        }
    
    # Nếu tất cả đều thất bại, trả về raw response và cảnh báo
    app.logger.warning(f"Could not parse response: {normalized_text}")
    return {
        "text": normalized_text,
        "pronunciation": "",
        "translation": ""
    }
@app.route('/extract-text', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    
    # Validate file
    if not (file and allowed_file(file.filename)):
        return jsonify({"error": "Invalid file type"}), 400
        
    if not validate_image(file.stream):
        return jsonify({"error": "Invalid image content"}), 400

    try:
        # Lưu file tạm
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        uploaded_file = prep_image(tmp_path)
        if not uploaded_file:
            return jsonify({"error": "Image upload failed"}), 500
            
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = model.generate_content([uploaded_file, PROMPT])
        
        # Phân tích kết quả
        response_data = parse_gemini_response(response.text)
        return jsonify({
            "status": "success",
            "data": response_data
        })

    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}")
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    app.run()