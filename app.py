from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import tempfile
from werkzeug.utils import secure_filename
import imghdr

app = Flask(__name__)

# Security Configuration
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_MIME_TYPES = {'image/png', 'image/jpeg', 'image/webp'}

# Gemini Configuration
API_KEY = os.getenv("GEMINI_AI_API_KEY")
print(API_KEY)
if not API_KEY:
    raise RuntimeError("GEMINI_AI_API_KEY environment variable not set")
genai.configure(api_key=API_KEY)

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
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        uploaded_file = prep_image(tmp_path)
        if not uploaded_file:
            return jsonify({"error": "Image upload failed"}), 500
            
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content([
            uploaded_file,
            request.form.get('prompt', "Extract text from image, return only the text in the image, if one sentence then leave one line")
        ])
        
        return jsonify({
            "status": "success",
            "text": response.text
        })
        
    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}")
        return jsonify({"error": "Processing failed"}), 500
        
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    app.run()#(host="0.0.0.0", port=5000)