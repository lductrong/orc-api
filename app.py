from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prep_image(image_path):
    """Upload image to Gemini and return file reference"""
    try:
        sample_file = genai.upload_file(path=image_path, display_name="Uploaded Image")
        print(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")
        return sample_file
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None

def extract_text_from_image(image_file, prompt):
    """Extract text from image using Gemini"""
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        response = model.generate_content([image_file, prompt])
        return response.text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

@app.route('/extract-text', methods=['POST'])
def extract_text():
    """API endpoint for text extraction from images"""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # Check if file has a name
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    # Get custom prompt if provided
    prompt = request.form.get('prompt', "Extract text from image, if one sentence then leave one line")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        # Process the image
        uploaded_file = prep_image(temp_file_path)
        if not uploaded_file:
            return jsonify({"error": "Failed to upload image to Gemini"}), 500
        
        extracted_text = extract_text_from_image(uploaded_file, prompt)
        if not extracted_text:
            return jsonify({"error": "Failed to extract text from image"}), 500
        
        # Clean up
        os.unlink(temp_file_path)
        
        return jsonify({
            "status": "success",
            "extracted_text": extracted_text,
            "prompt_used": prompt
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()