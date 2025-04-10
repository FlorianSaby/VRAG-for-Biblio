# -*- coding: utf-8 -*-

"""
Created on Thu Nov 28 10:37:45 2024
@author: florian saby
@mail: flo.saby@hotmail.fr
also install
conda install -c conda-forge poppler
"""
# Import libraries
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from flask import Flask, request, render_template, jsonify, session, url_for, send_from_directory
from flask_session import Session
from werkzeug.utils import secure_filename
import os
from rag_utils import send_request,load_database, rag


# Initialize the Flask app
app = Flask(__name__)
# Configuration
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

device_llm = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_llm = 'cpu'
# Load the model and tokenizer
model_name="Qwen/Qwen2-VL-2B-Instruct"
llm = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map=device_llm).eval()


min_pixels = 3 * 256 * 256
max_pixels = 3 * 512 * 512
processor = AutoProcessor.from_pretrained(model_name,min_pixels=min_pixels, max_pixels=max_pixels)


model_rag=load_database("Myxo_db")
nb_file_to_retrieve=1

max_length=2000
temperature=0.9
previous_chat=None

# Dummy function to simulate vision-language model processing
def process_vision_language_model(image_path, text_query,mode):
    global previous_chat    
    if mode=="chat":

        print(mode)
        image_lst = [image_path] if image_path is not None else []

        session.clear() 
        output,previous_chat=send_request(text_query,model_llm=llm, processor=processor,image_lst=image_lst,previous_chat=previous_chat)
        return {
            "image_description": "",
            "response_to_query": output
        }
    elif mode=="rag":
        output,previous_chat,pdf_lst=rag(model_rag,text_query,llm,processor,previous_chat=previous_chat,nb_file_to_retrieve=nb_file_to_retrieve)
        return {
            "image_description": pdf_lst,
            "response_to_query": output
        }
        

@app.route('/')
def index():
    # Ensure session is cleared at the start of each request.
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    response = {}
    message = request.form.get('message')
    image = request.files.get('image')
    mode = request.form.get('mode')

    # Handle image upload
    if image:
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        session['image_path'] = image_path  # Store image path in session
        response['image_url'] = url_for('uploaded_file', filename=filename)

    # Process the message and image
    result = process_vision_language_model(session.get('image_path'), message,mode)
    response['user_message'] = message
    response['bot_response'] = result['response_to_query']
    response['image_description'] = result['image_description']
    
    return jsonify(response)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

