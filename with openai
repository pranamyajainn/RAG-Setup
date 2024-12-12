import os
import json
import openai
import pandas as pd
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from fpdf import FPDF
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# OpenAI API configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
PREVIEW_FOLDER = 'previews'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'xlsx', 'json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREVIEW_FOLDER'] = PREVIEW_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREVIEW_FOLDER, exist_ok=True)

# Initialize ChromaDB client with error handling
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    chroma_client = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def chunk_text(text, chunk_size=1000):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def store_in_chromadb(collection_name, data):
    if chroma_client is None:
        raise ValueError("ChromaDB client not initialized")

    try:
        collection = chroma_client.get_or_create_collection(name=collection_name)
        collection.add(
            documents=data["documents"],
            metadatas=data["metadatas"],
            ids=data["ids"]
        )
    except Exception as e:
        print(f"Error storing in ChromaDB: {e}")
        raise


def query_chromadb(collection_name, query_text, top_k=5):
    if chroma_client is None:
        raise ValueError("ChromaDB client not initialized")

    try:
        collection = chroma_client.get_collection(name=collection_name)
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        return results
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        raise


def process_files(files):
    combined_data = {"documents": [], "metadatas": [], "ids": []}
    all_text = ""

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            file_extension = filename.rsplit('.', 1)[1].lower()
            try:
                if file_extension == "txt":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                elif file_extension == "pdf":
                    doc = fitz.open(file_path)
                    content = "".join(page.get_text() for page in doc)
                elif file_extension == "csv":
                    df = pd.read_csv(file_path)
                    content = df.to_string(index=False)
                elif file_extension == "xlsx":
                    df = pd.read_excel(file_path)
                    content = df.to_string(index=False)
                elif file_extension == "json":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.dumps(json.load(f), indent=4)
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")

                combined_data["documents"].append(content)
                combined_data["metadatas"].append({"filename": filename})
                combined_data["ids"].append(filename)

                all_text += content + "\n"

            except Exception as e:
                return f"Error processing file {filename}: {str(e)}", False

    # Store in ChromaDB
    try:
        store_in_chromadb("file_storage", combined_data)
    except Exception as e:
        return f"Error storing files in ChromaDB: {str(e)}", False

    return all_text, True


def use_openai_chat_api(prompt, model="gpt-3.5-turbo", max_tokens=1000):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that provides concise and accurate responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error occurred while querying OpenAI API: {str(e)}"

def sanitize_text(text):
    return text.encode('latin-1', 'replace').decode('latin-1')

def generate_nasdaq_pdf(filename, title, sections):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=title, ln=True, align="C")
    for section in sections:
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, txt=section["heading"], ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 10, txt=section["content"])
    pdf.output(filename)


def generate_pdf(output_filename, style, title, sections):
    if style == "nasdaq":
        generate_nasdaq_pdf(output_filename, title, sections)
    else:
        raise ValueError(f"Unknown style: {style}")


@app.route('/generate', methods=['POST'])
def generate_report():
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('files')
    prompt = request.form.get('prompt', '')
    style = request.form.get('style', 'nasdaq')  # Default to "nasdaq" if no style provided

    # Process files to extract content
    file_content, success = process_files(files)
    if not success:
        return jsonify({"error": file_content}), 400

    # Query ChromaDB
    try:
        chroma_results = query_chromadb("file_storage", prompt)
        matched_documents = "\n".join(chroma_results["documents"][0])
    except Exception as e:
        return jsonify({"error": f"Error querying ChromaDB: {str(e)}"}), 500

    # Query OpenAI API
    try:
        openai_response = use_openai_chat_api(matched_documents)
    except Exception as e:
        return jsonify({"error": f"Error querying OpenAI API: {str(e)}"}), 500

    # Generate PDF
    try:
        output_filename = os.path.join(app.config['PREVIEW_FOLDER'], 'report.pdf')
        sections = [
            {"heading": "Prompt", "content": prompt},
            {"heading": "Matched Documents", "content": matched_documents},
            {"heading": "OpenAI Response", "content": openai_response},
        ]
        generate_pdf(output_filename, style, "Generated Report", sections)
        return send_file(output_filename, as_attachment=True, download_name="report.pdf")
    except Exception as e:
        return jsonify({"error": f"Error generating report: {str(e)}"}), 500


@app.route('/preview', methods=['POST'])
def preview_report():
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('files')
    prompt = request.form.get('prompt', '')
    style = request.form.get('style', 'nasdaq')

    # Process files to extract content
    file_content, success = process_files(files)
    if not success:
        return jsonify({"error": file_content}), 400

    # Query ChromaDB
    try:
        chroma_results = query_chromadb("file_storage", prompt)
        matched_documents = "\n".join(chroma_results["documents"][0])
    except Exception as e:
        return jsonify({"error": f"Error querying ChromaDB: {str(e)}"}), 500

    # Query OpenAI API
    try:
        openai_response = use_openai_chat_api(matched_documents)
    except Exception as e:
        return jsonify({"error": f"Error querying OpenAI API: {str(e)}"}), 500

    # Generate PDF
    try:
        output_filename = os.path.join(app.config['PREVIEW_FOLDER'], 'preview.pdf')
        sections = [
            {"heading": "Prompt", "content": prompt},
            {"heading": "Matched Documents", "content": matched_documents},
            {"heading": "OpenAI Response", "content": openai_response},
        ]
        generate_pdf(output_filename, style, "Preview Report", sections)
        preview_url = f"/previews/preview.pdf"
        return jsonify({"preview_url": preview_url})
    except Exception as e:
        return jsonify({"error": f"Error generating preview: {str(e)}"}), 500

@app.route('/preview', methods=['POST'])
def preview_report():
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('files')
    prompt = request.form.get('prompt', '')
    style = request.form.get('style', 'nasdaq')

    # Process files to extract content
    file_content, success = process_files(files)
    if not success:
        return jsonify({"error": file_content}), 400

    # Query ChromaDB
    try:
        chroma_results = query_chromadb("file_storage", prompt)
        matched_documents = "\n".join(chroma_results["documents"][0])
    except Exception as e:
        return jsonify({"error": f"Error querying ChromaDB: {str(e)}"}), 500

    # Query OpenAI API
    try:
        openai_response = use_openai_chat_api(matched_documents)
    except Exception as e:
        return jsonify({"error": f"Error querying OpenAI API: {str(e)}"}), 500

    # Generate PDF
    try:
        output_filename = os.path.join(app.config['PREVIEW_FOLDER'], 'preview.pdf')
        sections = [
            {"heading": "Prompt", "content": prompt},
            {"heading": "Matched Documents", "content": matched_documents},
            {"heading": "OpenAI Response", "content": openai_response},
        ]
        generate_pdf(output_filename, style, "Preview Report", sections)
        preview_url = f"/previews/preview.pdf"  # Path relative to Flask server
        return jsonify({"preview_url": preview_url})
    except Exception as e:
        return jsonify({"error": f"Error generating preview: {str(e)}"}), 500
@app.route('/')
def index():
    return render_template('upload.html')


if __name__ == "__main__":
    app.run(debug=True)
