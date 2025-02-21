import os
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from PyPDF2 import PdfReader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image as PILImage
import pytesseract

# Load environment 
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'xlsx', 'json', 'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Initialize Chroma and embedding model
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model
)

# Initialize HuggingFace LLM pipeline
LLM_MODEL_NAME = "EleutherAI/gpt-neo"  # Replace with your desired HuggingFace model
llm_pipeline = pipeline("text-generation", model=LLM_MODEL_NAME, max_length=512, temperature=0.5)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_image(file_path):
    try:
        image = PILImage.open(file_path)
        text = pytesseract.image_to_string(image)
        return text.strip() or "No text could be extracted from the image."
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        return ''.join(page.extract_text() for page in reader.pages)
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def chunk_text_and_store(text, source_name):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    if not chunks:
        print(f"Warning: No chunks generated from {source_name}.")
        return []

    documents = [{"content": chunk, "metadata": {"source": source_name}} for chunk in chunks]
    try:
        vector_store.add_texts(
            texts=[doc["content"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents]
        )
    except Exception as e:
        print(f"Error adding chunks to Chroma for {source_name}: {e}")
    return chunks

def retrieve_relevant_chunks(prompt):
    try:
        response = qa_chain.run(prompt)
        return response
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return "No relevant information could be retrieved."

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import os
import matplotlib.pyplot as plt

def generate_detailed_pdf(prompt, combined_responses, file_summaries, df):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    import os
    import matplotlib.pyplot as plt

    pdf_path = os.path.join(STATIC_FOLDER, 'detailed_report.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=18,
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=11,
        leading=14,
        spaceAfter=8
    )
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])

    elements = []

    # Title
    elements.append(Paragraph("AI-Generated Research Paper Report", title_style))
    elements.append(Spacer(1, 12))

    # Introduction
    elements.append(Paragraph("Introduction", heading_style))
    elements.append(Paragraph(
        f"This report was generated based on the prompt: '{prompt}'. It provides a comprehensive overview of key findings, combining insights from multiple sources and highlighting relevant case studies and examples.",
        body_style
    ))
    elements.append(Spacer(1, 12))

    # Literature Review
    elements.append(Paragraph("Literature Review", heading_style))
    for summary in file_summaries:
        elements.append(Paragraph(summary, body_style))
    elements.append(Spacer(1, 12))

    # Detailed Findings
    elements.append(Paragraph("Detailed Findings", heading_style))
    for response in combined_responses:
        elements.append(Paragraph(response, body_style))
        elements.append(Spacer(1, 8))

    # Discussion
    elements.append(Paragraph("Discussion", heading_style))
    elements.append(Paragraph(
        "This section critically analyzes the findings, explores challenges in the domain, and identifies potential future directions to advance the field.",
        body_style
    ))
    elements.append(Spacer(1, 12))

    # Conclusion
    elements.append(Paragraph("Conclusion", heading_style))
    elements.append(Paragraph(
        "Based on the insights provided, this report recommends actionable steps and highlights key implications for researchers and practitioners.",
        body_style
    ))
    elements.append(Spacer(1, 12))

    # Data Visualization
    if not df.empty:
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            try:
                plot_path = os.path.join(STATIC_FOLDER, 'plot.png')
                numeric_df.plot(kind='bar', figsize=(8, 6))
                plt.title("Bar Plot of Numeric Data")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()

                elements.append(Paragraph("Data Visualization", heading_style))
                elements.append(Paragraph("Bar Plot of Numeric Data:", body_style))
                elements.append(ReportLabImage(plot_path, width=400, height=300))
                elements.append(Spacer(1, 12))
            except Exception as e:
                print(f"Error creating bar plot: {e}")

    # Include Tables (if applicable)
    if not df.empty:
        try:
            table_data = [list(df.columns)] + df.values.tolist()
            table = Table(table_data)
            table.setStyle(table_style)
            elements.append(Paragraph("Tabular Summary", heading_style))
            elements.append(table)
            elements.append(Spacer(1, 12))
        except Exception as e:
            print(f"Error creating table: {e}")

    doc.build(elements)
    return pdf_path

@app.route('/preview', methods=['POST'])
def preview_report():
    if 'files' not in request.files or len(request.files.getlist('files')) == 0:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('files')
    prompt = request.form.get('prompt', '')
    combined_responses = []
    file_summaries = []
    dfs = []

    for file in files:
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            file_extension = filename.rsplit(".", 1)[1].lower()
            file_content = ""

            try:
                if file_extension in {'jpg', 'jpeg', 'png'}:
                    file_content = extract_text_from_image(file_path)
                elif file_extension == "pdf":
                    file_content = extract_text_from_pdf(file_path)
                elif file_extension in {"csv", "xlsx"}:
                    df = pd.read_csv(file_path) if file_extension == "csv" else pd.read_excel(file_path)
                    dfs.append(df)
                    file_content = df.to_string(index=False)
                elif file_extension == "json":
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = json.dumps(json.load(f), indent=4)
                elif file_extension == "txt":
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()

                file_summary = f"<b>{filename}:</b> {file_content[:500]}..."
                file_summaries.append(file_summary)

                chunks = chunk_text_and_store(file_content, filename)
                response = retrieve_relevant_chunks(prompt)
                combined_responses.append(f"--- Insights from {filename} ---\n{response}")

            except Exception as e:
                return jsonify({"error": f"Error processing file {filename}: {str(e)}"}), 400

    if not combined_responses:
        return jsonify({"error": "The uploaded files contain no readable content."}), 400

    try:
        combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        pdf_path = generate_detailed_pdf(prompt, combined_responses, file_summaries, combined_df)
        return jsonify({"preview_url": f"/static/{os.path.basename(pdf_path)}"})

    except Exception as e:
        return jsonify({"error": f"Error generating PDF: {str(e)}"}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(os.path.join(STATIC_FOLDER, filename))

@app.route("/")
def index():
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
