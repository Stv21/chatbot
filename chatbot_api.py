from flask import Flask, request, jsonify, render_template
from flask_restful import Api, Resource
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai
import os

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

# Set up embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Configure Gemini API
genai.configure(api_key="AIzaSyBhnHtqNPqtyLpXL4cfKr4uAhS7U3DqJZg")

# Path for FAISS index
VECTOR_STORE_PATH = "faiss_index"

# Load or create FAISS index
vector_db = None

def load_or_create_faiss():
    global vector_db
    if os.path.exists(VECTOR_STORE_PATH):
        print("🔍 Loading existing FAISS index...")
        try:
            vector_db = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"❌ Error loading FAISS: {e}")
            print("⚠️ Deleting corrupted index and rebuilding...")
            os.system("rm -rf faiss_index")  # For Linux/macOS
            os.system("del /s /q faiss_index")  # For Windows
            create_faiss_index()
    else:
        create_faiss_index()

def create_faiss_index():
    global vector_db
    print("🚀 Scraping data & creating FAISS index...")

    try:
        # Step 1: Extract data from website
        loader = WebBaseLoader("https://brainlox.com/courses/category/technical")
        docs = loader.load()

        # Step 2: Split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(docs)

        # Step 3: Create embeddings & store in FAISS
        vector_db = FAISS.from_documents(texts, embedding_model)
        vector_db.save_local(VECTOR_STORE_PATH)
        print("✅ FAISS index created & saved!")
    except Exception as e:
        print(f"❌ Error creating FAISS: {e}")
        vector_db = None

# Call function to load or create FAISS index
load_or_create_faiss()

# Function to interact with Gemini API
def ask_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text if response.text else "⚠️ Gemini failed"
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"

# Format the response with clickable links
def format_response(context):
    lines = context.split("\n")
    formatted_lines = []
    for line in lines:
        if "•" in line or "*" in line:
            course_name = line.split(":")[-1].strip()
            formatted_line = f"<a href='https://brainlox.com/courses/category/technical' target='_blank'>{course_name}</a>"
            formatted_lines.append(formatted_line)
        else:
            formatted_lines.append(line)
    return "<br>".join(formatted_lines)

# API Endpoint for chatbot
class Chatbot(Resource):
    def post(self):
        if vector_db is None:
            return jsonify({"error": "FAISS index not available"}), 500
        
        data = request.get_json()
        query = data.get("query", "").lower()

        if not query:
            return jsonify({"error": "No query provided"}), 400

        if query in ["hi", "hello"]:
            return jsonify({"response": "Welcome to the Brainlox Chatbot! How can I assist you today?"})

        retriever = vector_db.as_retriever()
        results = retriever.get_relevant_documents(query)

        # Generate summary using Gemini API
        context = results[0].page_content if results else "Sorry, I don't have an answer for that."
        gemini_response = ask_gemini(f"Refine and improve this summary:\n\n{context}")

        # Format the response
        formatted_response = format_response(gemini_response)

        return jsonify({"response": formatted_response})

# Add API endpoint
api.add_resource(Chatbot, "/chat")

# Serve frontend
@app.route("/")
def home():
    return render_template("index.html")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
