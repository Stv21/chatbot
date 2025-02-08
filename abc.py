import os
import subprocess
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

# ✅ Set USER_AGENT (to avoid warnings)
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# ✅ Configure Gemini API (Optional)
genai.configure(api_key="AIzaSyBhnHtqNPqtyLpXL4cfKr4uAhS7U3DqJZg")

# ✅ Load data from website
url = "https://brainlox.com/courses/category/technical"
url_loader = WebBaseLoader(url)
documents = url_loader.load()

# ✅ Extract plain text from HTML
soup = BeautifulSoup(documents[0].page_content, "html.parser")
clean_text = soup.get_text(separator="\n", strip=True)

# ✅ Generate embeddings using a free model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Free model

# Create a custom embedding class
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts)

    def embed_query(self, text):
        return self.model.encode([text])[0]

# Wrap the SentenceTransformer in the custom embedding class
embeddings = SentenceTransformerEmbeddings(embedding_model)

# ✅ Store embeddings in FAISS (Fixed)
vector_store = FAISS.from_texts([clean_text], embeddings)  # ✅ Corrected FAISS method
vector_store.save_local("faiss_index")

# ✅ Process text using **Mistral (Ollama)**
def ask_mistral(text):
    try:
        response = subprocess.run(
            ["ollama", "run", "mistral", text], capture_output=True, text=True
        )
        return response.stdout.strip()
    except Exception as e:
        return f"⚠️ Mistral Error: {e}"

mistral_summary = ask_mistral(f"Summarize this course information:\n\n{clean_text}")

# ✅ Use Gemini (Optional)
def ask_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text if response.text else "⚠️ Gemini failed"
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"

gemini_response = ask_gemini(f"Refine and improve this summary:\n\n{mistral_summary}")

print("\n🔹 Final AI-Generated Course Summary:\n", gemini_response)