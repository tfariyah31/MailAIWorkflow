import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util

def extract_resume_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return " ".join(page.get_text() for page in doc)
