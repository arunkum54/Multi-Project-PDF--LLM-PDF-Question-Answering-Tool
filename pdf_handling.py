import os
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    Extract text from the uploaded PDF file.
    """
    if not os.path.exists(pdf_path):
        print(f"PDF file not found at {pdf_path}")
        return ""
    
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
    return text

def chunk_text(text, chunk_size=500):
    """
    Chunk the text into smaller parts for easier processing.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
