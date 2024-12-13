import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import os
from pdf_handling import chunk_text, extract_text_from_pdf

# Load GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token 

# Load SentenceTransformer model (pre-trained model from Hugging Face's hub)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model.to(device)
sentence_model.to(device)

def find_best_chunk(question, chunks):
    """
    Find the most relevant chunk for a given question using cosine similarity.
    """
    question_embedding = sentence_model.encode(question, convert_to_tensor=True, device=device)
    chunk_embeddings = sentence_model.encode(chunks, convert_to_tensor=True, device=device)
    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings)
    best_chunk_idx = similarities.argmax().item()
    return chunks[best_chunk_idx]

def generate_response(question, context):
    """
    Generate a response using GPT-2 based on the provided question and context.
    """
    input_text = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    
    inputs = gpt2_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=500 
    )

    with torch.no_grad():
        outputs = gpt2_model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_length=400,  
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.5,  
            do_sample=True,
            pad_token_id=gpt2_tokenizer.pad_token_id,
        )

    answer = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer.split("Answer:")[-1].strip()

def ask_gpt2(question, context):
    """
    Generate an answer to a question based on the provided context using GPT-2.
    """
    try:
        best_chunk = find_best_chunk(question, context)
        answer = generate_response(question, best_chunk)
        return answer
    except Exception as e:
        print(f"Error generating answer with GPT-2: {e}")
        return "Sorry, I couldn't generate an answer."

def ask_neo_from_pdf(project_name, pdf_filename, question):
    """
    Extract text from the PDF located in a specific project folder, chunk it, and generate a response using GPT-2.
    Assumes the PDF is located in the following structure: 'projects/{project_name}/{pdf_filename}'.
    """
    project_dir = os.path.join("projects", project_name)
    
    print(f"Looking for PDFs in: {project_dir}")  
    
    # Check if the project directory exists
    if not os.path.exists(project_dir):
        print(f"Project directory not found: {project_dir}")
        return "Project directory not found."
    pdf_path = os.path.join(project_dir, pdf_filename)
    print(f"Checking if PDF exists at {pdf_path}: {os.path.exists(pdf_path)}")

    if os.path.exists(pdf_path):
        # Extract text from the PDF
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        return ask_gpt2(question, chunks)
    else:
        print(f"PDF file not found at {pdf_path}")
        return "PDF file not found."
