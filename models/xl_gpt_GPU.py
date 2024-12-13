import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import os
from pdf_handling import chunk_text, extract_text_from_pdf

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token 


if not torch.backends.mps.is_available():
    raise RuntimeError("No GPU available. Please use a machine with GPU support.")

device = torch.device("mps")
gpt2_model.to(device)

def generate_response(question, context):
    """
    Generate a response using GPT-2 based on the provided question and context.
    """
    input_text = f"Based on the following passage:\n{context}\n\nQuestion: {question}\nAnswer (in one sentence):"
    
    inputs = gpt2_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=600)
    with torch.no_grad():
        outputs = gpt2_model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_length=400,           
            temperature=0.2,          # Reduce randomness
            top_k=20,                 
            top_p=0.8,                
            no_repeat_ngram_size=2,   # Avoid repeated phrases
            do_sample=True,           
            pad_token_id=gpt2_tokenizer.pad_token_id,
        )

    # Decode and clean the output
    answer = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer.split("Answer:")[-1].strip()

def ask_gpt2_large(question, context):
    """
    Generate an answer to a question based on the provided context using GPT-2 Large.
    """
    try:
        print(f"Question: {question}")
        print(f"Context:\n{context[:500]}...") 
        answer = generate_response(question, context)
        print(f"Answer: {answer}")  # Debugging
        return answer
    except Exception as e:
        print(f"Error generating answer with GPT-2: {e}")
        return "Sorry, I couldn't generate an answer."

def ask_gpt2_from_pdf(project_name, pdf_filename, question):
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
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        return ask_gpt2_large(question, chunks)
    else:
        print(f"PDF file not found at {pdf_path}")
        return "PDF file not found."