import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import os
from pdf_handling import chunk_text, extract_text_from_pdf

# Load Flan-T5-XL model and tokenizer
flan_t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
flan_t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flan_t5_model.to(device)

def generate_response(question, context):
    """
    Generate a response using Flan-T5-XL based on the provided question and context.
    """
    # Refined prompt for better focus
    input_text = f"Answer the question based on the passage: {context} Question: {question}"
    
    # Tokenize input
    inputs = flan_t5_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=600)

    # Perform inference without tracking gradients
    with torch.no_grad():
        outputs = flan_t5_model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_length=150,          
            temperature=0.2,         
            top_k=20,                
            top_p=0.8,               
            no_repeat_ngram_size=2,  
            do_sample=True          
        )

    # Decode and clean the output
    answer = flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer

def ask_flan_t5_xl(question, context):
    """
    Generate an answer to a question based on the provided context using Flan-T5-XL.
    """
    try:
        # Log context and question for debugging
        print(f"Question: {question}")
        print(f"Context:\n{context[:500]}...")
        
        # Generate a response
        answer = generate_response(question, context)
        print(f"Answer: {answer}")  # Debugging
        return answer
    except Exception as e:
        print(f"Error generating answer with Flan-T5-XL: {e}")
        return "Sorry, I couldn't generate an answer."

def ask_flan_t5_xl_from_pdf(project_name, pdf_filename, question):
    """
    Extract text from the PDF located in a specific project folder, chunk it, and generate a response using Flan-T5-XL.
    Assumes the PDF is located in the following structure: 'projects/{project_name}/{pdf_filename}'.
    """
    # Construct the project directory path
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
        
        # Generate a response using Flan-T5-XL based on the chunks
        return ask_flan_t5_xl(question, chunks)
    else:
        print(f"PDF file not found at {pdf_path}")
        return "PDF file not found."
