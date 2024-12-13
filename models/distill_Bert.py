import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import os
from pdf_handling import chunk_text, extract_text_from_pdf

# Load DistilBERT model and tokenizer
distilbert_model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load SentenceTransformer for semantic search
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight model for semantic similarity

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distilbert_model.to(device)
sentence_model.to(device)

VALIDATION_THRESHOLD = 0.4  # Minimum similarity score for answer validation

def find_best_chunks(question, chunks, top_n=1):
    """
    Find the most relevant chunks for a given question using cosine similarity.
    """
    question_embedding = sentence_model.encode(question, convert_to_tensor=True, device=device)
    chunk_embeddings = sentence_model.encode(chunks, convert_to_tensor=True, device=device)
    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings).squeeze()
    top_chunk_indices = similarities.topk(top_n).indices.tolist()
    
    return [chunks[idx] for idx in top_chunk_indices], similarities[top_chunk_indices]

def generate_answer(question, context):
    """
    Generate an answer using DistilBERT based on the provided question and context.
    """
    inputs = distilbert_tokenizer(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = distilbert_model(**inputs)
    
    # Extract start and end positions for the answer
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    answer_tokens = inputs.input_ids[0][start_idx:end_idx]
    answer = distilbert_tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer

def validate_answer(answer, context):
    """
    Validate the generated answer by comparing it with the context using semantic similarity.
    """
    answer_embedding = sentence_model.encode(answer, convert_to_tensor=True, device=device)
    context_embedding = sentence_model.encode(context, convert_to_tensor=True, device=device)
    similarity = util.pytorch_cos_sim(answer_embedding, context_embedding).item()
    return similarity

def ask_gpt2(question, context_chunks):
    """
    Generate an answer to a question based on the provided context using DistilBERT.
    This method finds the best chunks of text related to the question and validates the answer.
    """
    try:
        # Find the best matching chunks
        best_chunks, similarities = find_best_chunks(question, context_chunks, top_n=3)
        combined_context = " ".join(best_chunks)  # Combine top chunks for a richer context
        
        # Generate a response based on the combined context using DistilBERT
        answer = generate_answer(question, combined_context)
        
        # Validate the generated answer
        similarity = validate_answer(answer, combined_context)
        if similarity >= VALIDATION_THRESHOLD:
            return f"Answer: {answer} (Validation Score: {similarity:.2f})"
        else:
            return f"The answer generated does not align well with the context. (Validation Score: {similarity:.2f})"
    except Exception as e:
        print(f"Error generating answer with DistilBERT: {e}")
        return "Sorry, I couldn't generate an answer."

def ask_gpt2_from_pdf(project_name, pdf_filename, question):
    """
    Extract text from the PDF located in a specific project folder, chunk it, and generate a response using DistilBERT.
    Assumes the PDF is located in the following structure: 'projects/{project_name}/{pdf_filename}'.
    """
    # Construct the project directory path
    project_dir = os.path.join("projects", project_name)
    
    # Check if the project directory exists
    if not os.path.exists(project_dir):
        print(f"Project directory not found: {project_dir}")
        return "Project directory not found."
    
    # Build the PDF path using the project folder and the PDF filename
    pdf_path = os.path.join(project_dir, pdf_filename)

    if os.path.exists(pdf_path):
        # Extract text from the PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Chunk the extracted text into manageable parts
        chunks = chunk_text(text)
        return ask_gpt2(question, chunks)
    else:
        print(f"PDF file not found at {pdf_path}")
        return "PDF file not found."
