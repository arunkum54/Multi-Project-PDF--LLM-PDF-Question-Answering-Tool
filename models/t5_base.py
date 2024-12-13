import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import os
from pdf_handling import chunk_text, extract_text_from_pdf

t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

sentence_model = SentenceTransformer("all-MiniLM-L6-v2") 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)
sentence_model.to(device)

VALIDATION_THRESHOLD = 0.1

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
    Generate a synthetic answer using T5 based on the provided question and context.
    """
    input_text = f"question: {question} context: {context} </s>"
    inputs = t5_tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = t5_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=200, 
            num_beams=5, 
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
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
    Generate a synthetic answer to a question based on the provided context using T5.
    This method finds the best chunks of text related to the question and validates the answer.
    """
    try:
        best_chunks, similarities = find_best_chunks(question, context_chunks, top_n=3)
        combined_context = " ".join(best_chunks)  
        answer = generate_answer(question, combined_context)
        similarity = validate_answer(answer, combined_context)
        if similarity >= VALIDATION_THRESHOLD:
            return f"Answer: {answer} (Validation Score: {similarity:.2f})"
        else:
            return f"The answer generated does not align well with the context. (Validation Score: {similarity:.2f})"
    except Exception as e:
        print(f"Error generating answer with T5: {e}")
        return "Sorry, I couldn't generate an answer."

def ask_gpt2_from_pdf(project_name, pdf_filename, question):
    """
    Extract text from the PDF located in a specific project folder, chunk it, and generate a response using T5.
    Assumes the PDF is located in the following structure: 'projects/{project_name}/{pdf_filename}'.
    """
    project_dir = os.path.join("projects", project_name)
    if not os.path.exists(project_dir):
        print(f"Project directory not found: {project_dir}")
        return "Project directory not found."
    pdf_path = os.path.join(project_dir, pdf_filename)

    if os.path.exists(pdf_path):
        text = extract_text_from_pdf(pdf_path)
        
        # Chunk the extracted text into manageable parts
        chunks = chunk_text(text)
        
        return ask_gpt2(question, chunks)
    else:
        print(f"PDF file not found at {pdf_path}")
        return "PDF file not found."
