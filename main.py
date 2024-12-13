import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from threading import Thread
from project_management import create_project, list_projects, delete_project
from pdf_handling import extract_text_from_pdf, chunk_text
import shutil
from models.flan_XL import ask_flan_t5_xl_from_pdf
from sentence_transformers import SentenceTransformer, util

PROJECTS_DIR = os.path.join(os.getcwd(), "projects")

class PDFApp:
    def __init__(self, root):
        self.current_project = None
        self.root = root
        self.root.title("PDF Question Answering App")
        self.root.geometry("700x500")

        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Header
        tk.Label(self.root, text="PDF Question-Answering System", font=("Helvetica", 16)).pack(pady=10)

        tk.Button(self.root, text="Create Project", command=self.create_project_gui).pack(pady=5)
        tk.Button(self.root, text="List Projects", command=self.list_projects_gui).pack(pady=5)
        tk.Button(self.root, text="Delete Project", command=self.delete_project_gui).pack(pady=5)
        tk.Button(self.root, text="Switch Project", command=self.switch_project_gui).pack(pady=5)
        tk.Button(self.root, text="Ask Question", command=self.ask_question_gui).pack(pady=5)

        # Response Box
        tk.Label(self.root, text="Response Box", font=("Helvetica", 12)).pack(pady=5)
        self.response_box = tk.Text(self.root, wrap=tk.WORD, height=10, width=80, font=("Helvetica", 10))
        self.response_box.pack(pady=5)
        self.response_box.config(state=tk.DISABLED)  

        # Status Label
        self.status_label = tk.Label(self.root, text="Select an option to proceed.", font=("Helvetica", 10))
        self.status_label.pack(pady=10)

    def update_response_box(self, response):
        """
        Update the response box with the provided text.
        """
        self.response_box.config(state=tk.NORMAL)  
        self.response_box.delete(1.0, tk.END)  # Clear previous content
        self.response_box.insert(tk.END, response)  # Insert new response
        self.response_box.config(state=tk.DISABLED)  # Set back to read-only

    def update_status(self, message):
        """
        Update the status label with the provided message.
        """
        self.status_label.config(text=message)

    def create_project_gui(self):
        project_name = simpledialog.askstring("Create Project", "Enter the name of the project:")
        if project_name:
            pdf_path = filedialog.askopenfilename(title="Select PDF to Upload", filetypes=[("PDF files", "*.pdf")])
            if pdf_path:
                Thread(target=self.threaded_create_project, args=(project_name, pdf_path)).start()
            else:
                messagebox.showerror("Error", "No PDF selected. Project creation aborted.")
        else:
            messagebox.showerror("Error", "Project name cannot be empty.")

    def threaded_create_project(self, project_name, pdf_path):
        self.update_status("Creating project and uploading PDF...")
        os.makedirs(os.path.join(PROJECTS_DIR, project_name), exist_ok=True)
        destination = os.path.join(PROJECTS_DIR, project_name, os.path.basename(pdf_path))
        try:
            shutil.copy(pdf_path, destination)
            create_project(project_name)
            self.update_status("Project created successfully.")
            messagebox.showinfo("Success", f"Project '{project_name}' created and PDF uploaded.")
        except Exception as e:
            self.update_status("Error during project creation.")
            messagebox.showerror("Error", f"Failed to create project or upload PDF: {e}")

    # List Projects
    def list_projects_gui(self):
        projects = list_projects()
        if projects:
            messagebox.showinfo("Projects", "\n".join(projects))
        else:
            messagebox.showinfo("No Projects", "No projects are currently available.")

    # Delete Project
    def delete_project_gui(self):
        project_name = simpledialog.askstring("Delete Project", "Enter the name of the project to delete:")
        if project_name:
            confirm = messagebox.askyesno("Confirm", f"Are you sure you want to delete '{project_name}'?")
            if confirm:
                delete_project(project_name)
                messagebox.showinfo("Success", f"Project '{project_name}' deleted.")
            else:
                messagebox.showinfo("Cancelled", "Project deletion cancelled.")

    # Switch Project
    def switch_project_gui(self):
        project_name = simpledialog.askstring("Switch Project", "Enter the name of the project to switch to:")
        if project_name and os.path.exists(os.path.join(PROJECTS_DIR, project_name)):
            self.current_project = project_name
            messagebox.showinfo("Switched", f"Switched to project '{self.current_project}'.")
        else:
            messagebox.showerror("Error", f"Project '{project_name}' does not exist.")

    # Ask Question
    def ask_question_gui(self):
        if not self.current_project:
            messagebox.showwarning("No Project", "Please switch to a project first.")
            return

        question = simpledialog.askstring("Ask Question", "Enter your question:")
        if question:
            Thread(target=self.threaded_ask_question, args=(self.current_project, question)).start()

    def threaded_ask_question(self, project_name, question):
        self.update_status("Processing your question...")
        project_dir = os.path.join(PROJECTS_DIR, project_name)
        pdf_texts = []
        pdf_found = False

        print(f"Looking for PDFs in: {project_dir}")  

        # Search for the PDF in the project folder
        for file in os.listdir(project_dir):
            print(f"Checking file: {file}")  
            if file.endswith(".pdf"):
                pdf_path = os.path.join(project_dir, file)
                pdf_text = extract_text_from_pdf(pdf_path)  
                pdf_texts.append(pdf_text)
                pdf_found = True
                print(f"Extracted text from {file}: {pdf_text[:300]}...")  

        if not pdf_found:
            self.update_response_box("No PDFs found in the selected project.")
            self.update_status("No PDFs found.")
            return

        combined_text = " ".join(pdf_texts)
        chunks = chunk_text(combined_text) 

        print(f"Searching for question: {question}")  

        # Convert question and chunks to embeddings
        question_embedding = self.sentence_model.encode(question, convert_to_tensor=True)
        chunk_embeddings = self.sentence_model.encode(chunks, convert_to_tensor=True)

        # Compute similarities
        similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings)
        best_chunk_idx = similarities.argmax().item()
        best_chunk = chunks[best_chunk_idx]

        print(f"Best matching chunk: {best_chunk[:300]}...") 
        answer = ask_flan_t5_xl_from_pdf(project_name, file, question) 
        self.update_response_box(answer)
        self.update_status("Response generated successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFApp(root)
    root.mainloop()
