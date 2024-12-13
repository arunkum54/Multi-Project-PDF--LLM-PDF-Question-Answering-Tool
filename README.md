# Multi-Project-PDF--LLM-PDF-Question-Answering-Tool

Overview
This project is a web application built as part of the AI/ML Skill Evaluation task for Envint Global LLP. The tool enables users to create multiple projects, upload PDFs, and use a Large Language Model (LLM) to answer questions based on the uploaded PDFs.

Features
Core Features
Project Management: Create, list, and delete projects.
PDF Upload: Upload multiple PDFs to each project.
Question Answering: Ask questions and receive answers based on the PDFs within a specific project.
Project Switching: Switch between projects without re-uploading PDFs.
User-Friendly Interface: A simple GUI for non-technical users.
Python-based web application.
Command-line interface for project management and question answering.

Installation
Clone the repository:
git clone https://github.com/arunkum54/Multi-Project-PDF--LLM-PDF-Question-Answering-Tool.git  
cd LLM-PDF-Question-Answering-Tool 

Install dependencies:
pip install -r requirements.txt  

Run the application:
python main.py  

Usage
1. Create a New Project
Enter the project name to create a new project.

2. Upload PDFs
Upload one or more PDFs to the project.

3. Ask Questions
Enter your questions, and the LLM will respond based on the PDFs uploaded to the active project.

4. Switch Projects
Switch between projects to ask questions from a different set of PDFs.

Example Questions
What is the main goal of this tool?
What are the core functionalities required?
What additional features are considered good to have?

Dependencies
Python 3.8+
transformers
PyPDF2
torch (for GPU support)

Author
Arun Kumar Rana
[linkedin.com/in/arun738267/](https://www.linkedin.com/in/arun738267/)
