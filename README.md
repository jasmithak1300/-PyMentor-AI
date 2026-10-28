# PyMentor AI — An Intelligent Python Learning Assistant

## 1. Project Overview

PyMentor AI is an AI-powered Python learning platform designed to help students understand programming concepts, explore code, practise problem-solving, and learn from their study materials.

The application combines a web interface with a locally running language model to provide interactive programming assistance.

## 2. Problem Statement

Students learning Python often struggle to understand programming logic, debug code, and connect theoretical concepts with practical examples.

PyMentor AI aims to make learning more interactive by bringing explanations, practice exercises, quizzes, and study-note assistance together in one platform.

## 3. Objectives

- Explain Python concepts in beginner-friendly language.
- Generate AI-powered answers to programming questions.
- Explain Python code and its expected behaviour.
- Provide feedback on practice solutions.
- Test programming knowledge through quizzes.
- Retrieve relevant information from uploaded PDF study notes.

## 4. Features

### AI Python Tutor
Answers Python-related questions using the locally running Ollama language model.

### Python Learning Modules
Provides introductory examples covering variables, data types, conditions, loops, and functions.

### AI Code Explainer
Explains submitted Python code and identifies potential errors or improvements.

### Python Quiz Center
Provides multiple-choice questions and calculates the student's score.

### Python Practice Lab
Reviews solutions and provides AI-generated feedback.

### PDF Study Assistant
Extracts text from uploaded PDF notes, generates embeddings, retrieves relevant sections, and uses them as context for AI-generated answers.

## 5. Technology Stack

| Component | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| Language Model | Ollama, Llama 3.2 |
| Embedding Model | Nomic Embed Text |
| PDF Processing | pypdf |
| Vector Similarity | Cosine similarity |
| Optional future extension | ChromaDB |

## 6. System Architecture

1. A student interacts with the web interface.
2. JavaScript sends requests to Flask API endpoints.
3. Flask processes the request.
4. For AI features, Flask communicates with the local Ollama service.
5. For PDF questions, the application retrieves relevant text sections using embeddings.
6. The language model generates a response using the question and retrieved context.
7. Flask returns the response to the browser.

## 7. Installation

### Prerequisites

- Python
- pip
- Ollama
- The `llama3.2` model
- The `nomic-embed-text` model for PDF retrieval

### Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd PyMentor-AI
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Download the required models if needed

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Start the application

```bash
python app.py
```

Open the following address in your browser:

http://127.0.0.1:5000

## 8. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Display the application |
| GET | `/health` | Check application status |
| POST | `/api/tutor` | Answer Python questions |
| POST | `/api/explain` | Explain submitted code |
| POST | `/api/practice-check` | Review practice solutions |
| POST | `/api/notes/upload` | Upload and process PDF notes |
| POST | `/api/notes/ask` | Ask questions about uploaded notes |

## 9. Current Limitations

- Uploaded PDF embeddings are stored in memory and are not persistent.
- Only readable, text-based PDFs are supported; scanned PDFs require OCR.
- The application does not execute submitted Python code.
- AI responses may occasionally contain incorrect information and should be verified.
- The quiz currently contains a small fixed set of questions.
- ChromaDB is listed as an available extension but is not used in the current retrieval implementation.
- User accounts, persistent learning progress, and production deployment are not implemented.

## 10. Future Enhancements

- Persistent storage for uploaded documents and embeddings.
- ChromaDB-based vector retrieval.
- Student accounts and learning progress tracking.
- More Python exercises and adaptive quizzes.
- Secure code execution in an isolated environment.
- Support for additional programming languages.
- Improved source citations and PDF page references.

## 11. Conclusion

PyMentor AI demonstrates how web development, local language models, embeddings, and document retrieval can be combined to create an interactive Python learning assistant.

The project provides a foundation for expanding AI-assisted programming education.