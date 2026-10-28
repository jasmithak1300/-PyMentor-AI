from flask import Flask, request, jsonify, render_template_string
from pypdf import PdfReader
import json
import math
import urllib.request
import urllib.error

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

OLLAMA_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

# Temporary in-memory storage for uploaded PDF notes.
uploaded_notes = []
uploaded_pdf_name = ""


# =========================================================
# OLLAMA AI FUNCTIONS
# =========================================================

def ask_ollama(prompt):
    """Generate an answer using the local Ollama chat model."""

    payload = {
        "model": CHAT_MODEL,
        "prompt": prompt,
        "stream": False
    }

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result.get("response", "No response received.")


def get_embedding(text):
    """Generate an embedding using the Ollama embedding model."""

    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))

    embedding = result.get("embedding")

    if not embedding:
        raise ValueError("The embedding model returned no embedding.")

    return embedding


def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two vectors."""

    if len(vector_a) != len(vector_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vector_a, vector_b))

    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def error_response(error):
    """Return a readable AI service error."""

    app.logger.exception("AI request failed")

    return jsonify({
        "answer": (
            "I couldn't complete that request. Please check that "
            "Ollama is running and both required models are available."
        ),
        "error": str(error)
    }), 503


# =========================================================
# PDF PROCESSING AND RETRIEVAL
# =========================================================

def extract_pdf_pages(pdf_file):
    """Extract readable text while preserving original page numbers."""

    reader = PdfReader(pdf_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    return pages


def split_text_into_chunks(text, chunk_size=1000, overlap=150):
    """Split a page into overlapping text chunks."""

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def retrieve_relevant_notes(question, top_k=3):
    """Retrieve the most relevant uploaded-note sections."""

    if not uploaded_notes:
        return []

    question_embedding = get_embedding(question)
    scored = []

    for note in uploaded_notes:
        score = cosine_similarity(
            question_embedding,
            note["embedding"]
        )

        scored.append({
            "text": note["text"],
            "page": note["page"],
            "score": score
        })

    scored.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored[:top_k]


# =========================================================
# WEBSITE INTERFACE
# =========================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>PyMentor AI | Intelligent Python Learning Assistant</title>

<style>
:root {
    --bg: #0b1020;
    --panel: #121a2e;
    --panel2: #19233a;
    --text: #edf2ff;
    --muted: #9ba9c5;
    --accent: #8b7cff;
    --green: #5ee0b5;
    --border: #293550;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: "Segoe UI", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
}

header {
    padding: 18px 5%;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 15px;
    flex-wrap: wrap;
}

.logo {
    font-size: 23px;
    font-weight: 800;
}

.logo span {
    color: var(--accent);
}

.badge {
    background: #173a35;
    color: var(--green);
    padding: 8px 13px;
    border-radius: 20px;
    font-size: 12px;
}

.layout {
    display: grid;
    grid-template-columns: 230px minmax(0, 1fr);
    min-height: calc(100vh - 75px);
}

nav {
    padding: 25px 15px;
    border-right: 1px solid var(--border);
}

.nav-label {
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 2px;
    padding: 10px;
}

.nav-button {
    width: 100%;
    padding: 13px;
    margin: 4px 0;
    background: transparent;
    border: none;
    border-radius: 10px;
    color: var(--muted);
    text-align: left;
    cursor: pointer;
    font-size: 14px;
}

.nav-button:hover,
.nav-button.active {
    background: var(--panel2);
    color: white;
}

main {
    padding: 35px;
    max-width: 1250px;
    width: 100%;
    min-width: 0;
}

.page {
    display: none;
}

.page.active {
    display: block;
}

h1 {
    font-size: clamp(26px, 4vw, 38px);
    margin: 0 0 12px;
}

h2 {
    margin-top: 0;
}

p {
    line-height: 1.7;
}

.muted {
    color: var(--muted);
}

.hero {
    padding: 35px;
    border: 1px solid var(--border);
    border-radius: 20px;
    background: linear-gradient(130deg, #1c2345, #172c38);
    margin-bottom: 25px;
}

.hero h1 span {
    color: var(--accent);
}

.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 17px;
    margin-top: 22px;
}

.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 18px;
    min-width: 0;
}

.card h3 {
    margin-top: 0;
}

.icon {
    font-size: 28px;
    margin-bottom: 12px;
}

button,
select {
    font: inherit;
}

.primary {
    background: var(--accent);
    color: white;
    border: none;
    padding: 12px 18px;
    border-radius: 9px;
    cursor: pointer;
    font-weight: 600;
}

.primary:hover {
    opacity: .88;
}

.primary:disabled {
    opacity: .6;
    cursor: wait;
}

.secondary {
    background: var(--panel2);
    color: white;
    border: 1px solid var(--border);
    padding: 11px 15px;
    border-radius: 9px;
    cursor: pointer;
}

textarea,
input,
select {
    width: 100%;
    background: #0b1224;
    color: white;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    margin: 9px 0 15px;
}

textarea {
    min-height: 135px;
    resize: vertical;
    font-family: Consolas, monospace;
}

input[type="file"] {
    padding: 12px;
}

label {
    font-weight: 600;
}

.output {
    background: #080d19;
    border: 1px solid var(--border);
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 1.7;
}

.hidden {
    display: none;
}

.lesson {
    padding: 18px;
    background: var(--panel2);
    border-radius: 12px;
    margin: 12px 0;
}

.option {
    display: block;
    background: var(--panel2);
    padding: 12px;
    border-radius: 8px;
    margin: 8px 0;
    cursor: pointer;
}

.option input {
    width: auto;
    margin: 0 8px 0 0;
}

.status {
    color: var(--green);
    font-size: 13px;
}

footer {
    border-top: 1px solid var(--border);
    padding: 20px;
    text-align: center;
    color: var(--muted);
    font-size: 12px;
}

@media (max-width: 720px) {
    .layout {
        grid-template-columns: 1fr;
    }

    nav {
        border-right: none;
        border-bottom: 1px solid var(--border);
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
    }

    .nav-label {
        width: 100%;
    }

    .nav-button {
        width: auto;
        flex: 1 1 120px;
    }

    main {
        padding: 20px;
    }

    .hero {
        padding: 23px;
    }
}
</style>
</head>

<body>

<header>
    <div class="logo">PyMentor <span>AI</span></div>
    <div class="badge">● Local AI · Ollama</div>
</header>

<div class="layout">

<nav>
    <div class="nav-label">LEARNING SPACE</div>

    <button class="nav-button active" data-page="dashboard">🏠 Dashboard</button>
    <button class="nav-button" data-page="tutor">🤖 AI Python Tutor</button>
    <button class="nav-button" data-page="learn">📚 Learn Python</button>
    <button class="nav-button" data-page="explainer">💡 Code Explainer</button>
    <button class="nav-button" data-page="quiz">📝 Quiz Center</button>
    <button class="nav-button" data-page="practice">💻 Practice Lab</button>
    <button class="nav-button" data-page="notes">📄 PDF Study Assistant</button>

    <p class="muted" style="padding:10px;font-size:12px">
        Learn Python one concept at a time.
    </p>
</nav>

<main>

<section class="page active" id="dashboard">
    <div class="hero">
        <p class="status">YOUR PERSONAL PYTHON LEARNING SPACE</p>
        <h1>Learn Python.<br><span>Think Smarter.</span></h1>
        <p class="muted">
            Ask questions, understand code, practise programming,
            and explore your Python study notes with AI.
        </p>
        <button class="primary" onclick="showPage('tutor')">Start Learning →</button>
    </div>

    <h2>Your Learning Workspace</h2>

    <div class="cards">
        <div class="card">
            <div class="icon">🤖</div>
            <h3>AI Python Tutor</h3>
            <p class="muted">Ask Python questions and get AI-generated explanations.</p>
            <button class="secondary" onclick="showPage('tutor')">Open Tutor</button>
        </div>

        <div class="card">
            <div class="icon">📚</div>
            <h3>Learn Python</h3>
            <p class="muted">Explore beginner-friendly lessons and examples.</p>
            <button class="secondary" onclick="showPage('learn')">Explore Lessons</button>
        </div>

        <div class="card">
            <div class="icon">💡</div>
            <h3>Code Explainer</h3>
            <p class="muted">Understand code logic and expected output.</p>
            <button class="secondary" onclick="showPage('explainer')">Explain Code</button>
        </div>

        <div class="card">
            <div class="icon">📄</div>
            <h3>PDF Study Assistant</h3>
            <p class="muted">Upload notes and ask questions about their contents.</p>
            <button class="secondary" onclick="showPage('notes')">Study My Notes</button>
        </div>
    </div>
</section>


<section class="page" id="tutor">
    <h1>AI Python Tutor 🤖</h1>
    <p class="muted">Ask your local AI model questions about Python.</p>

    <div class="card">
        <label for="question">Your question</label>
        <textarea id="question" placeholder="Explain Python loops with an example..."></textarea>
        <button class="primary" onclick="askTutor()">Ask PyMentor AI →</button>
        <div id="tutorResult" class="output hidden"></div>
    </div>
</section>


<section class="page" id="learn">
    <h1>Learn Python 📚</h1>
    <p class="muted">Begin with these essential programming concepts.</p>

    <div class="lesson">
        <h3>01 · Variables</h3>
        <p>Variables store values for later use.</p>
        <pre class="output">name = "Alex"
age = 20

print(name)
print(age)</pre>
    </div>

    <div class="lesson">
        <h3>02 · Data Types</h3>
        <p>Python supports integers, floating-point numbers, strings and booleans.</p>
        <pre class="output">age = 20
price = 99.5
name = "Python"
is_learning = True

print(type(age))
print(type(name))</pre>
    </div>

    <div class="lesson">
        <h3>03 · Conditional Statements</h3>
        <p>Conditions let programs make decisions.</p>
        <pre class="output">marks = 85

if marks >= 40:
    print("Pass")
else:
    print("Try again")</pre>
    </div>

    <div class="lesson">
        <h3>04 · Loops</h3>
        <p>Loops repeat instructions.</p>
        <pre class="output">for number in range(1, 6):
    print(number)</pre>
    </div>

    <div class="lesson">
        <h3>05 · Functions</h3>
        <p>Functions group reusable instructions.</p>
        <pre class="output">def greet(name):
    return "Hello " + name

print(greet("Alex"))</pre>
    </div>

    <button class="primary" onclick="showPage('quiz')">Test Your Knowledge →</button>
</section>


<section class="page" id="explainer">
    <h1>AI Code Explainer 💡</h1>
    <p class="muted">Paste code and choose the type of explanation you need.</p>

    <div class="card">
        <label for="codeInput">Python code</label>
        <textarea id="codeInput">for i in range(1, 6):
    print(i * i)</textarea>

        <label for="explainMode">Explanation style</label>
        <select id="explainMode">
            <option value="beginner">Beginner-friendly</option>
            <option value="line">Explain line by line</option>
            <option value="output">Explain expected output</option>
            <option value="improve">Suggest improvements</option>
        </select>

        <button class="primary" onclick="explainCode()">Explain My Code →</button>
        <div id="explainResult" class="output hidden"></div>
    </div>
</section>


<section class="page" id="quiz">
    <h1>Python Quiz Center 📝</h1>
    <p class="muted">Answer five questions to test your fundamentals.</p>

    <div class="card" id="quizContainer"></div>
    <button class="primary" onclick="submitQuiz()">Submit Quiz</button>
    <div id="quizResult" class="output hidden"></div>
</section>


<section class="page" id="practice">
    <h1>Python Practice Lab 💻</h1>
    <p class="muted">Get AI feedback on your proposed solution.</p>

    <div class="card">
        <label for="practiceCode">Your solution</label>
        <textarea id="practiceCode">numbers = [1, 2, 3, 4, 5]
print(sum(numbers))</textarea>

        <label for="practiceTask">Practice question</label>
        <input id="practiceTask" value="Calculate the sum of a list of numbers.">

        <button class="primary" onclick="checkPractice()">Review My Code →</button>
        <div id="practiceResult" class="output hidden"></div>

        <p class="muted">This feature reviews code but does not execute it.</p>
    </div>
</section>


<section class="page" id="notes">
    <h1>PDF Study Assistant 📄</h1>
    <p class="muted">
        Upload a text-based PDF of Python notes. PyMentor will retrieve
        relevant sections to help answer your questions.
    </p>

    <div class="card">
        <h3>1. Upload your study notes</h3>

        <form id="uploadForm">
            <label for="pdfFile">Choose a PDF file (maximum 10 MB)</label>
            <input type="file" id="pdfFile" accept=".pdf,application/pdf" required>
            <button class="primary" type="submit">Upload and Process PDF</button>
        </form>

        <div id="uploadResult" class="output hidden"></div>
    </div>

    <div class="card">
        <h3>2. Ask your notes</h3>

        <label for="notesQuestion">Your question</label>
        <textarea id="notesQuestion" placeholder="Summarize the main concepts in my Python notes..."></textarea>

        <button class="primary" onclick="askNotes()">Ask My Notes →</button>
        <div id="notesResult" class="output hidden"></div>
    </div>
</section>

</main>
</div>

<footer>
    PyMentor AI · An Intelligent Python Learning Assistant
    <br><br>
    Educational project · Flask · Ollama · Retrieval-Augmented Generation
</footer>


<script>
function showPage(pageId) {
    document.querySelectorAll(".page").forEach(page => {
        page.classList.remove("active");
    });

    document.querySelectorAll(".nav-button").forEach(button => {
        button.classList.remove("active");
    });

    document.getElementById(pageId).classList.add("active");

    const button = document.querySelector(
        '.nav-button[data-page="' + pageId + '"]'
    );

    if (button) button.classList.add("active");

    window.scrollTo({top: 0, behavior: "smooth"});
}

document.querySelectorAll(".nav-button").forEach(button => {
    button.addEventListener("click", () => showPage(button.dataset.page));
});


async function sendAIRequest(url, data, resultId, button) {
    const output = document.getElementById(resultId);
    const originalText = button.textContent;

    output.classList.remove("hidden");
    output.textContent = "PyMentor AI is thinking...";

    button.disabled = true;
    button.textContent = "Please wait...";

    try {
        const response = await fetch(url, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || result.answer || "Request failed.");
        }

        output.textContent =
            result.answer || result.response || "No answer received.";

        if (result.source) {
            output.textContent += "\n\nSource PDF: " + result.source;
        }
    } catch (error) {
        output.textContent =
            "Request failed:\n" + error.message +
            "\n\nCheck that Ollama is running and try again.";
    } finally {
        button.disabled = false;
        button.textContent = originalText;
    }
}


function askTutor() {
    const question = document.getElementById("question").value.trim();

    if (!question) {
        alert("Please enter a Python question.");
        return;
    }

    sendAIRequest(
        "/api/tutor",
        {question},
        "tutorResult",
        document.querySelector("#tutor .primary")
    );
}


function explainCode() {
    const code = document.getElementById("codeInput").value.trim();
    const mode = document.getElementById("explainMode").value;

    if (!code) {
        alert("Please enter Python code.");
        return;
    }

    sendAIRequest(
        "/api/explain",
        {code, mode},
        "explainResult",
        document.querySelector("#explainer .primary")
    );
}


function checkPractice() {
    const code = document.getElementById("practiceCode").value.trim();
    const task = document.getElementById("practiceTask").value.trim();

    if (!code || !task) {
        alert("Enter both the task and your solution.");
        return;
    }

    sendAIRequest(
        "/api/practice-check",
        {code, task},
        "practiceResult",
        document.querySelector("#practice .primary")
    );
}


const questions = [
    {
        q: "Which keyword defines a function in Python?",
        options: ["function", "def", "fun", "define"],
        answer: 1
    },
    {
        q: "Which data type stores True or False?",
        options: ["str", "int", "bool", "list"],
        answer: 2
    },
    {
        q: "What does len([1, 2, 3]) return?",
        options: ["2", "3", "4", "6"],
        answer: 1
    },
    {
        q: "Which symbol starts a single-line Python comment?",
        options: ["//", "<!--", "#", "**"],
        answer: 2
    },
    {
        q: "Which loop iterates over a sequence?",
        options: ["repeat", "foreach", "for", "iterate"],
        answer: 2
    }
];


function renderQuiz() {
    document.getElementById("quizContainer").innerHTML =
        questions.map((question, index) => `
            <div class="lesson">
                <h3>${index + 1}. ${question.q}</h3>
                ${question.options.map((option, optionIndex) => `
                    <label class="option">
                        <input type="radio" name="q${index}" value="${optionIndex}">
                        ${option}
                    </label>
                `).join("")}
            </div>
        `).join("");
}


function submitQuiz() {
    let score = 0;
    let unanswered = 0;

    questions.forEach((question, index) => {
        const selected = document.querySelector(
            'input[name="q' + index + '"]:checked'
        );

        if (!selected) {
            unanswered++;
        } else if (Number(selected.value) === question.answer) {
            score++;
        }
    });

    const result = document.getElementById("quizResult");
    result.classList.remove("hidden");
    result.textContent =
        "Your score: " + score + "/" + questions.length +
        "\nPercentage: " + Math.round(score / questions.length * 100) +
        "%\nUnanswered questions: " + unanswered;
}


document.getElementById("uploadForm").addEventListener("submit", async event => {
    event.preventDefault();

    const fileInput = document.getElementById("pdfFile");
    const output = document.getElementById("uploadResult");
    const button = event.submitter;
    const originalText = button.textContent;

    if (!fileInput.files.length) {
        alert("Choose a PDF first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    output.classList.remove("hidden");
    output.textContent = "Extracting text and creating embeddings...";

    button.disabled = true;
    button.textContent = "Processing...";

    try {
        const response = await fetch("/api/notes/upload", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "PDF upload failed.");
        }

        output.textContent =
            result.message + "\nFile: " + result.filename +
            "\nSections indexed: " + result.chunks;
    } catch (error) {
        output.textContent = "Upload failed:\n" + error.message;
    } finally {
        button.disabled = false;
        button.textContent = originalText;
    }
});


function askNotes() {
    const question = document.getElementById("notesQuestion").value.trim();

    if (!question) {
        alert("Enter a question about your notes.");
        return;
    }

    sendAIRequest(
        "/api/notes/ask",
        {question},
        "notesResult",
        document.querySelector("#notes .card:nth-of-type(2) .primary")
    );
}


renderQuiz();
</script>

</body>
</html>
"""


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template_string(HTML)


# =========================================================
# AI PYTHON TUTOR
# =========================================================

@app.route("/api/tutor", methods=["POST"])
def tutor():
    data = request.get_json(silent=True) or {}

    question = (
        data.get("question")
        or data.get("message")
        or data.get("prompt")
        or ""
    ).strip()

    if not question:
        return jsonify({"answer": "Please enter a question."}), 400

    prompt = f"""
You are PyMentor AI, a helpful Python programming tutor.

Explain concepts in simple language. Give accurate examples,
break down complex code step by step, and encourage students
to understand programming logic.

Student question:
{question}
"""

    try:
        answer = ask_ollama(prompt)
        return jsonify({"answer": answer, "response": answer})
    except Exception as error:
        return error_response(error)


# =========================================================
# AI CODE EXPLAINER
# =========================================================

@app.route("/api/explain", methods=["POST"])
def explain():
    data = request.get_json(silent=True) or {}

    code = data.get("code", "").strip()
    mode = data.get("mode", "beginner")

    if not code:
        return jsonify({"answer": "Please enter some Python code."}), 400

    instructions = {
        "beginner": "Explain the code to a complete beginner.",
        "line": "Explain the code line by line.",
        "output": "Explain the expected output and execution flow.",
        "improve": "Review the code and suggest improvements."
    }

    prompt = f"""
You are PyMentor AI, a Python code explanation assistant.

{instructions.get(mode, instructions["beginner"])}

Identify possible errors and explain the expected behaviour.
Do not claim to have executed the code.

Python code:
{code}
"""

    try:
        answer = ask_ollama(prompt)
        return jsonify({"answer": answer, "response": answer})
    except Exception as error:
        return error_response(error)


# =========================================================
# PRACTICE REVIEW
# =========================================================

@app.route("/api/practice-check", methods=["POST"])
def practice_check():
    data = request.get_json(silent=True) or {}

    code = data.get("code", "").strip()
    task = data.get("task", "").strip()

    if not code or not task:
        return jsonify({
            "answer": "Please provide both the exercise and your code."
        }), 400

    prompt = f"""
You are PyMentor AI, a Python practice mentor.

Review the student's solution. Explain whether the approach appears
correct, identify possible bugs, and suggest improvements.
Provide a corrected example if needed.

Do not claim that you executed the code.

Exercise:
{task}

Student solution:
{code}
"""

    try:
        answer = ask_ollama(prompt)
        return jsonify({"answer": answer, "response": answer})
    except Exception as error:
        return error_response(error)


# =========================================================
# PDF UPLOAD API
# =========================================================

@app.route("/api/notes/upload", methods=["POST"])
def upload_notes():
    global uploaded_notes, uploaded_pdf_name

    if "file" not in request.files:
        return jsonify({"error": "Please select a PDF file."}), 400

    pdf_file = request.files["file"]

    if not pdf_file.filename:
        return jsonify({"error": "Please select a file."}), 400

    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    try:
        pages = extract_pdf_pages(pdf_file)

        if not pages:
            return jsonify({
                "error": (
                    "No readable text was found. "
                    "Scanned PDFs require OCR."
                )
            }), 400

        new_notes = []

        for page in pages:
            chunks = split_text_into_chunks(page["text"])

            for chunk in chunks:
                embedding = get_embedding(chunk)

                new_notes.append({
                    "text": chunk,
                    "embedding": embedding,
                    "page": page["page"]
                })

        if not new_notes:
            return jsonify({"error": "No text sections were found."}), 400

        # Replace the current notes only after processing succeeds.
        uploaded_notes = new_notes
        uploaded_pdf_name = pdf_file.filename

        return jsonify({
            "message": "PDF processed successfully!",
            "filename": uploaded_pdf_name,
            "chunks": len(uploaded_notes)
        })

    except Exception as error:
        app.logger.exception("PDF processing failed")

        return jsonify({
            "error": str(error)
        }), 500


# =========================================================
# PDF QUESTION ANSWERING
# =========================================================

@app.route("/api/notes/ask", methods=["POST"])
def ask_about_notes():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please enter a question."}), 400

    if not uploaded_notes:
        return jsonify({
            "answer": "Upload a readable PDF before asking questions."
        }), 400

    try:
        relevant_chunks = retrieve_relevant_notes(question, top_k=3)

        context = "\n\n".join(
            f"Source PDF page {item['page']}:\n{item['text']}"
            for item in relevant_chunks
        )

        prompt = f"""
You are PyMentor AI's PDF Study Assistant.

Answer the question using the supplied study-note context.
Explain ideas clearly for a student. Do not invent facts or quotations.
If the notes do not contain enough information, say so.
Clearly distinguish any additional general knowledge from information
found in the supplied notes.

Study-note context:
{context}

Student question:
{question}
"""

        answer = ask_ollama(prompt)

        return jsonify({
            "answer": answer,
            "source": uploaded_pdf_name,
            "sections_used": len(relevant_chunks)
        })

    except Exception as error:
        app.logger.exception("PDF question answering failed")

        return jsonify({
            "error": str(error),
            "answer": "Unable to answer from the notes. Check Ollama and try again."
        }), 500


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():
    return jsonify({
        "application": "PyMentor AI",
        "status": "running",
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBED_MODEL,
        "pdf_loaded": bool(uploaded_notes)
    })


if __name__ == "__main__":
    print("=" * 45)
    print("PyMentor AI - Python Learning Assistant")
    print("Website: http://127.0.0.1:5000")
    print("Chat model:", CHAT_MODEL)
    print("Embedding model:", EMBED_MODEL)
    print("Press CTRL+C to stop the server.")
    print("=" * 45)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )