# AI-Powered ATS Resume Checker

An AI-powered Applicant Tracking System (ATS) Resume Checker that evaluates resumes against job descriptions using Retrieval-Augmented Generation (RAG), semantic search, and Large Language Models.

The application analyzes resume compatibility, identifies missing skills and keywords, explains the ATS score, and provides personalized recommendations to improve interview readiness.

A Streamlit web interface allows users to upload their resume and a job description, ask follow-up questions, and receive conversational feedback.

---

## Features

- Upload Resume (PDF)
- Upload Job Description (PDF)
- ATS Match Score Calculation
- Keyword Matching
- Missing Skills Detection
- Retrieval-Augmented Generation (RAG)
- Semantic Search with FAISS
- Conversational Memory
- AI-powered Resume Recommendations
- Resume vs Job Description Comparison
- Streamlit Web Interface

---

## Demo Workflow

1. Upload a Resume.
2. Upload a Job Description.
3. Documents are parsed and cleaned.
4. Text is chunked.
5. Chunks are converted into embeddings.
6. Embeddings are indexed in FAISS.
7. Relevant chunks are retrieved using semantic search.
8. Groq Llama 3.3 generates contextual responses.
9. ATS score and keyword analysis are displayed.
10. Users can continue asking questions through the chatbot.

---

## Tech Stack

### AI / LLM

- LangChain
- Groq (Llama 3.3 70B)
- Hugging Face Embeddings
- Sentence Transformers

### Retrieval

- FAISS
- Recursive Character Text Splitter

### NLP

- Keyword Extraction
- Semantic Search
- Retrieval-Augmented Generation (RAG)

### Frontend

- Streamlit

### Python Libraries

- PyPDF
- Scikit-learn
- Pandas
- NumPy

---

## Project Structure

```
AI-ATS-Resume-Checker/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── README.md
├── faiss_index/
├── assets/
│
└── sample_data/
    ├── Resume.pdf
    └── Job_Description.pdf
```

---

## How It Works

### 1. Document Processing

- PDF loading
- Text cleaning
- Recursive chunking

### 2. Embedding Generation

Each chunk is converted into vector embeddings using:

- sentence-transformers/all-MiniLM-L6-v2

### 3. Vector Database

Embeddings are stored inside FAISS for efficient semantic retrieval.

### 4. ATS Analysis

The application extracts keywords from the job description, compares them with the resume, and computes an ATS compatibility score.

### 5. Retrieval-Augmented Generation

Relevant document chunks are retrieved and provided to the LLM, enabling grounded responses instead of relying solely on the model's internal knowledge.

### 6. Conversational Memory

The chatbot maintains conversation history, allowing users to ask follow-up questions naturally.

---

## Example Questions

- Summarize my resume.
- Compare my resume with the job description.
- Which skills are missing?
- Explain my ATS score.
- Rewrite my professional summary.
- What projects best match this role?
- How can I improve my resume?
- What keywords should I add?

---

## Installation

```bash
git clone https://github.com/jalasamir21/AI-ATS-Resume-Checker.git

cd AI-ATS-Resume-Checker

pip install -r requirements.txt
```

Run

```bash
streamlit run app.py
```

---

## Future Improvements

- Multiple Resume Comparison
- Support for DOCX resumes
- OCR for scanned PDFs
- Advanced ATS Scoring
- Export AI-generated reports
- Multi-language support
- Cloud deployment

---

## Author

**Jala Morad**
