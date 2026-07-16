import os
import re
import tempfile

from sklearn.feature_extraction.text import CountVectorizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

class ATSPipelineError(Exception):
    """User-facing pipeline error (bad/expired API key, rate limit, etc.)."""


def _friendly_groq_error(exc):
    status = getattr(exc, "status_code", None)
    message = str(exc).lower()

    if status == 401 or "invalid api key" in message or "invalid_api_key" in message:
        return "Your Groq API key looks invalid. Double-check it in the sidebar and try again."
    if status == 429 or "rate limit" in message or "quota" in message or "insufficient_quota" in message:
        return "You've hit your Groq API rate limit or quota. Wait a bit, or check usage at console.groq.com."
    if status == 403:
        return "Your Groq API key doesn't have permission to use this model."
    if "connection" in message or "timeout" in message:
        return "Couldn't reach the Groq API. Check your internet connection and try again."
    return f"The Groq API returned an error: {str(exc)}"


QA_PROMPT_TEMPLATE = """You are an ATS Resume Assistant.
Answer ONLY using the retrieved context below.
If the answer is not supported by the retrieved documents, say the information is unavailable.
Clearly distinguish resume information from job description information.
Provide concise, actionable ATS recommendations.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def load_pdf(uploaded_file, source_label):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    for p in pages:
        p.metadata["source"] = source_label
        p.page_content = clean_text(p.page_content)

    os.unlink(tmp_path)
    return pages


def extract_keywords(text, top_n=30):
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=top_n)
    vectorizer.fit([text])
    return set(vectorizer.get_feature_names_out())


def calculate_ats_score(resume_text, jd_text):
    jd_keywords = extract_keywords(jd_text)
    resume_lower = resume_text.lower()
    matched = {kw for kw in jd_keywords if kw in resume_lower}
    missing = jd_keywords - matched
    score = round(len(matched) / len(jd_keywords) * 100, 1) if jd_keywords else 0.0
    return {"score": score, "matched": sorted(matched), "missing": sorted(missing)}


def build_chunks(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        c.metadata.setdefault("source", "unknown")
        c.metadata["page"] = c.metadata.get("page", 0)
    return chunks


def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


def build_qa_chain(vectorstore, groq_api_key, model="llama-3.3-70b-versatile"):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20},
    )
    llm = ChatGroq(model=model, temperature=0, api_key=groq_api_key)

    try:
        llm.invoke("ping")
    except Exception as exc:
        raise ATSPipelineError(_friendly_groq_error(exc)) from exc

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "chat_history", "question"],
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


def ask(qa_chain, question):
    try:
        return qa_chain.invoke({"question": question})
    except Exception as exc:
        raise ATSPipelineError(_friendly_groq_error(exc)) from exc


def run_pipeline(resume_file, jd_file, groq_api_key, model="llama-3.3-70b-versatile"):
    resume_pages = load_pdf(resume_file, "resume")
    jd_pages = load_pdf(jd_file, "job_description")
    documents = resume_pages + jd_pages

    resume_full_text = " ".join(p.page_content for p in resume_pages)
    jd_full_text = " ".join(p.page_content for p in jd_pages)
    ats_result = calculate_ats_score(resume_full_text, jd_full_text)

    chunks = build_chunks(documents)
    vectorstore = build_vectorstore(chunks)
    qa_chain = build_qa_chain(vectorstore, groq_api_key, model=model)

    return ats_result, qa_chain