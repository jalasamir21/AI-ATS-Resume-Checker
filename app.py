import streamlit as st

from rag_pipeline import run_pipeline, ask, ATSPipelineError

st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="wide")
st.title("📄 ATS Resume Checker")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "ats_result" not in st.session_state:
    st.session_state.ats_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Setup")
    groq_api_key = st.text_input("Groq API Key", type="password")
    resume_file = st.file_uploader("Resume (PDF)", type="pdf")
    jd_file = st.file_uploader("Job Description (PDF)", type="pdf")
    analyze = st.button("Analyze", use_container_width=True)

if analyze:
    if not groq_api_key or not resume_file or not jd_file:
        st.sidebar.error("Provide the API key and both PDF files.")
    else:
        with st.spinner("Processing documents..."):
            try:
                ats_result, qa_chain = run_pipeline(resume_file, jd_file, groq_api_key)
            except ATSPipelineError as e:
                st.session_state.ats_result = None
                st.session_state.qa_chain = None
                st.error(f"⚠️ {e}")
            else:
                st.session_state.ats_result = ats_result
                st.session_state.qa_chain = qa_chain
                st.session_state.chat_history = []

if st.session_state.ats_result:
    ats_result = st.session_state.ats_result

    st.subheader("ATS Match Score")
    st.metric("Score", f"{ats_result['score']}%")
    st.progress(min(ats_result["score"] / 100, 1.0))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Matched keywords**")
        st.write(", ".join(ats_result["matched"]) if ats_result["matched"] else "None")
    with col2:
        st.markdown("**❌ Missing keywords**")
        st.write(", ".join(ats_result["missing"]) if ats_result["missing"] else "None")

    st.divider()
    st.subheader("Ask the ATS Assistant")

    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(content)

    question = st.chat_input("Ask about your resume vs the job description")
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = ask(st.session_state.qa_chain, question)
                except ATSPipelineError as e:
                    answer = f"⚠️ {e}"
                    st.error(answer)
                else:
                    answer = result["answer"]
                    st.write(answer)

        st.session_state.chat_history.append(("assistant", answer))
else:
    st.info("Upload a resume and job description PDF, add your Groq API key, then click Analyze.")