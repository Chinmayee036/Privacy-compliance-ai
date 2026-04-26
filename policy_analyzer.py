import io
from pathlib import Path

import requests
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

st.set_page_config(page_title="PrivGuard.AI - LLM + RAG", layout="wide")

st.markdown("""
<style>
            
/* 🔥 FORCE TAB FONT SIZE (STRONG SELECTOR) */
div[data-testid="stTabs"] button p {
    font-size: 18px !important;
    font-weight: 700 !important;
}
            
/* Also target button itself */
div[data-testid="stTabs"] button {
    padding: 12px 26px !important;
}

/* Active tab */
div[data-testid="stTabs"] button[aria-selected="true"] p {
    color: #2563eb !important;
}

/* Spacing */
div[data-testid="stTabs"] [role="tablist"] {
    gap: 25px;
}


/* Hover effect */
div[data-testid="stTabs"] button:hover {
    color: #1d4ed8;
}

/* 🔍 Search bar */
.stTextInput input {
    border-radius: 12px;
    padding: 10px;
    border: 1px solid #cbd5e1;
}

/* 📄 Chunk cards */
.chunk-card {
    background: #ffffff;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}

/* Chunk title */
.chunk-title {
    font-weight: 600;
    color: #1e40af;
    margin-bottom: 6px;
}

/* Source label */
.chunk-source {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)
# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_DIR = Path("data")
MODEL_NAME = "gemini-2.5-flash"

LAW_SOURCES = {
    "dpdp": "https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf",
    "gdpr": "https://gdpr-info.eu",
    "it_act": "https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf",
}

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:

    st.markdown("""
    ## 🔐 PrivGuard.AI  
    <p style='color:gray;'>Privacy Compliance Assistant</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ⚙️ Settings")

    use_live_data = st.toggle("🌐 Live Law Data", value=True)

    law_mode = st.selectbox(
        "📜 Law Scope",
        ["All", "India (DPDP + IT Act)", "Europe (GDPR)"]
    )

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # Status cards
    st.markdown("### 📊 System Status")

    if use_live_data:
        st.success("Live Data Active")
    else:
        st.info("Using Local Files")

    if law_mode == "All":
        st.info("All Laws Loaded")
    elif "India" in law_mode:
        st.success("DPDP + IT Act Active")
    else:
        st.success("GDPR Active")

    st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
<h1 style='text-align: center; margin-bottom: 0;'>🔐 PrivGuard.AI</h1>
<h4 style='text-align: center; color: #475569; margin-top: 5px;'>
AI-powered Privacy Compliance Assistant
</h4>
<p style='text-align: center; color: #64748b;'>
Analyze policies • Ask legal questions • Check compliance instantly
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
from dotenv import load_dotenv
import os

load_dotenv()

def check_api_key():
    return os.getenv("GEMINI_API_KEY")

def load_local_documents(data_dir: Path):
    documents = []

    if not data_dir.exists():
        return documents

    for file_path in data_dir.glob("*.txt"):
        try:
            text = file_path.read_text(encoding="utf-8")
            documents.append(
                {
                    "source": file_path.name,
                    "text": text
                }
            )
        except Exception:
            continue

    return documents


@st.cache_data
def fetch_text_from_url(url: str):
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        text_parts = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        return "\n".join(text_parts)

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return text


@st.cache_data
def load_live_documents():
    documents = []

    for name, url in LAW_SOURCES.items():
        if not url:
            continue

        try:
            text = fetch_text_from_url(url)
            if text.strip():
                documents.append(
                    {
                        "source": f"{name}_live",
                        "text": text
                    }
                )
        except Exception as e:
            documents.append(
                {
                    "source": f"{name}_live_error",
                    "text": f"ERROR: Could not fetch {url}. Reason: {e}"
                }
            )

    return documents


def filter_documents_by_law(docs, selected_mode):
    if selected_mode == "India (DPDP + IT Act)":
        return [
            d for d in docs
            if "dpdp" in d["source"].lower() or "it_act" in d["source"].lower() or "it" in d["source"].lower()
        ]

    elif selected_mode == "Europe (GDPR)":
        return [
            d for d in docs
            if "gdpr" in d["source"].lower()
        ]

    return docs


def get_documents(selected_live_mode, selected_law_mode):
    local_docs = load_local_documents(DATA_DIR)

    if not selected_live_mode:
        docs = local_docs
    else:
        live_docs = load_live_documents()
        valid_live_docs = [doc for doc in live_docs if not doc["text"].startswith("ERROR:")]
        docs = valid_live_docs if valid_live_docs else local_docs

    docs = filter_documents_by_law(docs, selected_law_mode)
    return docs


def chunk_text(text, source, chunk_size=700, overlap=120):
    text = " ".join(text.split())
    chunks = []

    start = 0
    chunk_id = 1

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": source,
                    "text": chunk
                }
            )

        start += max(1, chunk_size - overlap)
        chunk_id += 1

    return chunks


@st.cache_data
def build_rag_index(selected_live_mode, selected_law_mode):
    docs = get_documents(selected_live_mode, selected_law_mode)

    all_chunks = []
    for doc in docs:
        if doc["text"].startswith("ERROR:"):
            continue
        all_chunks.extend(chunk_text(doc["text"], doc["source"]))

    if not all_chunks:
        return [], None, None

    corpus = [chunk["text"] for chunk in all_chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)

    return all_chunks, vectorizer, matrix


def retrieve_chunks(query, chunks, vectorizer, matrix, top_k=3):
    if not chunks or vectorizer is None or matrix is None:
        return []

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).flatten()

    ranked_indices = sims.argsort()[::-1][:top_k]
    results = []

    for idx in ranked_indices:
        chunk = chunks[idx].copy()
        chunk["score"] = float(sims[idx])
        results.append(chunk)

    return results


def build_prompt(question, retrieved_chunks, mode, selected_law_mode):
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk['source']} | Chunk: {chunk['chunk_id']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    style_instruction = {
        "Simple": "Explain in very simple beginner-friendly language.",
        "Student": "Explain clearly for a student, with slightly more detail.",
        "Professional": "Explain in a concise professional compliance style."
    }.get(mode, "Explain clearly.")

    prompt = f"""
You are a privacy compliance assistant.

Use ONLY the retrieved context below to answer the user's question.
Answer according to the selected jurisdiction scope: {selected_law_mode}.
Do not bring in laws outside the retrieved context.
If the answer is not clearly supported by the context, say:
"I could not confirm that clearly from the retrieved legal text."

{style_instruction}

User question:
{question}

Retrieved context:
{context}

Return your answer in this format:
1. Direct Answer
2. Explanation
3. Compliance Meaning
4. Sources Used
"""
    return prompt.strip()


def generate_answer(question, retrieved_chunks, mode, selected_law_mode):
    api_key = check_api_key()

    if not api_key:
        return "Gemini API key not found. Please enter it in the sidebar."

    client = genai.Client(api_key=api_key)
    prompt = build_prompt(question, retrieved_chunks, mode, selected_law_mode)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error while calling Gemini: {e}"


def split_questions(user_input):
    parts = user_input.replace("\n", " ").split("?")
    questions = []

    for part in parts:
        cleaned = part.strip(" .,!;:")
        if cleaned:
            questions.append(cleaned + "?")

    return questions


def risk_level(score):
    if score >= 80:
        return "Low"
    if score >= 50:
        return "Moderate"
    return "High"


def get_confidence_info(retrieved_chunks):
    if not retrieved_chunks:
        return 0.0, "Low"

    avg_score = sum(chunk["score"] for chunk in retrieved_chunks) / len(retrieved_chunks)

    if avg_score >= 0.50:
        label = "High"
    elif avg_score >= 0.20:
        label = "Medium"
    else:
        label = "Low"

    return avg_score, label


def get_compliance_questions(selected_law_mode):
    if selected_law_mode == "India (DPDP + IT Act)":
        return [
            ("Do you collect personal data?", 10, "Collection practice is clear.", "It is unclear whether personal data is collected."),
            ("Do you take user consent?", 15, "Consent mechanism is present.", "Consent mechanism is missing."),
            ("Do you clearly tell users the purpose of collection?", 15, "Purpose notice is clear.", "Purpose of data collection is not clearly told."),
            ("Do you provide correction/deletion rights?", 15, "User rights option is present.", "Correction/deletion option is missing."),
            ("Do you protect stored data securely?", 15, "Security safeguards are present.", "Security safeguards are weak or missing."),
            ("Do you define how long data is stored?", 10, "Retention period is defined.", "Retention period is not defined."),
            ("Do you provide a grievance/contact mechanism?", 10, "Grievance/contact mechanism is available.", "Grievance/contact mechanism is missing."),
            ("Do you have breach response readiness?", 10, "Breach response plan exists.", "Breach response plan is missing."),
        ]

    elif selected_law_mode == "Europe (GDPR)":
        return [
            ("Do you have a lawful basis or consent for processing?", 15, "Lawful basis/consent is present.", "Lawful basis or consent is missing."),
            ("Do you clearly explain why data is processed?", 15, "Purpose and transparency are clear.", "Purpose/transparency is weak or missing."),
            ("Do you allow access/rectification/erasure?", 15, "User rights are supported.", "Access/rectification/erasure rights are missing."),
            ("Do you collect only necessary data?", 10, "Data minimization is followed.", "Data minimization is missing."),
            ("Do you define retention limits?", 10, "Retention limits are defined.", "Retention limits are missing."),
            ("Do you protect data securely?", 15, "Security safeguards are present.", "Security safeguards are weak or missing."),
            ("Can users withdraw consent or opt out?", 10, "Withdrawal/opt-out is supported.", "Withdrawal/opt-out is missing."),
            ("Do you provide a complaint/contact mechanism?", 10, "Complaint/contact mechanism is available.", "Complaint/contact mechanism is missing."),
        ]

    else:
        return [
            ("Do you collect personal data?", 10, "Collection practice is clear.", "It is unclear whether personal data is collected."),
            ("Do you take user consent or lawful permission?", 15, "Consent/lawful basis is present.", "Consent/lawful basis is missing."),
            ("Do you clearly explain the purpose?", 15, "Purpose is clearly explained.", "Purpose is not clearly told to users."),
            ("Do you provide privacy policy/notice?", 15, "Privacy policy is present.", "Privacy policy is missing."),
            ("Do you allow deletion/correction/access rights?", 15, "User rights are supported.", "Deletion/correction/access rights are missing."),
            ("Do you protect data securely?", 15, "Security protection is present.", "Security protection is weak or missing."),
            ("Do you define retention period?", 10, "Retention period is defined.", "Retention period is not defined."),
            ("Do you provide grievance/contact mechanism?", 5, "Complaint/contact mechanism is available.", "Complaint/contact mechanism is missing."),
        ]


def get_policy_checks(selected_law_mode):
    if selected_law_mode == "India (DPDP + IT Act)":
        return [
            ("Consent", ["consent", "permission", "agree"], 15),
            ("Purpose", ["purpose", "used for", "reason", "service improvement"], 15),
            ("Correction/Deletion", ["delete", "deletion", "erase", "remove", "correction"], 15),
            ("Security", ["security", "secure", "protect", "protected", "safe"], 15),
            ("Personal Data Description", ["personal data", "user data", "information", "email", "phone"], 10),
            ("Retention", ["retention", "retain", "stored for", "how long"], 10),
            ("Grievance/Contact", ["grievance", "contact", "complaint", "reach us", "email us"], 10),
            ("Breach Readiness", ["breach", "incident", "unauthorized access", "data leak"], 10),
        ]

    elif selected_law_mode == "Europe (GDPR)":
        return [
            ("Lawful Basis / Consent", ["consent", "lawful basis", "permission", "agree"], 15),
            ("Purpose Limitation", ["purpose", "used for", "reason"], 15),
            ("Access/Rectification/Erasure", ["access", "rectification", "erase", "erasure", "delete", "correction"], 15),
            ("Data Minimization", ["necessary", "minimum", "data minimization", "limited to"], 10),
            ("Security", ["security", "secure", "protect", "protected", "safe"], 15),
            ("Retention Limitation", ["retention", "retain", "stored for", "how long"], 10),
            ("Complaint/Contact", ["contact", "complaint", "reach us", "email us"], 10),
            ("Withdraw Consent / Opt Out", ["withdraw", "withdraw consent", "unsubscribe", "opt out"], 10),
        ]

    else:
        return [
            ("Consent / Lawful Basis", ["consent", "lawful basis", "permission", "agree"], 15),
            ("Purpose", ["purpose", "used for", "reason", "service improvement"], 15),
            ("User Rights", ["delete", "deletion", "erase", "remove", "correction", "access", "rectification"], 15),
            ("Security", ["security", "secure", "protect", "protected", "safe"], 15),
            ("Personal Data Description", ["personal data", "user data", "information", "email", "phone"], 10),
            ("Retention", ["retention", "retain", "stored for", "how long"], 10),
            ("Contact / Grievance", ["contact", "grievance", "complaint", "reach us", "email us"], 10),
            ("Withdraw / Opt Out", ["withdraw", "withdraw consent", "unsubscribe", "opt out"], 10),
        ]


# -------------------------------------------------
# BUILD INDEX
# -------------------------------------------------
chunks, vectorizer, matrix = build_rag_index(use_live_data, law_mode)

if not chunks:
    st.warning("No legal data found for the selected law scope. Please check your live links or local backup files.")

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Chat", "✅ Compliance", "📄 Policy AI", "📚 Law Data"]
)

# -------------------------------------------------
# TAB 1: CHATBOT
# -------------------------------------------------
with tab1:
    st.header("Chat with PrivGuard.AI")

    mode = st.selectbox(
        "Choose explanation style:",
        ["Simple", "Student", "Professional"]
    )

    user_input = st.text_area("Type one or more questions here:", height=120)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        ask_button = st.button("Get Answer")

    with col2:
        clear_button = st.button("Clear Chat")

    with col3:
        show_history = st.checkbox("Show History")

    if clear_button:
        st.session_state.chat_history = []
        st.rerun()

    if ask_button:
        if not user_input.strip():
            st.warning("Please type a question first.")
        else:
            questions = split_questions(user_input)

            for q in questions:
                retrieved = retrieve_chunks(q, chunks, vectorizer, matrix, top_k=3)

                if not retrieved:
                    answer = "No legal chunks could be retrieved for the selected law scope."
                    avg_confidence = 0.0
                    confidence_label = "Low"
                else:
                    with st.spinner("Thinking..."):
                        answer = generate_answer(q, retrieved, mode, law_mode)

                    avg_confidence, confidence_label = get_confidence_info(retrieved)

                st.session_state.chat_history.append(
                    {
                        "question": q,
                        "answer": answer,
                        "retrieved": retrieved,
                        "avg_confidence": avg_confidence,
                        "confidence_label": confidence_label,
                        "law_mode": law_mode
                    }
                )

    # Show only latest conversation by default
    if st.session_state.chat_history:
        if show_history:
            items_to_show = st.session_state.chat_history
        else:
            items_to_show = [st.session_state.chat_history[-1]]

        for item in items_to_show:
            st.markdown("### Conversation")

            st.markdown(
                f"""
                <div style="
                    background-color:#E8F0FE;
                    padding:14px;
                    border-radius:12px;
                    margin-bottom:10px;
                    border-left:6px solid #4A90E2;">
                    <b>You:</b><br>{item['question']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div style="
                    background-color:#F6F8FA;
                    padding:14px;
                    border-radius:12px;
                    margin-bottom:10px;
                    border-left:6px solid #34A853;">
                    <b>PrivGuard.AI:</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(item["answer"])
            st.caption(f"Law scope used: {item['law_mode']}")

            st.metric("Confidence Score", f"{item['avg_confidence']:.2f}", item["confidence_label"])
            st.progress(min(item["avg_confidence"], 1.0))

            if item["confidence_label"] == "High":
                st.success("High confidence: retrieved chunks strongly matched the question.")
            elif item["confidence_label"] == "Medium":
                st.warning("Medium confidence: retrieved chunks were somewhat relevant.")
            else:
                st.error("Low confidence: retrieved chunks had weak similarity.")

            st.success("✔ This answer used retrieved legal text before generation.")

            with st.expander("See retrieved context used"):
                for chunk in item["retrieved"]:
                    st.markdown(
                        f"**Source:** {chunk['source']} | **Chunk:** {chunk['chunk_id']} | **Score:** {chunk['score']:.3f}"
                    )
                    st.write(chunk["text"])
                    st.markdown("---")

            st.markdown("---")

# -------------------------------------------------
# TAB 2: COMPLIANCE CHECKER
# -------------------------------------------------
with tab2:
    st.header("Compliance Checklist")
    st.caption(f"Law scope: {law_mode}")

    questions = get_compliance_questions(law_mode)
    answers = []

    col1, col2 = st.columns(2)
    half = (len(questions) + 1) // 2
    left_questions = questions[:half]
    right_questions = questions[half:]

    with col1:
        for idx, (question_text, points, good_msg, bad_msg) in enumerate(left_questions):
            ans = st.radio(question_text, ["Yes", "No"], key=f"comp_left_{idx}")
            answers.append((question_text, ans, points, good_msg, bad_msg))

    with col2:
        for idx, (question_text, points, good_msg, bad_msg) in enumerate(right_questions):
            ans = st.radio(question_text, ["Yes", "No"], key=f"comp_right_{idx}")
            answers.append((question_text, ans, points, good_msg, bad_msg))

    # ✅ EVERYTHING BELOW MUST BE INSIDE THIS
    if st.button("Check Compliance Score"):

        score = 0
        violations = []

        for question_text, ans, points, good_msg, bad_msg in answers:
            if ans == "Yes":
                score += points
            else:
                retrieved = retrieve_chunks(
                    question_text,
                    chunks,
                    vectorizer,
                    matrix,
                    top_k=2
                )

                violations.append({
                    "issue": bad_msg,
                    "question": question_text,
                    "chunks": retrieved
                })

        level = risk_level(score)

        st.subheader("Risk Score Dashboard")

        col1, col2 = st.columns(2)
        col1.metric("Compliance Score", f"{score}/100")
        col2.metric("Risk Level", level)

        st.progress(score / 100)

        st.subheader("❌ Compliance Violations (with Law Reference)")

        if not violations:
            st.success("No major compliance gaps found.")
        else:
            for v in violations:
                st.markdown(f"""
                <div class="card" style="border-left:5px solid #ef4444;">
                ❌ <b>Violation:</b> {v['issue']}<br>
                <b>Requirement:</b> {v['question']}
                </div>
                """, unsafe_allow_html=True)

                if v["chunks"]:
                    with st.expander("📄 Supporting Law Chunks"):
                        for chunk in v["chunks"]:
                            st.markdown(
                                f"**{chunk['source']} | Chunk {chunk['chunk_id']} (Score: {chunk['score']:.2f})**"
                            )
                            st.write(chunk["text"])
                            st.markdown("---")
                else:
                    st.warning("No direct legal reference found.")
# -------------------------------------------------
# TAB 3: POLICY ANALYZER
# -------------------------------------------------
with tab3:
    st.header("Privacy Policy Analyzer")
    st.caption(f"Law scope: {law_mode}")
    st.write("Upload a .txt policy file or paste policy text below.")

    uploaded_file = st.file_uploader("Upload policy file", type=["txt"])
    policy_text = ""

    if uploaded_file is not None:
        policy_text = uploaded_file.read().decode("utf-8")
        st.text_area("Uploaded Policy Preview", policy_text, height=200)
    else:
        policy_text = st.text_area("Paste your privacy policy text here:", height=200)

    if st.button("Analyze Policy"):
        if not policy_text.strip():
            st.warning("Please upload or paste some policy text first.")
        else:
            text = policy_text.lower()
            checks = get_policy_checks(law_mode)

            present = []
            missing = []
            score = 0

            for label, keywords, weight in checks:
                if any(word in text for word in keywords):
                    present.append(label)
                    score += weight
                else:
                    missing.append(label)

            level = risk_level(score)

            st.subheader("Policy Risk Dashboard")
            x, y, z = st.columns(3)
            x.metric("Policy Score", f"{score}/100")
            y.metric("Risk Level", level)
            z.metric("Missing Elements", len(missing))
            st.progress(score / 100)

            if level == "Low":
                st.success("This policy covers many basic compliance elements for the selected law scope.")
            elif level == "Moderate":
                st.warning("This policy covers some compliance elements but still has gaps.")
            else:
                st.error("This policy is missing several important compliance elements.")

            left, right = st.columns(2)

            with left:
                st.subheader("Present Elements")
                for item in present:
                    st.write(f"- {item}")

            with right:
                st.subheader("Missing Elements")
                for item in missing:
                    st.write(f"- {item}")


# -------------------------------------------------
# TAB 4: LAW DATA EXPLORER
# -------------------------------------------------
with tab4:
    st.header("📚 Law Data Explorer")
    st.caption(f"Law scope: {law_mode}")

    # 🔍 Search bar (chunk number OR keyword)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        search_input = st.text_input("🔍 Search laws (chunk no or keyword)")
    # 🔀 Split chunks by law
    dpdp_chunks = [c for c in chunks if "dpdp" in c["source"].lower()]
    gdpr_chunks = [c for c in chunks if "gdpr" in c["source"].lower()]
    it_chunks = [c for c in chunks if "it_act" in c["source"].lower() or "it" in c["source"].lower()]

    # 🔍 Filter function
    def filter_chunks(chunk_list):
        if not search_input.strip():
            return chunk_list

        # if number → match chunk_id
        if search_input.isdigit():
            num = int(search_input)
            return [c for c in chunk_list if c["chunk_id"] == num]

        # if text → search inside chunk text
        return [
            c for c in chunk_list
            if search_input.lower() in c["text"].lower()
        ]

    dpdp_chunks = filter_chunks(dpdp_chunks)
    gdpr_chunks = filter_chunks(gdpr_chunks)
    it_chunks = filter_chunks(it_chunks)

    # 📂 Sub-tabs
    subtab1, subtab2, subtab3 = st.tabs(["🇮🇳 DPDP", "🇪🇺 GDPR", "⚖️ IT Act"])

    # ---------------- DPDP ----------------
    with subtab1:
        st.markdown("### 🇮🇳 DPDP Act")
        st.caption(f"{len(dpdp_chunks)} chunks found")

        if dpdp_chunks:
            for chunk in dpdp_chunks:
                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-title">📄 Chunk {chunk['chunk_id']}</div>
                    <div class="chunk-source">Source: {chunk['source']}</div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("View Full Text"):
                    st.write(chunk["text"])
        else:
            st.warning("No matching results. Try another keyword or chunk number.")

    # ---------------- GDPR ----------------
    with subtab2:
        st.subheader("GDPR Chunks")

        if gdpr_chunks:
            for chunk in gdpr_chunks:
                with st.expander(f"Chunk {chunk['chunk_id']}"):
                    st.write(chunk["text"])
        else:
            st.info("No matching GDPR chunks found.")

    # ---------------- IT ACT ----------------
    with subtab3:
        st.subheader("IT Act Chunks")

        if it_chunks:
            for chunk in it_chunks:
                with st.expander(f"Chunk {chunk['chunk_id']}"):
                    st.write(chunk["text"])
        else:
            st.info("No matching IT Act chunks found.")