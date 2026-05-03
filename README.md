# Privacy-compliance-ai
# 🔐 PrivGuard.AI

AI-powered Privacy Compliance Assistant using LLM + RAG

---

## 🚀 Overview

PrivGuard.AI helps users understand, analyze, and evaluate privacy laws and policies using an intelligent Retrieval-Augmented Generation (RAG) system.

Instead of generating generic answers, it retrieves actual legal text (DPDP, GDPR, IT Act) and uses it to produce grounded, explainable responses.

---

## ✨ Features

### 💬 AI Chat (RAG-based)
- Ask legal/privacy questions
- Answers generated using retrieved law data
- Supports:
  - 🇮🇳 DPDP + IT Act
  - 🇪🇺 GDPR
- Confidence score for each response

---

### ✅ Compliance Checker
- Interactive checklist
- Generates:
  - Compliance score (/100)
  - Risk level (Low / Moderate / High)
- Shows violations with legal references

---

### 📄 Policy Analyzer
- Upload or paste privacy policy
- Detects:
  - Missing compliance elements
  - Risk level
- Quick keyword-based evaluation

---

### 📚 Law Explorer
- Browse legal documents in chunks
- Search by:
  - Keywords
  - Chunk number
- Organized by law (DPDP / GDPR / IT Act)

---

## 🧠 Tech Stack

- Frontend: Streamlit  
- LLM: Gemini API  
- RAG Pipeline:  
  - TF-IDF Vectorization  
  - Cosine Similarity  
- Data Processing:  
  - BeautifulSoup (HTML parsing)  
  - PyPDF (PDF extraction)  

---

## ⚙️ How It Works

1. Fetch legal documents (live or local)
2. Split into chunks
3. Convert to vectors using TF-IDF
4. Retrieve top relevant chunks for a query
5. Pass context + question to Gemini
6. Generate structured, grounded answer

---

## 📦 Installation

```bash
git clone https://github.com/Chinmayee036/privguard-ai.git
cd privguard-ai
pip install -r requirements.txt
```

---

## 🔑 Setup

Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

---

## ▶️ Run

```bash
streamlit run app.py
```

---

## 📊 Key Highlights

- RAG-based legal reasoning (not hallucination)
- Multi-law support (India + Europe)
- Explainable AI with source chunks
- Real-time law fetching
- Clean UI with multiple modules

---

## ⚠️ Disclaimer

This project is for educational purposes only and does not constitute legal advice.

---

## 🙌 Acknowledgment

Built during a hackathon to explore real-world applications of AI in privacy and compliance.
