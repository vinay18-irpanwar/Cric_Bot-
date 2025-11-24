import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()  # Load the .env file

api_key = os.getenv("GOOGLE_API_KEY")

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ------------------ UI CONFIG ------------------
st.set_page_config(
    page_title="Cricket RAG Bot 🏏",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(to right, #f8f9fa, #eef2f7);
    }
    .stChatMessage {
        border-radius: 12px !important;
        padding: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏏 Cricket Rules RAG Bot")
st.subheader("Ask anything related to cricket rules, formats, scoring, dismissals, and more!")

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("Enter Google API Key", type="password", value=api_key)

# ------------------ MAIN APP ------------------
uploaded_file = st.file_uploader("📄 Upload Cricket PDF", type=["pdf"])

if uploaded_file and api_key:
    with st.spinner("🔄 Extracting text from PDF..."):
        loader = PyMuPDFLoader(uploaded_file)
        pages = loader.load()

    # Convert pages to raw text
    text_data = "".join([p.page_content for p in pages])

    st.success("📄 PDF Loaded Successfully!")
    st.write("Preview:")
    st.text(text_data[:500] + " ...")

    # ---- CHUNKING ----
    with st.spinner("✂️ Splitting text into chunks..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        chunks = splitter.split_text(text_data)

    st.write(f"📦 Total Chunks Created: {len(chunks)}")

    doc_chunks = [Document(page_content=c) for c in chunks]

    # ---- EMBEDDINGS & VECTOR STORE ----
    with st.spinner("🧠 Creating embeddings & building vector store..."):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key
        )

        vector_store = FAISS.from_documents(doc_chunks, embeddings)

    st.success("🧠 Vector Store Ready!")

    # ---- USER QUERY ----
    query = st.text_input("❓ Ask a question related to Cricket Rules")

    if query:
        with st.spinner("🔍 Searching relevant context..."):
            results = vector_store.similarity_search_with_score(query, k=4)
            context = " ".join([doc[0].page_content for doc in results])

        # ---- LLM ----
        LLM = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key
        )

        prompt = PromptTemplate(
            template=(
                "Act as a cricket expert and answer the query: {query}\n"
                "Use the following context:\n{context}\n"
            ),
            input_variables=["query", "context"]
        )

        chain = prompt | LLM | StrOutputParser()

        answer = chain.invoke({"query": query, "context": context})

        st.markdown("### 🏏 Expert Answer:")
        st.success(answer)

else:
    st.info("⬆️ Upload a PDF & Enter API Key to begin.")

