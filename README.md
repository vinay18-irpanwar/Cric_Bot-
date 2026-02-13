# 🏏 Circk-Bot  
### A Cricket Rules RAG (Retrieval-Augmented Generation) Application

Circk-Bot is a **Retrieval-Augmented Generation (RAG)** based AI application that allows users to upload a cricket-related PDF and ask intelligent questions based on its content.

The system follows a complete RAG pipeline:

> **Upload File → Chunking → Embeddings → Vector Database → Semantic Search → LLM Response**

It combines document retrieval with generative AI to provide accurate, context-aware answers about cricket rules, formats, scoring systems, dismissals, and more.

---

## 🚀 Project Overview

Circk-Bot enables users to:

- Upload cricket-related PDF documents  
- Automatically extract and process text  
- Convert text into embeddings  
- Store embeddings inside a FAISS vector database  
- Perform semantic similarity search  
- Generate expert-level answers using Google Gemini LLM  

This application demonstrates a practical implementation of a **RAG architecture using LangChain + Google Generative AI**.

---

## 🧠 How the RAG Flow Works

1. **PDF Upload**  
   User uploads a cricket-related PDF.

2. **Text Extraction**  
   PyMuPDF extracts raw text from the document.

3. **Chunking**  
   Text is split into smaller chunks using `RecursiveCharacterTextSplitter`.

4. **Embeddings Creation**  
   Each chunk is converted into vector embeddings using Google’s embedding model.

5. **Vector Storage (FAISS)**  
   Embeddings are stored in a FAISS vector database.

6. **Semantic Search**  
   User query is compared against stored vectors to retrieve relevant chunks.

7. **LLM Response Generation**  
   Retrieved context + user query → Passed to Gemini LLM → Final answer generated.

---

## 🛠️ Tech Stack

### 🖥️ Frontend
- Streamlit

### ⚙️ Backend / AI Pipeline
- Python
- LangChain
- Google Generative AI (Gemini Models)
- FAISS (Vector Database)

### 📄 Document Processing
- PyMuPDF
- RecursiveCharacterTextSplitter

### 🧩 Embeddings & LLM
- `gemini-embedding-001`
- `gemini-2.5-flash`

### 🔐 Environment Management
- python-dotenv

---

## 📦 Installation & Setup Guide

Follow the steps below to run this project locally.

---

### 1️⃣ Clone the Repository

bash
git clone https://github.com/your-username/circk-bot.git
cd circk-bot

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv


Activate:

Windows

venv\Scripts\activate


Mac/Linux

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt


If requirements file is not available:

pip install streamlit langchain langchain-community langchain-google-genai faiss-cpu pymupdf python-dotenv

4️⃣ Setup Environment Variables

Create a .env file in the root directory:

GOOGLE_API_KEY=your_google_api_key_here


You can obtain your API key from Google AI Studio.

5️⃣ Run the Application
streamlit run app.py


The application will open in your browser at:

http://localhost:8501

💻 Usage Instructions

Enter your Google API key in the sidebar (if not stored in .env)

Upload a cricket-related PDF file

Wait for text extraction and embedding creation

Ask a cricket-related question

Receive AI-generated expert answer

📂 Project Structure
circk-bot/
│
├── app.py                # Main Streamlit application
├── .env                  # Environment variables
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation

🔎 Example Queries

What are the rules for LBW dismissal?

How does Duckworth-Lewis method work?

What is the powerplay rule in ODI cricket?

What are the different types of no-balls?

📈 Future Improvements

Support multiple PDFs

Add persistent vector storage

Add chat history memory

Improve chunking strategy

Add hybrid search (keyword + semantic)

Deploy on cloud (Streamlit Cloud / AWS / GCP)

Add authentication system

🤝 Contribution Guidelines

Contributions are welcome!

To contribute:

Fork the repository

Create a new branch

Make changes

Commit with proper messages

Submit a Pull Request

Please ensure:

Code is clean and well-commented

No API keys are committed

Proper documentation is maintained

📜 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute this project.

🎯 Learning Outcomes

This project demonstrates:

Real-world RAG implementation

Vector databases with FAISS

Semantic search

Prompt engineering

LLM integration

End-to-end AI application development

⭐ Support

If you found this project useful:

Star the repository

Share it with others

Contribute improvements
