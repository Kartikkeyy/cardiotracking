#  AI Medical Assistant Chatbot — RAG-based Application   

---

##  Project Overview

An intelligent **Medical Domain Chatbot** built using **Retrieval-Augmented Generation (RAG)** technology. This application enables users to upload medical documents (textbooks, research papers, clinical reports) and receive accurate, context-aware answers to their medical queries.

The system leverages advanced natural language processing to retrieve relevant information from uploaded documents before generating responses, ensuring accuracy and reducing AI hallucinations in the medical domain.

---

##  What is RAG?

**RAG (Retrieval-Augmented Generation)** is an AI architecture that combines information retrieval with language generation. Instead of relying solely on the model's training data, RAG:

1. Retrieves relevant information from a custom knowledge base
2. Uses that context to generate accurate, grounded responses
3. Prevents hallucinations by anchoring answers in actual document content

This approach is particularly valuable in specialized domains like healthcare where accuracy is critical.

---

##  System Architecture

```
User Query
   ↓
Query Embedding (Google Generative AI)
   ↓
Vector Similarity Search (Pinecone)
   ↓
Retrieve Relevant Document Chunks
   ↓
Context + Query → LLM (Groq LLaMA3-70B)
   ↓
Generated Answer
```

**Document Processing Pipeline:**
```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Storage (Pinecone)
```

---

##  Key Features

- ** Document Upload**: Support for multiple medical PDF documents
- ** Intelligent Retrieval**: Semantic search across uploaded documents
- ** Accurate Responses**: Context-aware answers using state-of-the-art LLM
- ** Fast Processing**: Optimized vector search with Pinecone
- ** Privacy Focused**: Your documents stay in your vector database
- ** Source Tracking**: Responses include references to source documents
- ** Chat Interface**: User-friendly Streamlit frontend
- ** REST API**: FastAPI backend for easy integration

---

##  Technology Stack

| Component       | Technology                          |
| --------------- | ----------------------------------- |
| Language Model  | Groq API (LLaMA 3.3 70B)           |
| Embeddings      | Google Generative AI (Gemini)       |
| Vector Database | Pinecone (Serverless)               |
| Framework       | LangChain                           |
| Backend API     | FastAPI                             |
| Frontend        | Streamlit                           |
| Document Parser | PyPDF                               |
| Deployment      | Render                              |

---

##  API Endpoints

### Upload Documents
```http
POST /upload_pdfs/
Content-Type: multipart/form-data

Uploads one or more PDF files to the knowledge base
```

### Ask Question
```http
POST /ask/
Content-Type: application/x-www-form-urlencoded

Body: question=<your medical query>

Returns: AI-generated answer with source references
```

---

##  Project Structure

```
medical-assistant/
│
├── assets/                      # Documentation and media
│   ├── DIABETES.pdf
│   ├── MedicalAssistant.pdf
│   └── medicalAssistant.png
│
├── client/                      # Streamlit frontend
│   ├── components/
│   │   ├── chatUI.py           # Chat interface
│   │   ├── upload.py           # File upload component
│   │   └── history_download.py # Chat history
│   ├── utils/
│   │   └── api.py              # API client
│   ├── app.py                  # Main Streamlit app
│   ├── config.py               # Configuration
│   └── requirements.txt
│
└── server/                      # FastAPI backend
    ├── modules/
    │   ├── llm.py              # LLM chain setup
    │   ├── load_vectorstore.py # Vector DB operations
    │   ├── query_handlers.py   # Query processing
    │   └── pdf_handlers.py     # PDF processing
    ├── routes/
    │   ├── upload_pdfs.py      # Upload endpoint
    │   └── ask_question.py     # Query endpoint
    ├── middlewares/
    │   └── exception_handlers.py
    ├── uploaded_docs/           # Temporary PDF storage
    ├── main.py                  # FastAPI app
    ├── logger.py                # Logging configuration
    └── requirements.txt
```

---

##  Getting Started

### Prerequisites

- Python 3.9+
- Google AI API Key
- Groq API Key
- Pinecone API Key

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/ratnesh003/medical-assistant.git
cd medical-assistant
```

#### 2. Setup Backend

```bash
cd server

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Create .env file with:
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=medicalindex1

# Run the server
uvicorn main:app --reload --port 8000
```

#### 3. Setup Frontend

```bash
cd client

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

#### 4. Access the Application

- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

##  Configuration

### Environment Variables

Create a `.env` file in the `server/` directory:

```env
# Google AI (for embeddings)
GOOGLE_API_KEY=your_key_here

# Groq (for LLM)
GROQ_API_KEY=your_key_here

# Pinecone (for vector storage)
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=medicalindex1
```

### Pinecone Setup

1. Create a free account at [Pinecone](https://www.pinecone.io/)
2. Create a new index with:
   - Dimension: 768
   - Metric: dotproduct
   - Cloud: AWS
   - Region: us-east-1

---

##  Deployment

### Deploy to Render

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Configure build settings:
   - **Build Command**: `pip install -r server/requirements.txt`
   - **Start Command**: `uvicorn server.main:app --host 0.0.0.0 --port 10000`
4. Add environment variables in Render dashboard
5. Deploy!

---

##  Usage

1. **Upload Documents**: Use the web interface to upload medical PDFs
2. **Wait for Processing**: Documents are chunked, embedded, and stored
3. **Ask Questions**: Type medical queries in natural language
4. **Get Answers**: Receive accurate responses with source references

### Example Queries

- "What are the symptoms of Type 2 diabetes?"
- "How is hypertension diagnosed?"
- "What are the side effects of metformin?"
- "Explain the treatment protocol for acute myocardial infarction"

---

##  How It Works

1. **Document Upload**: PDFs are uploaded via the web interface
2. **Text Extraction**: PyPDF extracts text from each page
3. **Chunking**: Documents are split into 500-character chunks with 50-character overlap
4. **Embedding**: Each chunk is converted to a 768-dimensional vector using Google's embedding model
5. **Storage**: Vectors are stored in Pinecone with metadata (source, page number)
6. **Query Processing**: User questions are embedded using the same model
7. **Retrieval**: Top 3 most similar chunks are retrieved from Pinecone
8. **Generation**: Retrieved context is passed to LLaMA 3.3 70B for answer generation
9. **Response**: User receives answer with source document references


**Built with ❤️ using LangChain, FastAPI, and Streamlit**
