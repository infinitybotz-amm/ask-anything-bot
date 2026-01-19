from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import openai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
# In standalone mode, main.py is in backend/, so root is parent.parent
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ... (imports remain)
from fastapi import File, UploadFile
import shutil
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Global Vector Store
vector_store = None

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
@app.get("/")
async def read_index():
    return FileResponse(BASE_DIR / 'frontend' / 'index.html')

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Configure OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

class ChatRequest(BaseModel):
    model: str
    message: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_store
    try:
        # Save file temporarily
        temp_file_path = BASE_DIR / "temp_upload" / file.filename
        temp_file_path.parent.mkdir(exist_ok=True)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Load document based on extension
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(str(temp_file_path))
        elif file.filename.endswith(".docx"):
            loader = Docx2txtLoader(str(temp_file_path))
        else:
            loader = TextLoader(str(temp_file_path))
            
        documents = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Create Embeddings (Default to Gemini for free tier, fallback to OpenAI if needed)
        if GOOGLE_API_KEY:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        elif OPENAI_API_KEY:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        else:
            raise HTTPException(status_code=500, detail="No API Key available for embeddings")
            
        # Create Vector Store
        vector_store = FAISS.from_documents(texts, embeddings)
        
        # Cleanup
        os.remove(temp_file_path)
        
        return {"status": "success", "filename": file.filename, "chunks": len(texts)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    global vector_store
    try:
        # Validate model selection
        valid_models = ["gemini", "gemini-flash", "openai"]
        if request.model not in valid_models:
            raise HTTPException(status_code=400, detail=f"Invalid model. Valid options are: {', '.join(valid_models)}")
        
        # Check if RAG is possible (vector store exists)
        if vector_store:
            retriever = vector_store.as_retriever()
            
            if request.model == "gemini":
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
            elif request.model == "gemini-flash":
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
            elif request.model == "openai":
                llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
            else:
                raise HTTPException(status_code=400, detail="Invalid model")
                
            prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS format your answer using Markdown.
- Use **Bold** for key terms.
- Use ### Headers for sections.
- Ensure clear spacing between paragraphs.

Context: {context}

Question: {question}
Answer:"""
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            chain_type_kwargs = {"prompt": PROMPT}
                
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=retriever,
                chain_type_kwargs=chain_type_kwargs
            )
            response = qa_chain.run(request.message)
            return {"response": response}

        # Fallback to normal chat if no document loaded
        if request.model == "gemini":
            if not GOOGLE_API_KEY:
                raise HTTPException(status_code=500, detail="Google API Key not configured")
            
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(request.message + "\n\nPlease format your response using Markdown. Use **Bold** for key terms and ### Headers for sections.")
            return {"response": response.text}
        
        elif request.model == "gemini-flash":
            if not GOOGLE_API_KEY:
                raise HTTPException(status_code=500, detail="Google API Key not configured")
            
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(request.message + "\n\nPlease format your response using Markdown. Use **Bold** for key terms and ### Headers for sections.")
            return {"response": response.text}
            
        elif request.model == "openai":
            if not OPENAI_API_KEY:
                raise HTTPException(status_code=500, detail="OpenAI API Key not configured")
            
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": request.message + "\n\nPlease format your response using Markdown. Use **Bold** for key terms and ### Headers for sections."}]
            )
            return {"response": response.choices[0].message.content}
            
        else:
            raise HTTPException(status_code=400, detail="Invalid model selected")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
