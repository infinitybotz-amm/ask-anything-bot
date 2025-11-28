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
    return FileResponse('../frontend/index.html')

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

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if request.model == "gemini":
            if not GOOGLE_API_KEY:
                raise HTTPException(status_code=500, detail="Google API Key not configured")
            
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(request.message)
            return {"response": response.text}
            
        elif request.model == "openai":
            if not OPENAI_API_KEY:
                raise HTTPException(status_code=500, detail="OpenAI API Key not configured")
            
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o", # Or gpt-3.5-turbo
                messages=[{"role": "user", "content": request.message}]
            )
            return {"response": response.choices[0].message.content}
            
        else:
            raise HTTPException(status_code=400, detail="Invalid model selected")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
