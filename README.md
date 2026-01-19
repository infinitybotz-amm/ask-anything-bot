# Ask Anything Chatbot 🤖

A powerful, AI-driven chatbot application capable of answering questions using Google's Gemini or OpenAI's GPT-4 models. It features **RAG (Retrieval-Augmented Generation)** capabilities, allowing users to upload documents (PDF, DOCX) and ask questions based on their content.

![Infinity Botz](frontend/infinity-botz.png)

## 🚀 Features

-   **Multi-Model Support**: Switch seamlessly between **Google Gemini** (Flash models) and **OpenAI GPT-4**.
-   **Document Analysis (RAG)**: Upload **PDF** or **DOCX** files to chat with your documents. The bot uses FAISS vector search to find relevant context.
-   **Rich Text Formatting**: Responses are formatted with **Markdown**, supporting **Bold** terms, Headers, and structured lists for better readability.
-   **Modern UI**: A clean, responsive React-based frontend with a professional design.

## 🛠️ Tech Stack

-   **Backend**: FastAPI (Python), LangChain, FAISS
-   **Frontend**: React (Embedded), Vanilla CSS, Marked.js
-   **AI Models**: Google Gemini, OpenAI GPT-4

## 📋 Prerequisites

-   Python 3.10+
-   API Keys for Google Gemini and/or OpenAI

## ⚙️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/infinitybotz-amm/ask-anything-bot.git
    cd ask-anything-bot/ask-anything
    ```

2.  **Set up Virtual Environment (Optional but Recommended)**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a `.env` file in the `ask-anything` directory and add your API keys:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    OPENAI_API_KEY=your_openai_api_key_here
    ```

## ▶️ Running the Application

You can start the application using the provided script:

```bash
# Using the helper script (Mac/Linux)
bash run_backend.sh

# OR directly with Python
cd backend
python3 -m uvicorn main:app --reload --port 8000
```

Once running, open your browser and navigate to:
👉 **http://localhost:8000**

## 💡 Usage

1.  **Select Model**: Choose between Gemini or GPT-4 from the dropdown menu.
2.  **Chat**: Type your question in the input box and hit Enter.
3.  **Upload Document**: Click "Upload Document" to analyze a PDF or Word file. Once loaded, the bot will answer questions specifically based on that document.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
