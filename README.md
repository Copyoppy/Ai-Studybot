# StudyBot - AI-Powered Study Assistant

An intelligent study assistant API built with **FastAPI**, **LangChain**, **Groq (Llama 3.3)**, and **MongoDB**. StudyBot answers academic questions with clarity, provides real-world examples, and maintains conversation history for personalized learning sessions.

## Features

- **AI-Powered Responses** — Uses Llama 3.3 70B via Groq for fast, high-quality answers
- **Conversation Memory** — Session-based chat history stored in MongoDB
- **REST API** — Clean FastAPI endpoints for chat, history retrieval, and session management
- **Interactive Terminal Mode** — Chat directly from the command line
- **CORS Enabled** — Ready for frontend integration

## Tech Stack

- **FastAPI** — Web framework
- **LangChain + Groq** — LLM orchestration
- **MongoDB** — Chat history persistence
- **Pydantic** — Data validation
- **Uvicorn** — ASGI server

## Getting Started

### Prerequisites

- Python 3.10+
- MongoDB (local or cloud via MongoDB Atlas)
- [Groq API Key](https://console.groq.com/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/studybot.git
   cd studybot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key
   MONGODB_URI=mongodb://localhost:27017
   MONGODB_DB_NAME=studybot
   MONGODB_COLLECTION=chat_history
   ```

### Running the App

**Start the API server:**
```bash
uvicorn app:app --reload
```
The API will be available at `http://127.0.0.1:8000` and interactive docs at `http://127.0.0.1:8000/docs`.

**Or use the interactive terminal chat:**
```bash
python app.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/chat` | Send a message and get an AI response |
| `GET` | `/history/{session_id}` | Retrieve chat history for a session |
| `DELETE` | `/history/{session_id}` | Clear chat history for a session |

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user1", "message": "Explain photosynthesis"}'
```

## License

This project was created by Kingsford Sunday Ntim and is open source and available under the [MIT License](LICENSE).
