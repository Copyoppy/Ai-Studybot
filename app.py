import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import SecretStr

# ---------------------------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "studybot")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chat_history")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

# ---------------------------------------------------------------------------
# 2. MongoDB setup
# ---------------------------------------------------------------------------
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[MONGODB_DB_NAME]
chat_collection = db[MONGODB_COLLECTION]

# ---------------------------------------------------------------------------
# 3. LLM setup (Groq via LangChain)
# ---------------------------------------------------------------------------
llm = ChatGroq(
    api_key=SecretStr(GROQ_API_KEY),
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

# ---------------------------------------------------------------------------
# 4. System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are 'StudyBot', a friendly, encouraging, and knowledgeable AI study "
    "partner. Your primary goal is to empower users in their learning journey.\n\n"
    "Your core responsibilities are:\n"
    "1.  **Answer Questions with Clarity:** Break down complex topics into their "
    "core components. Use structured formats like lists, bullet points, and "
    "step-by-step guides to enhance clarity. Your explanations should be "
    "tailored to a student audience—avoid jargon where possible, or explain it "
    "clearly if it's essential.\n"
    "2.  **Explain with Insight:** Don't just define concepts; explain them with "
    "intuition. Use real-world examples, historical context, or creative "
    "analogies to make abstract ideas tangible and memorable.\n"
    "3.  **Encourage and Motivate:** Maintain a positive and encouraging tone. "
    "Start responses with phrases like 'That's a great question!' or 'Excellent "
    "topic!'. Frame learning as an exciting journey.\n"
    "4.  **Promote Deeper Learning:** After answering a question, proactively "
    "suggest 2-3 related questions or advanced topics. Frame them as 'Next "
    "Steps' or 'Curious to learn more?'. For example: 'Now that we've "
    "covered photosynthesis, you might find it interesting to explore how it "
    "compares to cellular respiration.'\n"
    "5.  **Stay Focused on Academics:** Your expertise is in academic subjects "
    "(math, science, literature, history, etc.). If a user asks for personal "
    "advice, opinions on non-academic matters, or harmful content, gently "
    "decline and guide them back. For example: 'That's an interesting "
    "thought! However, my purpose is to help with your studies. Do you have "
    "any academic questions I can assist you with?'\n"
    "6.  **Leverage Conversation History:** Pay close attention to the "
    "conversation context to provide answers that are relevant and build "
    "upon what has already been discussed. Avoid repeating information and "
    "instead, connect new concepts to previous ones."
)

# ---------------------------------------------------------------------------
# 5. Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str


class MessageOut(BaseModel):
    role: str
    content: str
    timestamp: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[MessageOut]

# ---------------------------------------------------------------------------
# 6. Database helper functions
# ---------------------------------------------------------------------------

MAX_HISTORY_MESSAGES = 10  # number of past messages to include as context


def save_message(session_id: str, role: str, content: str) -> None:
    """Persist a single chat message to MongoDB."""
    chat_collection.insert_one(
        {
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc),
        }
    )


def get_chat_history(session_id: str, limit: int = MAX_HISTORY_MESSAGES) -> list[dict]:
    """Return the most recent messages for a session, ordered oldest-first."""
    cursor = (
        chat_collection.find(
            {"session_id": session_id},
            {"_id": 0, "role": 1, "content": 1, "timestamp": 1},
        )
        .sort("timestamp", -1)
        .limit(limit)
    )
    messages = list(cursor)
    messages.reverse()  # oldest first
    return messages


def clear_chat_history(session_id: str) -> int:
    """Delete all messages for a session and return the count deleted."""
    result = chat_collection.delete_many({"session_id": session_id})
    return result.deleted_count


def history_to_langchain_messages(history: list[dict]) -> list:
    """Convert stored history dicts into LangChain message objects."""
    lc_messages: list = []
    for msg in history:
        if msg["role"] == "human":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages

# ---------------------------------------------------------------------------
# 7. FastAPI application
# ---------------------------------------------------------------------------
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Verify MongoDB connection on application startup."""
    try:
        mongo_client.admin.command("ping")
        print("Connected to MongoDB successfully.")
    except Exception as exc:
        print(f"MongoDB connection failed: {exc}")
    yield


app = FastAPI(
    title="Study Bot API",
    description="AI-powered study assistant with conversation memory.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Routes ----------------------------------------------------------------

@app.get("/")
async def root():
    """Health-check / welcome endpoint."""
    return {
        "message": "Welcome to the Study Bot App!",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a study question and receive an AI-generated answer."""
    try:
        # 1. Retrieve previous context
        history = get_chat_history(request.session_id)
        lc_history = history_to_langchain_messages(history)

        # 2. Build the full message list
        messages = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + lc_history
            + [HumanMessage(content=request.message)]
        )

        # 3. Call the LLM
        ai_response = llm.invoke(messages)
        response_text = str(ai_response.content)

        # 4. Persist both messages
        save_message(request.session_id, "human", request.message)
        save_message(request.session_id, "ai", response_text)

        return ChatResponse(session_id=request.session_id, response=response_text)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Retrieve conversation history for a session."""
    history = get_chat_history(session_id, limit=50)
    messages_out = [
        MessageOut(
            role=m["role"],
            content=m["content"],
            timestamp=m["timestamp"].isoformat() if isinstance(m["timestamp"], datetime) else str(m["timestamp"]),
        )
        for m in history
    ]
    return HistoryResponse(session_id=session_id, messages=messages_out)


@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    """Clear conversation history for a session."""
    deleted = clear_chat_history(session_id)
    return {"session_id": session_id, "deleted_count": deleted}

# ---------------------------------------------------------------------------
# 8. Entry point — interactive terminal chat
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Check for --api flag to start the FastAPI server instead
    if "--api" in sys.argv:
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    else:
        # Interactive terminal chat mode
        session_id = "terminal_session"

        print("=" * 60)
        print("  STUDY BOT — AI-Powered Study Assistant")
        print("=" * 60)
        print()
        print("Send a study question and receive an AI-generated answer:")
        print("(Type 'quit' or 'exit' to stop)\n")

        try:
            mongo_client.admin.command("ping")
            print("Connected to MongoDB successfully.\n")
        except Exception as exc:
            print(f"MongoDB connection failed: {exc}")
            print("Chat will work but history won't be saved.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye! Happy studying!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("\nGoodbye! Happy studying!")
                break

            # Build messages with history
            history = get_chat_history(session_id)
            lc_history = history_to_langchain_messages(history)
            messages = (
                [SystemMessage(content=SYSTEM_PROMPT)]
                + lc_history
                + [HumanMessage(content=user_input)]
            )

            print("\nThinking...\n")

            try:
                ai_response = llm.invoke(messages)
                response_text = str(ai_response.content)

                # Save to MongoDB
                save_message(session_id, "human", user_input)
                save_message(session_id, "ai", response_text)

                print(f"Study Bot: {response_text}\n")
            except Exception as exc:
                print(f"Error: {exc}\n")