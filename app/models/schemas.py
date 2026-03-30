from pydantic import BaseModel
from typing import List, Tuple

class ChatRequest(BaseModel):
    session_id: str
    question: str
    chat_history: List[Tuple[str, str]] = []

class ChatResponse(BaseModel):
    answer: str
