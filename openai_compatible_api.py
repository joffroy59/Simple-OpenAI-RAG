import os
import time
import uuid
import json
from typing import Any
from typing import Literal

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic import Field

from rag_pipeline import create_rag_chain


app = FastAPI(title="RAG OpenAI-Compatible API", version="1.0.0")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    temperature: float | None = None
    stream: bool = False


def _resolve_model_name() -> str:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "azure":
        return os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "azure-chat")
    return os.getenv("LLM_MODEL", "gpt-3.5-turbo")


def _extract_user_query(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    raise HTTPException(status_code=400, detail="At least one user message is required.")


def _split_text_for_stream(answer: str, chunk_size: int = 40) -> list[str]:
    if not answer:
        return []

    chunks = []
    current = ""
    for word in answer.split(" "):
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current + " ")
            current = word

    if current:
        chunks.append(current)
    return chunks


def _build_completion_response(
    answer: str,
    source_files: list[str],
    model_name: str,
    completion_id: str,
    created: int,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "rag_sources": source_files,
    }


def _stream_chat_completion(
    answer: str,
    source_files: list[str],
    model_name: str,
    completion_id: str,
    created: int,
):
    role_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    for text_chunk in _split_text_for_stream(answer):
        content_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text_chunk},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(content_chunk)}\n\n"

    finish_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
        "rag_sources": source_files,
    }
    yield f"data: {json.dumps(finish_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    model_name = _resolve_model_name()
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": now,
                "owned_by": "rag-backend",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(payload: ChatCompletionsRequest):
    query = _extract_user_query(payload.messages)

    qa_chain = create_rag_chain()
    rag_response = qa_chain({"query": query})

    answer = rag_response.get("result", "")
    source_documents = rag_response.get("source_documents", [])
    source_files = []
    for doc in source_documents:
        source_path = doc.metadata.get("source") if doc.metadata else None
        if source_path:
            source_files.append(source_path)

    model_name = payload.model or _resolve_model_name()
    now = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if payload.stream:
        return StreamingResponse(
            _stream_chat_completion(
                answer=answer,
                source_files=source_files,
                model_name=model_name,
                completion_id=completion_id,
                created=now,
            ),
            media_type="text/event-stream",
        )

    return _build_completion_response(
        answer=answer,
        source_files=source_files,
        model_name=model_name,
        completion_id=completion_id,
        created=now,
    )
