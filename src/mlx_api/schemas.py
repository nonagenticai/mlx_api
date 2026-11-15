"""Request bodies (provider wire contract: see README).

Size limits that must answer 413 (not 400) are checked in the routes, not here, because a pydantic
length violation becomes a 400 validation error.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

ErrorType = Literal["auth", "invalid_request", "retryable", "not_found"]


def error_type_for(status: int) -> ErrorType:
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status in (408, 429) or status >= 500:
        return "retryable"
    return "invalid_request"


def error_response(
    status: int, message: str, type_: str | None = None, code: str | None = None
) -> JSONResponse:
    body = {"error": {"message": message, "type": type_ or error_type_for(status), "code": code}}
    return JSONResponse(body, status_code=status)


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    input: str | list[str]
    dimensions: int | None = None
    encoding_format: Literal["float", "base64"] = "float"
    input_type: Literal["query", "document"] | None = None
    user: str | None = None


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    query: str = Field(min_length=1)
    documents: list[str]
    top_n: int | None = Field(default=None, ge=1)


class DocumentURL(BaseModel):
    type: Literal["document_url"]
    document_url: str


class ImageURL(BaseModel):
    type: Literal["image_url"]
    image_url: str


class OcrRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    document: DocumentURL | ImageURL = Field(discriminator="type")
    pages: list[int] | None = None
    include_image_base64: bool = False


class ImageURLRef(BaseModel):
    url: str


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURLRef


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant"]
    content: str | list[Annotated[TextPart | ImagePart, Field(discriminator="type")]]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False
    response_format: dict[str, Any] | None = None
