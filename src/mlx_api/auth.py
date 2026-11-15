"""Bearer-key auth on every path except ``/health``."""

from __future__ import annotations

import secrets

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from mlx_api.errors import ContractError
from mlx_api.schemas import error_response

OPEN_PATHS = frozenset({"/health"})


class AuthError(ContractError):
    status = 401
    type = "auth"


def check_bearer(header: str | None, api_key: str) -> None:
    scheme, _, token = (header or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("missing bearer token", code="missing_api_key")
    if not secrets.compare_digest(token.strip().encode(), api_key.encode()):
        raise AuthError("invalid API key", code="invalid_api_key")


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, api_key: str) -> None:
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path not in OPEN_PATHS:
            try:
                check_bearer(request.headers.get("authorization"), self.api_key)
            except AuthError as exc:
                return error_response(exc.status, exc.message, exc.type, exc.code)
        return await call_next(request)
