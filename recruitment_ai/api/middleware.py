"""Error handling, request logging, and rate limiting middleware."""
import time
import traceback
import logging
from collections import defaultdict
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from recruitment_ai.config.settings import settings

logger = logging.getLogger(__name__)


async def error_handler_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        error_id = str(hash(str(e)))[:8]
        logger.error(f"Error {error_id}: {traceback.format_exc()}")

        if settings.DEBUG:
            return JSONResponse(
                status_code=500,
                content={
                    "error": str(e),
                    "error_id": error_id,
                    "traceback": traceback.format_exc().split("\n")[-5:],
                },
            )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "error_id": error_id,
                "message": "An unexpected error occurred. Please try again.",
            },
        )


async def request_logger_middleware(request: Request, call_next):
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} -> {response.status_code}")
    return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.window = settings.RATE_LIMIT_WINDOW_SECONDS
        self.max_requests = settings.RATE_LIMIT_REQUESTS
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - self.window

        timestamps = self.requests[client_ip]
        timestamps[:] = [t for t in timestamps if t > window_start]

        if len(timestamps) >= self.max_requests:
            logger.warning("Rate limit exceeded for %s", client_ip)
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limit_exceeded", "message": "Too many requests. Please slow down."},
                headers={"Retry-After": str(int(self.window))},
            )

        timestamps.append(now)
        return await call_next(request)
