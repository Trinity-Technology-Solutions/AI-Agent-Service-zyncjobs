"""Error handling middleware for the API."""
from fastapi import Request
from fastapi.responses import JSONResponse
import traceback
import logging
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
