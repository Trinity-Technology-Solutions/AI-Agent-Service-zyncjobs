from pydantic import BaseModel
from typing import Optional


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
    status_code: int = 500


class ValidationErrorResponse(BaseModel):
    detail: str
    errors: list[dict] = []
    status_code: int = 422
