"""Validation schemas and error formatting for API payloads."""

from typing import Any

from src.config import MAX_BATCH_SIZE, MAX_CHAR_LENGTH


def validate_predict_payload(data: Any) -> tuple[bool, str | None]:
    """Validate single prediction payload.

    Rules:
    - Must be a dictionary
    - Must contain 'text' key
    - 'text' must be a string
    - 'text' cannot be empty or whitespace-only
    - 'text' length must not exceed 5000 characters
    """
    if not isinstance(data, dict):
        return False, "Request body must be a JSON object"

    if "text" not in data:
        return False, "Missing required field: 'text'"

    text_val = data["text"]
    if not isinstance(text_val, str):
        return False, "'text' must be a string"

    text_stripped = text_val.strip()
    if not text_stripped:
        return False, "'text' must be a non-empty string"

    if len(text_val) > MAX_CHAR_LENGTH:
        return False, f"'text' exceeds maximum length of {MAX_CHAR_LENGTH} characters"

    return True, None


def validate_batch_payload(data: Any) -> tuple[bool, str | None]:
    """Validate batch prediction payload.

    Rules:
    - Must be a dictionary
    - Must contain 'texts' key
    - 'texts' must be a list
    - 'texts' length must be between 1 and 100
    - Each text item must be a string <= 5000 chars and non-empty
    """
    if not isinstance(data, dict):
        return False, "Request body must be a JSON object"

    if "texts" not in data:
        return False, "Missing required field: 'texts'"

    texts_val = data["texts"]
    if not isinstance(texts_val, list):
        return False, "'texts' must be a list of strings"

    if len(texts_val) == 0:
        return False, "'texts' cannot be empty"

    if len(texts_val) > MAX_BATCH_SIZE:
        return (
            False,
            f"'texts' array length ({len(texts_val)}) exceeds limit of {MAX_BATCH_SIZE}",
        )

    for idx, item in enumerate(texts_val):
        if not isinstance(item, str):
            return False, f"Item at index {idx} must be a string"
        if not item.strip():
            return False, f"Item at index {idx} cannot be empty"
        if len(item) > MAX_CHAR_LENGTH:
            return False, f"Item at index {idx} exceeds {MAX_CHAR_LENGTH} characters"

    return True, None


def make_error_response(
    code: str, message: str, status_code: int = 422
) -> tuple[dict[str, Any], int]:
    """Format standardized error response matching API.md."""
    return {
        "error": {
            "code": code,
            "message": message,
        }
    }, status_code
