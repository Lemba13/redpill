"""
models.py — Pydantic request/response models for the feedback service.
"""

from pydantic import BaseModel, field_validator


class VoteRequest(BaseModel):
    """Body accepted by POST /api/vote."""

    item_id: str
    vote: str

    @field_validator("vote")
    @classmethod
    def vote_must_be_valid(cls, v: str) -> str:
        if v not in ("up", "down"):
            raise ValueError("vote must be 'up' or 'down'")
        return v

    @field_validator("item_id")
    @classmethod
    def item_id_must_be_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("item_id must not be empty")
        return v.strip()
