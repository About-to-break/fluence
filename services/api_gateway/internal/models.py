from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    ACCEPTED = "accepted"
    PUBLISHED = "published"
    PUBLISH_FAILED = "publish_failed"
    COMPLETED = "completed"
    FAILED = "failed"


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1)
    force_fast: bool = False


class TranslateAcceptedResponse(BaseModel):
    uuid: str
    lamport_ts: int
    status: JobStatus
    force_fast: bool


class TranslateStatusResponse(BaseModel):
    uuid: str
    lamport_ts: int
    status: JobStatus
    text: str
    force_fast: bool
    translated_text: str | None = None
    error_detail: str | None = None


class HealthResponse(BaseModel):
    status: str


@dataclass(slots=True)
class JobRecord:
    uuid: str
    lamport_ts: int
    text: str
    force_fast: bool
    status: JobStatus
    translated_text: str | None = None
    error_detail: str | None = None
