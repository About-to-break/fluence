import json
import logging

from .lamport import LamportClock
from .models import JobStatus
from .store import InMemoryJobStore


def apply_output_payload(
    job_store: InMemoryJobStore,
    clock: LamportClock,
    payload: dict,
):
    job_uuid = payload.get("uuid")
    if not isinstance(job_uuid, str) or job_uuid == "":
        logging.warning("Skipping q.out_gateway payload without uuid")
        return None

    existing_job = job_store.get(job_uuid)
    if existing_job is None:
        logging.warning("Skipping q.out_gateway payload for unknown job %s", job_uuid)
        return None

    remote_lamport_ts = payload.get("lamport_ts")
    local_lamport_ts = None
    if isinstance(remote_lamport_ts, int):
        local_lamport_ts = clock.update(remote_lamport_ts)

    error_detail = payload.get("error_detail") or payload.get("error")
    status_value = payload.get("status")
    if isinstance(status_value, str) and status_value.lower() in {"failed", "error"}:
        return job_store.set_result(
            job_uuid,
            JobStatus.FAILED,
            lamport_ts=local_lamport_ts,
            translated_text=None,
            error_detail=str(error_detail or "Translation failed"),
        )

    translated_text = payload.get("text")
    if translated_text is None:
        translated_text = payload.get("translated_text")

    if isinstance(translated_text, str) and translated_text != "":
        return job_store.set_result(
            job_uuid,
            JobStatus.COMPLETED,
            lamport_ts=local_lamport_ts,
            translated_text=translated_text,
            error_detail=None,
        )

    if error_detail is not None:
        return job_store.set_result(
            job_uuid,
            JobStatus.FAILED,
            lamport_ts=local_lamport_ts,
            translated_text=None,
            error_detail=str(error_detail),
        )

    logging.warning("Skipping unsupported q.out_gateway payload for job %s", job_uuid)
    return None


async def _safe_ack(message) -> None:
    ack = getattr(message, "ack", None)
    if ack is None:
        return

    await ack()


async def _safe_nack(message, requeue: bool = True) -> None:
    nack = getattr(message, "nack", None)
    if nack is None:
        return

    await nack(requeue=requeue)


async def process_output_message(
    message,
    job_store: InMemoryJobStore,
    clock: LamportClock,
):
    try:
        payload = json.loads(message.body.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        logging.warning("Skipping malformed q.out_gateway payload")
        await _safe_ack(message)
        return None

    try:
        result = apply_output_payload(job_store, clock, payload)
    except Exception:
        logging.exception("Failed to process q.out_gateway payload")
        await _safe_nack(message, requeue=True)
        return None

    await _safe_ack(message)
    return result
