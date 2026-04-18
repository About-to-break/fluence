import asyncio
import orjson
import logging
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .lamport import LamportClock
from .models import (
    HealthResponse,
    JobRecord,
    JobStatus,
    TranslateAcceptedResponse,
    TranslateRequest,
    TranslateStatusResponse,
)
from .queue import MessageConsumer, MessageProducer, get_consumer, get_producer
from .store import InMemoryJobStore


def _resolve_consumer(
    config: SimpleNamespace,
    consumer: MessageConsumer | None,
) -> MessageConsumer | None:
    if consumer is not None:
        return consumer

    required_fields = (
        "MESSAGE_BROKER",
        "RABBITMQ_URI",
        "RABBITMQ_OUTPUT_QUEUE",
    )
    if not all(hasattr(config, field) for field in required_fields):
        return None

    return get_consumer(config)


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

    translated_text = payload.get("translated_text")
    if translated_text is None:
        translated_text = payload.get("text")

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


def create_app(
    config: SimpleNamespace,
    producer: MessageProducer | None = None,
    consumer: MessageConsumer | None = None,
    store: InMemoryJobStore | None = None,
    lamport_clock: LamportClock | None = None,
) -> FastAPI:
    job_store = store or InMemoryJobStore()
    clock = lamport_clock or LamportClock()
    message_producer = producer or get_producer(config)
    message_consumer = _resolve_consumer(config, consumer)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        consumer_task = None

        async def handle_output_message(message, **kwargs):
            try:
                payload = orjson.loads(message.body.decode("utf-8"))
            except (AttributeError, UnicodeDecodeError, orjson.JSONDecodeError):
                logging.warning("Skipping malformed q.out_gateway payload")
                return

            apply_output_payload(job_store, clock, payload)

        async def run_output_consumer():
            if message_consumer is None:
                return

            try:
                await message_consumer.start_consuming(handle_output_message)
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("q.out_gateway consumer stopped unexpectedly")

        app.state.config = config
        app.state.store = job_store
        app.state.lamport_clock = clock
        app.state.producer = message_producer
        app.state.consumer = message_consumer
        if message_consumer is not None:
            consumer_task = asyncio.create_task(run_output_consumer())
            app.state.consumer_task = consumer_task
        yield
        if consumer_task is not None:
            consumer_task.cancel()
            await message_consumer.close()
            with suppress(asyncio.CancelledError):
                await consumer_task
        await message_producer.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz():
        return HealthResponse(status="ok")

    @app.post("/v1/translate", response_model=TranslateAcceptedResponse, status_code=202)
    async def translate(request: TranslateRequest):
        job_uuid = str(uuid4())
        lamport_ts = clock.next()

        job = JobRecord(
            uuid=job_uuid,
            lamport_ts=lamport_ts,
            text=request.text,
            force_fast=request.force_fast,
            status=JobStatus.ACCEPTED,
        )
        job_store.create(job)

        message = {
            "uuid": job_uuid,
            "lamport_ts": lamport_ts,
            "text": request.text,
            "force_fast": request.force_fast,
        }

        try:
            await message_producer.produce(message)
        except Exception as exc:
            logging.exception("Failed to publish translation job %s", job_uuid)
            failed_job = job_store.set_status(
                job_uuid,
                JobStatus.PUBLISH_FAILED,
                error_detail=str(exc),
            )
            return JSONResponse(
                status_code=503,
                content={
                    "uuid": failed_job.uuid,
                    "lamport_ts": failed_job.lamport_ts,
                    "status": failed_job.status.value,
                    "force_fast": failed_job.force_fast,
                    "detail": "Failed to publish translation job",
                },
            )

        published_job = job_store.set_status(job_uuid, JobStatus.PUBLISHED)

        return TranslateAcceptedResponse(
            uuid=published_job.uuid,
            lamport_ts=published_job.lamport_ts,
            status=published_job.status,
            force_fast=published_job.force_fast,
        )

    @app.get(
        "/v1/translate/{job_uuid}",
        response_model=TranslateStatusResponse,
        response_model_exclude_none=True,
    )
    async def get_translate_job(job_uuid: str):
        job = job_store.get(job_uuid)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        return TranslateStatusResponse(
            uuid=job.uuid,
            lamport_ts=job.lamport_ts,
            status=job.status,
            text=job.text,
            force_fast=job.force_fast,
            translated_text=job.translated_text,
            error_detail=job.error_detail,
        )

    return app
