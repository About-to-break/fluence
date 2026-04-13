from threading import Lock

from .models import JobRecord, JobStatus


class InMemoryJobStore:
    def __init__(self):
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()

    def create(self, job: JobRecord) -> JobRecord:
        with self._lock:
            self._jobs[job.uuid] = job
            return self._jobs[job.uuid]

    def get(self, job_uuid: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_uuid)

    def set_status(
        self,
        job_uuid: str,
        status: JobStatus,
        error_detail: str | None = None,
    ) -> JobRecord:
        with self._lock:
            job = self._jobs[job_uuid]
            job.status = status
            job.error_detail = error_detail
            return job

    def set_result(
        self,
        job_uuid: str,
        status: JobStatus,
        lamport_ts: int | None = None,
        translated_text: str | None = None,
        error_detail: str | None = None,
    ) -> JobRecord:
        with self._lock:
            job = self._jobs[job_uuid]
            job.status = status
            if lamport_ts is not None:
                job.lamport_ts = lamport_ts
            job.translated_text = translated_text
            job.error_detail = error_detail
            return job
