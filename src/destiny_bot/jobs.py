"""Background jobs (importing, exporting, downloading voices) with progress reporting.

A job function receives its :class:`Job` handle, reports progress with
``job.update(fraction, message)`` and checks ``job.cancelled`` (or calls
``job.raise_if_cancelled()``) to stop early. Whatever it returns becomes the job's
``result``; an exception becomes a friendly ``error``.
"""

from __future__ import annotations

import logging
import secrets
import shutil
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Literal

from .ingest import IngestError
from .tts.base import TTSError

log = logging.getLogger(__name__)

JobStatus = Literal["running", "done", "error", "cancelled"]

RETENTION_SECONDS = 3600.0
MAX_FINISHED_JOBS = 200
_ERROR_DETAIL_CHARS = 200

# Exceptions defined in modules that may not be imported yet; their messages are written
# for the user. They are looked up in sys.modules, so checking never imports anything.
_FRIENDLY_ELSEWHERE = (
    ("destiny_bot.ai", "AIUnavailable"),
    ("destiny_bot.ai", "AIError"),
    ("destiny_bot.tts.catalog", "VoiceDownloadError"),
    ("destiny_bot.tts.catalog", "DownloadCancelled"),
    ("destiny_bot.export", "ExportCancelled"),
)


class JobCancelled(Exception):
    """Raised inside a job function to stop because the user cancelled it."""


class JobError(RuntimeError):
    """A job failure whose message is safe to show to the user as is."""


def _friendly_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [JobError, IngestError, TTSError]
    for module_name, attr in _FRIENDLY_ELSEWHERE:
        cls = getattr(sys.modules.get(module_name), attr, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            types.append(cls)
    return tuple(types)


def is_friendly_error(exc: BaseException) -> bool:
    """True if the exception's message was written for the user."""
    return isinstance(exc, _friendly_types())


def friendly_error(exc: BaseException) -> str:
    """A message for the UI: friendly errors pass through, anything else gets a short summary."""
    text = str(exc).strip()
    if is_friendly_error(exc) and text:
        return text
    short = text.splitlines()[0] if text else type(exc).__name__
    if len(short) > _ERROR_DETAIL_CHARS:
        short = short[: _ERROR_DETAIL_CHARS - 1].rstrip() + "…"
    return f"Something went wrong: {short}"


class Job:
    """One background task. Attributes are updated from the worker thread; use :meth:`to_dict` to read them."""

    def __init__(self, kind: str, *, files_dir: Path | None = None, message: str = "") -> None:
        self.id = "j_" + secrets.token_hex(8)
        self.kind = kind
        self.status: JobStatus = "running"
        self.progress = 0.0
        self.message = message
        self.error: str | None = None
        self.result: Any = None
        self.cancel_event = threading.Event()
        self.created_at = time.time()
        self.finished_at: float | None = None
        self.files_dir = files_dir
        self._cleanups: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._future: Future[None] | None = None

    @property
    def cancelled(self) -> bool:
        """True once the user asked to cancel."""
        return self.cancel_event.is_set()

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    def update(self, progress: float | None = None, message: str | None = None) -> None:
        """Report progress (0..1) and/or a status message. Usable as an ingest/export progress callback."""
        with self._lock:
            if self.status != "running":
                return
            if progress is not None:
                try:
                    value = float(progress)
                except (TypeError, ValueError):
                    value = self.progress
                if value == value:  # skip NaN
                    self.progress = min(1.0, max(0.0, value))
            if message is not None:
                self.message = str(message)

    def raise_if_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled()

    def add_cleanup(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` once the job finishes, whatever the outcome (even if it never started)."""
        self._cleanups.append(fn)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the job finishes; returns False on timeout."""
        return self._done.wait(timeout)

    def to_dict(self) -> dict[str, Any]:
        """The Job shape from docs/API.md."""
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "progress": round(self.progress, 4),
                "message": self.message,
                "error": self.error,
                "result": self.result,
            }

    def _finish(
        self,
        status: JobStatus,
        *,
        result: Any = None,
        error: str | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            if self._done.is_set():
                return
            self.status = status
            self.result = result
            self.error = error
            if message is not None:
                self.message = message
            if status == "done":
                self.progress = 1.0
            self.finished_at = time.time()
        if status != "done" and self.files_dir is not None:
            # Partial output of a failed or cancelled job is useless; don't leave it on disk.
            shutil.rmtree(self.files_dir, ignore_errors=True)
        for cleanup in self._cleanups:
            try:
                cleanup()
            except Exception:  # cleanup must never hide the job's own outcome
                log.exception("Cleanup for job %s failed", self.id)
        self._done.set()


class JobManager:
    """Runs jobs on a small thread pool and remembers them for an hour after they finish."""

    def __init__(self, *, max_workers: int = 2, retention_seconds: float = RETENTION_SECONDS) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="destiny-job")
        self._retention = retention_seconds
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._closed = False

    def submit(
        self,
        kind: str,
        fn: Callable[[Job], Any],
        *,
        files_root: Path | None = None,
        message: str = "Waiting to start…",
        cleanup: Callable[[], None] | None = None,
    ) -> Job:
        """Start ``fn(job)`` in the background and return the job right away.

        With ``files_root``, the job gets its own output folder ``files_root/<job id>``
        (created here) as ``job.files_dir``. ``cleanup`` always runs when the job ends.
        """
        self.prune()
        job = Job(kind, message=message)
        if cleanup is not None:
            job.add_cleanup(cleanup)
        if files_root is not None:
            job.files_dir = Path(files_root) / job.id
            job.files_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._closed:
                job._finish("cancelled", message="Destiny Bot is shutting down")
                return job
            self._jobs[job.id] = job
            job._future = self._executor.submit(self._run, job, fn)
        return job

    def get(self, job_id: str) -> Job:
        """The job with this id. Raises KeyError if it's unknown or was pruned."""
        self.prune()
        with self._lock:
            return self._jobs[job_id]

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at)

    def cancel(self, job_id: str) -> Job:
        """Ask a job to stop. A job that hasn't started yet is cancelled immediately."""
        job = self.get(job_id)
        self._cancel(job)
        return job

    def cancel_all(self, *, wait: float = 0.0) -> None:
        """Cancel every unfinished job, waiting up to ``wait`` seconds for running ones to stop."""
        jobs = self.list()
        for job in jobs:
            self._cancel(job)
        deadline = time.monotonic() + wait
        for job in jobs:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            job.wait(remaining)

    def clear(self, *, wait: float = 0.0) -> None:
        """Cancel everything and forget every job (used after erasing all data)."""
        self.cancel_all(wait=wait)
        with self._lock:
            self._jobs = {job_id: job for job_id, job in self._jobs.items() if not job.finished}

    def prune(self, now: float | None = None) -> None:
        """Forget jobs that finished more than the retention time ago (and delete their files)."""
        now = time.time() if now is None else now
        with self._lock:
            finished = sorted(
                (j for j in self._jobs.values() if j.finished_at is not None),
                key=lambda j: j.finished_at or 0.0,
            )
            expired = [j for j in finished if now - (j.finished_at or now) > self._retention]
            overflow = len(finished) - len(expired) - MAX_FINISHED_JOBS
            if overflow > 0:
                expired += [j for j in finished if j not in expired][:overflow]
            for job in expired:
                self._jobs.pop(job.id, None)
        for job in expired:
            if job.files_dir is not None:
                shutil.rmtree(job.files_dir, ignore_errors=True)

    def shutdown(self, *, wait: bool = False) -> None:
        """Cancel all jobs and stop the worker threads (called when the server stops)."""
        with self._lock:
            self._closed = True
        for job in self.list():
            self._cancel(job)
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _cancel(self, job: Job) -> None:
        if job.finished:
            return
        job.cancel_event.set()
        future = job._future
        if future is not None and future.cancel():
            job._finish("cancelled", message="Cancelled")
        else:
            job.update(message="Cancelling…")

    def _run(self, job: Job, fn: Callable[[Job], Any]) -> None:
        if job.cancelled:
            job._finish("cancelled", message="Cancelled")
            return
        job.update(message="Starting…")
        try:
            result = fn(job)
        except Exception as exc:
            if job.cancelled or isinstance(exc, JobCancelled):
                job._finish("cancelled", message="Cancelled")
                return
            message = friendly_error(exc)
            if is_friendly_error(exc):  # expected (bad file, no voice, ...): the UI shows the message
                log.info("%s job %s failed: %s", job.kind, job.id, exc)
            else:
                log.error("%s job %s failed", job.kind, job.id, exc_info=exc)
            job._finish("error", error=message, message=message)
            return
        if job.cancelled:
            job._finish("cancelled", message="Cancelled")
        else:
            job._finish("done", result=result, message="Done")
