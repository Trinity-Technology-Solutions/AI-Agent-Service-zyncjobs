"""Celery app — background task processing for long-running AI operations.
Architecture doc: AI Service → Celery Worker → Redis Broker → Task Result.

Usage:
    # In api router
    task = process_resume.delay(file_content, file_type)
    return {"task_id": task.id, "status": "pending"}

    # Poll result
    result = AsyncResult(task.id, app=celery_app)
    if result.ready():
        return result.get()
"""
import logging
from celery import Celery
from recruitment_ai.config.settings import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "zyncjobs_ai",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["recruitment_ai.services.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
)

logger.info("Celery app configured (broker=%s)", settings.REDIS_URL)
