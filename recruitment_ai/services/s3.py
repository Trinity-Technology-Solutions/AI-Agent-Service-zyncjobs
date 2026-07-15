"""S3 service — file storage for resumes, JD documents, and generated PDFs.
Architecture doc: AI Service → S3 (direct) or AI Service → Backend → S3 (proxied).
Supports both direct S3 access and backend-proxied access.

Usage:
    from recruitment_ai.services.s3 import s3_service

    # Upload a file
    url = await s3_service.upload(buffer, "resume.pdf", folder="resumes")

    # Download as bytes
    data = await s3_service.download(url)

    # Generate presigned URL (temporary access)
    signed = await s3_service.get_signed_url(key, expires=300)
"""
import io
import logging
from typing import Optional
from recruitment_ai.config.settings import settings

logger = logging.getLogger(__name__)


class S3Service:
    """Minimal S3 wrapper. Only initialized when AWS credentials are available."""

    def __init__(self):
        self._client = None
        self._bucket = settings.S3_BUCKET
        self._region = settings.S3_REGION
        self._prefix = settings.S3_PREFIX

    @property
    def client(self):
        if self._client is None:
            try:
                import aioboto3
                self._session = aioboto3.Session()
                self._client = "lazy"
                logger.info("S3 client ready (bucket=%s, region=%s)", self._bucket, self._region)
            except ImportError:
                logger.warning("aioboto3 not installed — S3 operations disabled")
                return None
        return self._client

    async def _get_s3(self):
        if self.client is None:
            return None
        try:
            session = aioboto3.Session()
            return await session.client("s3", region_name=self._region)
        except Exception as e:
            logger.warning("S3 client init failed: %s", e)
            return None

    async def upload(self, data: bytes, filename: str, folder: str = "uploads", content_type: str = "") -> Optional[str]:
        """Upload bytes to S3. Returns public URL or None."""
        s3 = await self._get_s3()
        if not s3:
            return None
        key = f"{self._prefix}{folder}/{filename}"
        kwargs = {"Bucket": self._bucket, "Key": key, "Body": data}
        if content_type:
            kwargs["ContentType"] = content_type
        try:
            await s3.put_object(**kwargs)
            url = f"https://s3.{self._region}.amazonaws.com/{self._bucket}/{key}"
            logger.info("S3 upload OK: %s", key)
            return url
        except Exception as e:
            logger.warning("S3 upload failed: %s", e)
            return None

    async def upload_from_buffer(self, buffer: io.BytesIO, filename: str, folder: str = "uploads") -> Optional[str]:
        return await self.upload(buffer.getvalue(), filename, folder)

    async def download(self, key_or_url: str) -> Optional[bytes]:
        """Download file from S3 by key or full URL. Returns bytes or None."""
        key = self._extract_key(key_or_url)
        if not key:
            return None
        s3 = await self._get_s3()
        if not s3:
            return None
        try:
            resp = await s3.get_object(Bucket=self._bucket, Key=key)
            return await resp["Body"].read()
        except Exception as e:
            logger.warning("S3 download failed (%s): %s", key, e)
            return None

    async def get_signed_url(self, key_or_url: str, expires: int = 300) -> Optional[str]:
        """Generate presigned GET URL with expiry in seconds."""
        key = self._extract_key(key_or_url)
        if not key:
            return None
        s3 = await self._get_s3()
        if not s3:
            return None
        try:
            url = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires,
            )
            return url
        except Exception as e:
            logger.warning("S3 presigned URL failed: %s", e)
            return None

    async def delete(self, key_or_url: str) -> bool:
        key = self._extract_key(key_or_url)
        if not key:
            return False
        s3 = await self._get_s3()
        if not s3:
            return False
        try:
            await s3.delete_object(Bucket=self._bucket, Key=key)
            return True
        except Exception as e:
            logger.warning("S3 delete failed: %s", e)
            return False

    async def exists(self, key_or_url: str) -> bool:
        key = self._extract_key(key_or_url)
        if not key:
            return False
        s3 = await self._get_s3()
        if not s3:
            return False
        try:
            await s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    def _extract_key(self, key_or_url: str) -> Optional[str]:
        """Extract S3 key from a full URL or return as-is if already a key."""
        if not key_or_url:
            return None
        if key_or_url.startswith("https://") or key_or_url.startswith("http://"):
            try:
                from urllib.parse import urlparse
                parsed = urlparse(key_or_url)
                path = parsed.path.lstrip("/")
                parts = path.split("/", 1)
                if len(parts) > 1 and parts[0] == self._bucket:
                    return parts[1]
                if len(parts) > 1:
                    return parts[1]
                return path
            except Exception:
                return None
        return key_or_url


s3_service = S3Service()
