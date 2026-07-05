import io
import uuid
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import settings


def _s3_client():
    return boto3.client("s3", region_name=settings.aws_region)


def upload_bytes_to_s3(data: bytes, filename: str, content_type: str) -> str | None:
    if not settings.store_uploads_in_s3 or not settings.s3_bucket:
        return None

    key = (
        f"{settings.s3_upload_prefix}/"
        f"{datetime.now(timezone.utc).strftime('%Y/%m/%d')}/"
        f"{uuid.uuid4().hex}_{filename}"
    )

    try:
        _s3_client().put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return f"s3://{settings.s3_bucket}/{key}"
    except (BotoCoreError, ClientError):
        return None
