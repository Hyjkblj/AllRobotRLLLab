"""Explicit infrastructure probes used during deployment startup."""

from __future__ import annotations


def check_redis(url: str) -> bool:
    from redis import Redis

    client = Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
    return bool(client.ping())


def check_minio(*, endpoint: str, access_key: str, secret_key: str, bucket: str, secure: bool) -> bool:
    from minio import Minio

    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    return bool(client.bucket_exists(bucket))


__all__ = ["check_minio", "check_redis"]

