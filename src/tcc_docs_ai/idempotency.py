from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError

from tcc_docs_ai.errors import IdempotencyConflictError, UpstreamUnavailableError


@dataclass
class IdempotencyRecord:
    request_hash: str
    events: list[bytes] = field(default_factory=list)
    completed: bool = False


@dataclass(frozen=True)
class IdempotencyStart:
    replay: IdempotencyRecord | None
    lease_token: str | None


class IdempotencyStore(Protocol):
    async def start(self, scope: str, key: str, request_hash: str) -> IdempotencyStart: ...

    async def append(self, scope: str, key: str, lease_token: str, event: bytes) -> None: ...

    async def complete(self, scope: str, key: str, lease_token: str) -> None: ...

    async def abandon(self, scope: str, key: str, lease_token: str) -> None: ...


class InMemoryIdempotencyStore:
    """Deterministic test store. Production uses RedisIdempotencyStore."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], IdempotencyRecord] = {}
        self._leases: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()

    async def start(self, scope: str, key: str, request_hash: str) -> IdempotencyStart:
        async with self._lock:
            record = self._records.get((scope, key))
            if record:
                if record.request_hash != request_hash:
                    raise IdempotencyConflictError(
                        "Idempotency key was already used with a different request"
                    )
                if not record.completed:
                    raise IdempotencyConflictError("Identical request is already in progress")
                return IdempotencyStart(replay=record, lease_token=None)
            lease_token = uuid.uuid4().hex
            self._records[(scope, key)] = IdempotencyRecord(request_hash=request_hash)
            self._leases[(scope, key)] = lease_token
            return IdempotencyStart(replay=None, lease_token=lease_token)

    async def append(self, scope: str, key: str, lease_token: str, event: bytes) -> None:
        async with self._lock:
            self._assert_lease(scope, key, lease_token)
            self._records[(scope, key)].events.append(event)

    async def complete(self, scope: str, key: str, lease_token: str) -> None:
        async with self._lock:
            self._assert_lease(scope, key, lease_token)
            self._records[(scope, key)].completed = True
            self._leases.pop((scope, key), None)

    async def abandon(self, scope: str, key: str, lease_token: str) -> None:
        async with self._lock:
            if self._leases.get((scope, key)) != lease_token:
                return
            self._records.pop((scope, key), None)
            self._leases.pop((scope, key), None)

    def _assert_lease(self, scope: str, key: str, lease_token: str) -> None:
        if self._leases.get((scope, key)) != lease_token:
            raise UpstreamUnavailableError("Idempotency lease is no longer active")


_START_SCRIPT = """
local request_hash = redis.call('HGET', KEYS[1], 'request_hash')
if not request_hash then
  redis.call('HSET', KEYS[1],
    'request_hash', ARGV[1], 'status', 'in_progress', 'lease_token', ARGV[2])
  redis.call('EXPIRE', KEYS[1], ARGV[3])
  redis.call('DEL', KEYS[2])
  return {'started'}
end
if request_hash ~= ARGV[1] then
  return {'conflict_request'}
end
local status = redis.call('HGET', KEYS[1], 'status')
if status ~= 'completed' then
  return {'conflict_in_progress'}
end
local events = redis.call('LRANGE', KEYS[2], 0, -1)
if #events == 0 then
  return {'missing_events'}
end
table.insert(events, 1, 'replay')
return events
"""

_APPEND_SCRIPT = """
if redis.call('HGET', KEYS[1], 'status') ~= 'in_progress'
  or redis.call('HGET', KEYS[1], 'lease_token') ~= ARGV[1] then
  return 0
end
redis.call('RPUSH', KEYS[2], ARGV[2])
redis.call('EXPIRE', KEYS[1], ARGV[3])
redis.call('EXPIRE', KEYS[2], ARGV[3])
return 1
"""

_COMPLETE_SCRIPT = """
if redis.call('HGET', KEYS[1], 'status') ~= 'in_progress'
  or redis.call('HGET', KEYS[1], 'lease_token') ~= ARGV[1] then
  return 0
end
redis.call('HSET', KEYS[1], 'status', 'completed')
redis.call('HDEL', KEYS[1], 'lease_token')
redis.call('EXPIRE', KEYS[1], ARGV[2])
redis.call('EXPIRE', KEYS[2], ARGV[2])
return 1
"""

_ABANDON_SCRIPT = """
if redis.call('HGET', KEYS[1], 'lease_token') ~= ARGV[1] then
  return 0
end
redis.call('DEL', KEYS[1], KEYS[2])
return 1
"""


class RedisIdempotencyStore:
    """Atomic generation replay shared by every service replica."""

    def __init__(
        self,
        client: Redis,
        *,
        key_prefix: str,
        in_progress_ttl_seconds: int,
        completed_ttl_seconds: int,
    ) -> None:
        self._client = client
        self._key_prefix = key_prefix.rstrip(":")
        self._in_progress_ttl_seconds = in_progress_ttl_seconds
        self._completed_ttl_seconds = completed_ttl_seconds

    async def start(self, scope: str, key: str, request_hash: str) -> IdempotencyStart:
        lease_token = uuid.uuid4().hex
        meta_key, events_key = self._keys(scope, key)
        try:
            result = await self._client.eval(
                _START_SCRIPT,
                2,
                meta_key,
                events_key,
                request_hash,
                lease_token,
                self._in_progress_ttl_seconds,
            )
        except RedisError as exc:
            raise UpstreamUnavailableError("Redis idempotency store is unavailable") from exc

        response = list(result)
        status = self._text(response[0])
        if status == "started":
            return IdempotencyStart(replay=None, lease_token=lease_token)
        if status == "conflict_request":
            raise IdempotencyConflictError(
                "Idempotency key was already used with a different request"
            )
        if status == "conflict_in_progress":
            raise IdempotencyConflictError("Identical request is already in progress")
        if status == "missing_events":
            raise UpstreamUnavailableError("Completed idempotency replay has no stored events")
        if status == "replay":
            return IdempotencyStart(
                replay=IdempotencyRecord(
                    request_hash=request_hash,
                    events=[self._bytes(event) for event in response[1:]],
                    completed=True,
                ),
                lease_token=None,
            )
        raise UpstreamUnavailableError("Redis returned an invalid idempotency state")

    async def append(self, scope: str, key: str, lease_token: str, event: bytes) -> None:
        meta_key, events_key = self._keys(scope, key)
        try:
            appended = await self._client.eval(
                _APPEND_SCRIPT,
                2,
                meta_key,
                events_key,
                lease_token,
                event,
                self._in_progress_ttl_seconds,
            )
        except RedisError as exc:
            raise UpstreamUnavailableError("Redis idempotency store is unavailable") from exc
        if appended != 1:
            raise UpstreamUnavailableError("Idempotency lease is no longer active")

    async def complete(self, scope: str, key: str, lease_token: str) -> None:
        meta_key, events_key = self._keys(scope, key)
        try:
            completed = await self._client.eval(
                _COMPLETE_SCRIPT,
                2,
                meta_key,
                events_key,
                lease_token,
                self._completed_ttl_seconds,
            )
        except RedisError as exc:
            raise UpstreamUnavailableError("Redis idempotency store is unavailable") from exc
        if completed != 1:
            raise UpstreamUnavailableError("Idempotency lease is no longer active")

    async def abandon(self, scope: str, key: str, lease_token: str) -> None:
        meta_key, events_key = self._keys(scope, key)
        try:
            await self._client.eval(
                _ABANDON_SCRIPT,
                2,
                meta_key,
                events_key,
                lease_token,
            )
        except RedisError as exc:
            raise UpstreamUnavailableError("Redis idempotency store is unavailable") from exc

    def _keys(self, scope: str, key: str) -> tuple[str, str]:
        identity = hashlib.sha256(f"{scope}\0{key}".encode()).hexdigest()
        # A hash tag mantém meta e eventos no mesmo slot em Redis Cluster.
        root = f"{self._key_prefix}:{{{identity}}}"
        return f"{root}:meta", f"{root}:events"

    @staticmethod
    def _text(value: object) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    @staticmethod
    def _bytes(value: object) -> bytes:
        return value if isinstance(value, bytes) else str(value).encode()
