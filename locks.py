"""
Redis-based row locking for convergence safety.

Prevents concurrent convergence or upserts from overwriting rows
that are currently being processed. Uses Redis SET with NX+EX for
atomic lock acquisition with TTL expiry.
"""
from __future__ import annotations

import time
import redis


class RowLockManager:
    """
    Manages row-level locks in Redis.

    Key format: lock:{workflow}:{primary_key_value}
    Value: lock owner ID (e.g., workflow run ID or Temporal workflow ID)
    TTL: configurable, prevents deadlocks if a worker crashes.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 300,  # 5 minutes
    ):
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = default_ttl

    def _key(self, workflow: str, row_id: str) -> str:
        return f"lock:{workflow}:{row_id}"

    def acquire(
        self,
        workflow: str,
        row_ids: list[str],
        owner: str,
        ttl: int | None = None,
    ) -> tuple[list[str], list[str]]:
        """
        Try to acquire locks on a batch of row IDs.

        Returns (locked_ids, already_locked_ids).
        locked_ids: IDs we successfully locked.
        already_locked_ids: IDs someone else holds — skip these rows.
        """
        ttl = ttl or self.default_ttl
        locked = []
        already_locked = []

        pipe = self.client.pipeline(transaction=False)
        keys = [(rid, self._key(workflow, rid)) for rid in row_ids]

        # Attempt SET NX EX for each ID
        for rid, key in keys:
            pipe.set(key, owner, nx=True, ex=ttl)
        results = pipe.execute()

        for (rid, _), acquired in zip(keys, results):
            if acquired:
                locked.append(rid)
            else:
                already_locked.append(rid)

        return locked, already_locked

    def release(self, workflow: str, row_ids: list[str], owner: str) -> int:
        """
        Release locks owned by this owner. Uses Lua script for atomic
        check-and-delete (only delete if value matches owner).

        Returns count of locks released.
        """
        lua = """
        local released = 0
        for i, key in ipairs(KEYS) do
            if redis.call('get', key) == ARGV[1] then
                redis.call('del', key)
                released = released + 1
            end
        end
        return released
        """
        keys = [self._key(workflow, rid) for rid in row_ids]
        if not keys:
            return 0
        return self.client.eval(lua, len(keys), *keys, owner)

    def check_locked(self, workflow: str, row_ids: list[str]) -> list[str]:
        """Return which of the given row_ids are currently locked."""
        if not row_ids:
            return []
        pipe = self.client.pipeline(transaction=False)
        keys = [self._key(workflow, rid) for rid in row_ids]
        for key in keys:
            pipe.exists(key)
        results = pipe.execute()
        return [rid for rid, exists in zip(row_ids, results) if exists]

    def extend(self, workflow: str, row_ids: list[str], owner: str, ttl: int | None = None):
        """Extend TTL on locks we own (heartbeat)."""
        ttl = ttl or self.default_ttl
        lua = """
        local extended = 0
        for i, key in ipairs(KEYS) do
            if redis.call('get', key) == ARGV[1] then
                redis.call('expire', key, ARGV[2])
                extended = extended + 1
            end
        end
        return extended
        """
        keys = [self._key(workflow, rid) for rid in row_ids]
        if not keys:
            return 0
        return self.client.eval(lua, len(keys), *keys, owner, str(ttl))

    def ping(self) -> bool:
        """Check if Redis is reachable."""
        try:
            return self.client.ping()
        except Exception:
            return False
