"""Shared state store for Rails lifecycle orchestration."""

import json
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
from anyio import Lock
from pydantic import BaseModel, ConfigDict, Field

from .types import QueueConfig, StateEvent, StoreConfig


class Store(BaseModel):
    """Thread-safe shared state store for Rails and agent tools.
    
    This store is the heart of Rails - it provides a shared state space
    that both Rails conditions and agent tools can access, enabling
    sophisticated feedback loops and lifecycle orchestration.
    """

    config: StoreConfig = Field(default_factory=StoreConfig)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Initialize store with configuration."""
        super().__init__(**data)

        # Initialize internal state as private attributes
        self._counters: dict[str, int] = defaultdict(int)
        self._state: dict[str, Any] = {}
        self._queues: dict[str, deque] = {}
        self._events: deque[StateEvent] = deque(maxlen=self.config.max_event_history)
        self._lock = Lock()
        self._event_subscribers: list = []

        # Initialize default queues from config
        for queue_name, queue_config in self.config.default_queues.items():
            maxlen = queue_config.max_size
            self._queues[queue_name] = deque(maxlen=maxlen)

    # Counter operations
    async def increment(self, key: str, amount: int = 1, triggered_by: str | None = None) -> int:
        """Increment a counter and emit event.
        
        Args:
            key: Counter key to increment
            amount: Amount to increment by
            triggered_by: Tool or condition that triggered this
            
        Returns:
            New counter value
        """
        async with self._lock:
            old_value = self._counters[key]
            self._counters[key] += amount
            new_value = self._counters[key]

            if self.config.emit_events:
                await self._emit_event(StateEvent(
                    event_type="counter_increment",
                    key=key,
                    value=new_value,
                    previous_value=old_value,
                    triggered_by=triggered_by
                ))

            return new_value

    async def get_counter(self, key: str, default: int = 0) -> int:
        """Get current counter value."""
        async with self._lock:
            return self._counters.get(key, default)

    async def reset_counter(self, key: str, triggered_by: str | None = None) -> None:
        """Reset counter to zero."""
        async with self._lock:
            old_value = self._counters.get(key, 0)
            self._counters[key] = 0

            if self.config.emit_events:
                await self._emit_event(StateEvent(
                    event_type="counter_increment",
                    key=key,
                    value=0,
                    previous_value=old_value,
                    triggered_by=triggered_by
                ))

    # State operations
    async def set(self, key: str, value: Any, triggered_by: str | None = None) -> None:
        """Set state value and emit event."""
        async with self._lock:
            old_value = self._state.get(key)
            self._state[key] = value

            if self.config.emit_events:
                await self._emit_event(StateEvent(
                    event_type="state_set",
                    key=key,
                    value=value,
                    previous_value=old_value,
                    triggered_by=triggered_by
                ))

    async def get(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        async with self._lock:
            return self._state.get(key, default)

    async def delete(self, key: str, triggered_by: str | None = None) -> bool:
        """Delete state key."""
        async with self._lock:
            if key in self._state:
                old_value = self._state[key]
                del self._state[key]

                if self.config.emit_events:
                    await self._emit_event(StateEvent(
                        event_type="state_set",
                        key=key,
                        value=None,
                        previous_value=old_value,
                        triggered_by=triggered_by
                    ))
                return True
            return False

    # Queue operations - Critical for lifecycle management
    async def push_queue(self, queue: str, item: Any, triggered_by: str | None = None) -> None:
        """Push item to queue (FIFO by default)."""
        async with self._lock:
            if queue not in self._queues:
                config = self.config.default_queues.get(queue, QueueConfig())
                self._queues[queue] = deque(maxlen=config.max_size)

            q = self._queues[queue]

            # Handle deduplication if configured
            queue_config = self.config.default_queues.get(queue, QueueConfig())
            if queue_config.auto_dedup and item in q:
                return

            q.append(item)

            if self.config.emit_events:
                await self._emit_event(StateEvent(
                    event_type="queue_push",
                    key=queue,
                    value=item,
                    triggered_by=triggered_by
                ))

    async def pop_queue(self, queue: str, triggered_by: str | None = None) -> Any | None:
        """Pop item from queue."""
        async with self._lock:
            if queue not in self._queues or not self._queues[queue]:
                return None

            q = self._queues[queue]
            queue_config = self.config.default_queues.get(queue, QueueConfig())

            item = q.popleft() if queue_config.fifo else q.pop()

            if self.config.emit_events:
                await self._emit_event(StateEvent(
                    event_type="queue_pop",
                    key=queue,
                    value=item,
                    triggered_by=triggered_by
                ))

            return item

    async def get_queue(self, queue: str) -> list[Any]:
        """Get queue contents as list (without modifying)."""
        async with self._lock:
            if queue not in self._queues:
                return []
            return list(self._queues[queue])

    async def queue_length(self, queue: str) -> int:
        """Get queue length."""
        async with self._lock:
            if queue not in self._queues:
                return 0
            return len(self._queues[queue])

    async def clear_queue(self, queue: str, triggered_by: str | None = None) -> None:
        """Clear all items from queue."""
        async with self._lock:
            if queue in self._queues:
                self._queues[queue].clear()

                if self.config.emit_events:
                    await self._emit_event(StateEvent(
                        event_type="queue_pop",
                        key=queue,
                        value=None,
                        triggered_by=triggered_by
                    ))

    # Event operations
    async def _emit_event(self, event: StateEvent) -> None:
        """Emit state change event."""
        self._events.append(event)

        # Notify subscribers (for real-time monitoring)
        for subscriber in self._event_subscribers:
            try:
                await subscriber(event)
            except Exception:
                pass  # Don't let subscriber errors affect store

    async def get_events(self, since: datetime | None = None) -> list[StateEvent]:
        """Get events, optionally filtered by timestamp."""
        async with self._lock:
            if since is None:
                return list(self._events)
            return [e for e in self._events if e.timestamp > since]

    def subscribe_events(self, callback) -> None:
        """Subscribe to real-time events."""
        self._event_subscribers.append(callback)

    def unsubscribe_events(self, callback) -> None:
        """Unsubscribe from events."""
        if callback in self._event_subscribers:
            self._event_subscribers.remove(callback)

    # Event streaming for observability
    async def event_stream(self) -> AsyncIterator[StateEvent]:
        """Stream events as they occur."""
        queue = []

        async def collector(event):
            queue.append(event)

        self.subscribe_events(collector)
        try:
            while True:
                if queue:
                    yield queue.pop(0)
                else:
                    await anyio.sleep(0.01)  # Small sleep to prevent busy loop
        finally:
            self.unsubscribe_events(collector)

    # Persistence operations
    async def persist(self, path: Path | None = None) -> None:
        """Persist store state to disk."""
        if not self.config.persist_on_exit and path is None:
            return

        if path is None and self.config.persistence_path is None:
            return
        path = path or Path(self.config.persistence_path)

        async with self._lock:
            state = {
                "counters": dict(self._counters),
                "state": self._state,
                "queues": {k: list(v) for k, v in self._queues.items()},
                "timestamp": datetime.now().isoformat()
            }

        path.write_text(json.dumps(state, default=str, indent=2))

    async def restore(self, path: Path | None = None) -> None:
        """Restore store state from disk."""
        if path is None and self.config.persistence_path is None:
            return
        path = path or Path(self.config.persistence_path)
        if not path.exists():
            return

        state = json.loads(path.read_text())

        async with self._lock:
            self._counters = defaultdict(int, state.get("counters", {}))
            self._state = state.get("state", {})
            self._queues = {
                k: deque(v, maxlen=self.config.default_queues.get(k, QueueConfig()).max_size)
                for k, v in state.get("queues", {}).items()
            }

    # Bulk operations
    async def get_snapshot(self) -> dict[str, Any]:
        """Get complete snapshot of store state."""
        async with self._lock:
            return {
                "counters": dict(self._counters),
                "state": dict(self._state),
                "queues": {k: list(v) for k, v in self._queues.items()},
                "events": [e.model_dump() for e in self._events][-100:]  # Last 100 events
            }

    async def clear(self) -> None:
        """Clear all store state."""
        async with self._lock:
            self._counters.clear()
            self._state.clear()
            self._queues.clear()
            self._events.clear()

    # Synchronous convenience methods for tools
    def increment_sync(self, key: str, amount: int = 1) -> int:
        """Synchronous counter increment for use in tools."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.increment(key, amount))
        else:
            # We're in an event loop, schedule the coroutine
            future = asyncio.ensure_future(self.increment(key, amount))
            return asyncio.run_coroutine_threadsafe(future, loop).result()

    def get_sync(self, key: str, default: Any = None) -> Any:
        """Synchronous state getter for use in tools."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.get(key, default))
        else:
            future = asyncio.ensure_future(self.get(key, default))
            return asyncio.run_coroutine_threadsafe(future, loop).result()

    def set_sync(self, key: str, value: Any) -> None:
        """Synchronous state setter for use in tools."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.set(key, value))
        else:
            future = asyncio.ensure_future(self.set(key, value))
            asyncio.run_coroutine_threadsafe(future, loop).result()
