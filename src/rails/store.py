"""Thread-safe store for Rails state management."""

import asyncio
from typing import Any, Dict, Optional
from collections import defaultdict

from .types import CounterValue, StateValue, StorageKey


class Store:
    """Thread-safe store for Rails state and counters.
    
    Provides counters and state management with async lock protection
    for concurrent access scenarios.
    """
    
    def __init__(self) -> None:
        """Initialize empty store with thread safety."""
        self._counters: Dict[str, CounterValue] = defaultdict(int)
        self._state: Dict[StorageKey, StateValue] = {}
        self._lock = asyncio.Lock()
        
    async def increment(self, key: str, amount: int = 1) -> CounterValue:
        """Thread-safe counter increment.
        
        Args:
            key: Counter key to increment
            amount: Amount to increment by (default 1)
            
        Returns:
            New counter value after increment
        """
        async with self._lock:
            self._counters[key] += amount
            return self._counters[key]
            
    async def get_counter(self, key: str, default: CounterValue = 0) -> CounterValue:
        """Get current counter value.
        
        Args:
            key: Counter key to retrieve
            default: Default value if counter not set
            
        Returns:
            Current counter value (default if not set)
        """
        async with self._lock:
            return self._counters.get(key, default)
            
    async def set_counter(self, key: str, value: CounterValue) -> None:
        """Set counter to specific value.
        
        Args:
            key: Counter key to set
            value: New counter value
        """
        async with self._lock:
            self._counters[key] = value
            
    async def reset_counter(self, key: str) -> None:
        """Reset counter to zero.
        
        Args:
            key: Counter key to reset
        """
        async with self._lock:
            self._counters[key] = 0
            
    async def set(self, key: StorageKey, value: StateValue) -> None:
        """Set state value.
        
        Args:
            key: State key
            value: Value to store
        """
        async with self._lock:
            self._state[key] = value
            
    async def get(self, key: StorageKey, default: StateValue = None) -> StateValue:
        """Get state value.
        
        Args:
            key: State key to retrieve
            default: Default value if key not found
            
        Returns:
            Stored value or default
        """
        async with self._lock:
            return self._state.get(key, default)
            
    async def exists(self, key: StorageKey) -> bool:
        """Check if state key exists.
        
        Args:
            key: State key to check
            
        Returns:
            True if key exists
        """
        async with self._lock:
            return key in self._state
            
    async def delete(self, key: StorageKey) -> bool:
        """Delete state key.
        
        Args:
            key: State key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        async with self._lock:
            if key in self._state:
                del self._state[key]
                return True
            return False
            
    async def clear(self) -> None:
        """Clear all state and counters."""
        async with self._lock:
            self._state.clear()
            self._counters.clear()
            
    # Synchronous methods for simpler usage patterns
    def increment_sync(self, key: str, amount: int = 1) -> CounterValue:
        """Synchronous counter increment."""
        self._counters[key] += amount
        return self._counters[key]
        
    def get_counter_sync(self, key: str, default: CounterValue = 0) -> CounterValue:
        """Synchronous counter retrieval."""
        return self._counters.get(key, default)
        
    def set_sync(self, key: StorageKey, value: StateValue) -> None:
        """Synchronous state setter."""
        self._state[key] = value
        
    def get_sync(self, key: StorageKey, default: StateValue = None) -> StateValue:
        """Synchronous state getter."""
        return self._state.get(key, default)
        
    def exists_sync(self, key: StorageKey) -> bool:
        """Synchronous state key existence check."""
        return key in self._state
        
    def delete_sync(self, key: StorageKey) -> bool:
        """Synchronous state key deletion."""
        if key in self._state:
            del self._state[key]
            return True
        return False