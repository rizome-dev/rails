"""Type definitions for Rails library."""

from typing import Any, Dict, List, Protocol, runtime_checkable
from abc import ABC, abstractmethod


# Framework-agnostic message type
Message = Dict[str, Any]


@runtime_checkable
class Condition(Protocol):
    """Protocol for condition callables that return boolean results."""
    
    def __call__(self, store: 'Store') -> bool:
        """Evaluate condition against current store state.
        
        Args:
            store: Current Rails store instance
            
        Returns:
            True if condition is met, False otherwise
        """
        ...


@runtime_checkable
class Injector(Protocol):
    """Protocol for injection strategies."""
    
    def inject(self, messages: List[Message], new_message: Message) -> List[Message]:
        """Inject a new message into the message chain.
        
        Args:
            messages: Current message chain
            new_message: Message to inject
            
        Returns:
            Modified message chain
        """
        ...


# Store type aliases
CounterValue = int
StateValue = Any
StorageKey = str