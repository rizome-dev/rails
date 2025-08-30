"""Base adapter and middleware system for Rails framework integration."""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, ConfigDict

from ..core import Rails, rails_context
from ..store import Store
from ..types import Message, Role


@runtime_checkable
class MessageProcessor(Protocol):
    """Protocol for message processing functions."""

    async def process(self, messages: list[Any]) -> list[Any]:
        """Process messages through the framework."""
        ...


class BaseAdapter(BaseModel, ABC):
    """Base adapter for framework integration."""

    rails: Rails
    framework_name: str = "unknown"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def to_rails_messages(self, framework_messages: list[Any]) -> list[Message]:
        """Convert framework messages to Rails format.
        
        Args:
            framework_messages: Messages in framework format
            
        Returns:
            Messages in Rails format
        """
        ...

    @abstractmethod
    async def from_rails_messages(self, rails_messages: list[Message]) -> list[Any]:
        """Convert Rails messages to framework format.
        
        Args:
            rails_messages: Messages in Rails format
            
        Returns:
            Messages in framework format
        """
        ...

    async def register_store_access(self, store: Store) -> None:
        """Make Rails store accessible to framework tools.
        
        Args:
            store: Rails store instance
        """
        # Set Rails context for tool access
        token = rails_context.set(self.rails)
        logger.debug(f"Registered Rails store for {self.framework_name} tools")
        return token

    async def process_messages(self, messages: list[Any]) -> list[Any]:
        """Process messages through Rails.
        
        Args:
            messages: Framework messages
            
        Returns:
            Processed messages
        """
        # Convert to Rails format
        rails_messages = await self.to_rails_messages(messages)

        # Process through Rails
        processed = await self.rails.process(rails_messages)

        # Update counters
        await self.rails.store.increment("turns")
        if len(processed) > len(rails_messages):
            await self.rails.store.increment("injections", len(processed) - len(rails_messages))

        # Convert back to framework format
        return await self.from_rails_messages(processed)

    async def wrap(self, agent: Any) -> Any:
        """Wrap an agent with Rails integration.
        
        Args:
            agent: Agent to wrap
            
        Returns:
            Wrapped agent
        """
        # Default implementation - subclasses should override
        logger.warning(f"Using default wrap() for {self.framework_name}")
        return agent

    async def __aenter__(self):
        """Context manager entry."""
        await self.rails.__aenter__()
        await self.register_store_access(self.rails.store)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.rails.__aexit__(exc_type, exc_val, exc_tb)


class MiddlewareAdapter(BaseAdapter):
    """Adapter that acts as middleware in a processing chain."""

    next_processor: MessageProcessor | None = None

    async def __call__(self, messages: list[Any]) -> list[Any]:
        """Process messages as middleware.
        
        Args:
            messages: Input messages
            
        Returns:
            Processed messages
        """
        # Process through Rails
        processed = await self.process_messages(messages)

        # Pass to next processor if available
        if self.next_processor:
            if hasattr(self.next_processor, 'process'):
                processed = await self.next_processor.process(processed)
            elif callable(self.next_processor):
                if inspect.iscoroutinefunction(self.next_processor):
                    processed = await self.next_processor(processed)
                else:
                    processed = self.next_processor(processed)

        return processed

    def chain(self, processor: MessageProcessor) -> 'MiddlewareAdapter':
        """Chain another processor after this middleware.
        
        Args:
            processor: Next processor in chain
            
        Returns:
            Self for chaining
        """
        self.next_processor = processor
        return self


class GenericAdapter(BaseAdapter):
    """Generic adapter for any message processing function."""

    framework_name: str = "generic"
    message_converter: Callable | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def to_rails_messages(self, framework_messages: list[Any]) -> list[Message]:
        """Convert using provided converter or passthrough."""
        if self.message_converter:
            return [self.message_converter(msg) for msg in framework_messages]

        # Assume dict-like messages with role and content
        rails_messages = []
        for msg in framework_messages:
            if isinstance(msg, dict):
                rails_messages.append(Message(
                    role=Role(msg.get("role", "user")),
                    content=msg.get("content", "")
                ))
            elif isinstance(msg, Message):
                rails_messages.append(msg)
            else:
                # Try to extract content
                rails_messages.append(Message(
                    role=Role.USER,
                    content=str(msg)
                ))

        return rails_messages

    async def from_rails_messages(self, rails_messages: list[Message]) -> list[Any]:
        """Convert back to framework format."""
        # Return as dicts for generic compatibility
        return [msg.model_dump() for msg in rails_messages]


def create_middleware_stack(*middlewares) -> Callable:
    """Create a middleware stack from multiple processors.
    
    Args:
        *middlewares: Middleware processors to chain
        
    Returns:
        Combined middleware function
    """
    if not middlewares:
        async def identity(messages):
            return messages
        return identity

    # Chain middlewares
    stack = middlewares[0]
    for mw in middlewares[1:]:
        if hasattr(stack, 'chain'):
            stack.chain(mw)
        else:
            # Wrap in a lambda to chain
            prev_stack = stack
            async def chained(messages, s=prev_stack, m=mw):
                result = await s(messages)
                if inspect.iscoroutinefunction(m):
                    return await m(result)
                else:
                    return m(result)
            stack = chained

    return stack


def rails_middleware(rails: Rails) -> Callable:
    """Create a middleware function from Rails instance.
    
    Args:
        rails: Rails instance
        
    Returns:
        Middleware function
    """
    async def middleware(messages: list[Any], next_handler: Callable | None = None) -> list[Any]:
        # Set Rails context
        token = rails_context.set(rails)

        try:
            # Process through Rails
            if messages and isinstance(messages[0], Message):
                processed = await rails.process(messages)
            else:
                # Convert to Rails format
                adapter = GenericAdapter(rails=rails)
                rails_messages = await adapter.to_rails_messages(messages)
                processed_rails = await rails.process(rails_messages)
                processed = await adapter.from_rails_messages(processed_rails)

            # Call next handler if provided
            if next_handler:
                if inspect.iscoroutinefunction(next_handler):
                    processed = await next_handler(processed)
                else:
                    processed = next_handler(processed)

            return processed

        finally:
            # Reset context
            rails_context.reset(token)

    return middleware


def create_adapter(processor: Callable | None = None,
                  rails: Rails | None = None) -> GenericAdapter:
    """Factory function to create a generic Rails adapter.
    
    Args:
        processor: Function to process messages
        rails: Rails instance to use
        
    Returns:
        Configured GenericAdapter
        
    Example:
        def my_processor(messages):
            # Process messages here
            return processed_messages
            
        adapter = create_adapter(processor=my_processor)
        result = await adapter.process_messages(messages)
    """
    rails = rails or Rails()
    adapter = GenericAdapter(rails=rails)

    if processor:
        # Wrap processor in adapter
        original_process = adapter.process_messages

        async def wrapped_process(messages):
            # Process through Rails first
            rails_processed = await original_process(messages)
            # Then through custom processor
            if inspect.iscoroutinefunction(processor):
                return await processor(rails_processed)
            else:
                return processor(rails_processed)

        adapter.process_messages = wrapped_process

    return adapter
