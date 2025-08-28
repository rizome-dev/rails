"""Base adapter pattern for Rails framework integrations.

This module provides the foundation for creating Rails adapters for any agent framework.
The adapter pattern allows Rails to integrate seamlessly with different frameworks while
maintaining a consistent interface.
"""

from typing import List, Dict, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
import asyncio

from ..core import Rails
from ..types import Message


class BaseRailsAdapter(ABC):
    """Base class for Rails framework adapters.
    
    This class provides the common interface and functionality that all
    framework adapters should implement. It handles Rails lifecycle management
    and provides hooks for framework-specific message handling.
    
    Usage:
        class MyFrameworkAdapter(BaseRailsAdapter):
            async def process_messages(self, messages):
                # Framework-specific processing
                return processed_messages
                
        adapter = MyFrameworkAdapter(rails)
        result = await adapter.run(messages)
    """
    
    def __init__(self, rails: Optional[Rails] = None):
        """Initialize the adapter with a Rails instance.
        
        Args:
            rails: Rails instance to use. If None, creates a new one.
        """
        self.rails = rails or Rails()
        self._context_active = False
        
    async def run(self, messages: List[Message], **kwargs) -> Any:
        """Run the adapter with Rails integration.
        
        This is the main entry point that orchestrates the Rails lifecycle
        with framework-specific processing.
        
        Args:
            messages: Input messages to process
            **kwargs: Framework-specific arguments
            
        Returns:
            Framework-specific result
        """
        # Apply Rails message injection
        modified_messages = await self.rails.check(messages)
        
        # Let the specific adapter process the messages
        result = await self.process_messages(modified_messages, **kwargs)
        
        # Update Rails state based on the interaction
        await self.update_rails_state(messages, modified_messages, result)
        
        return result
    
    @abstractmethod
    async def process_messages(self, messages: List[Message], **kwargs) -> Any:
        """Process messages with the specific framework.
        
        This method must be implemented by each framework adapter to handle
        the actual framework-specific processing.
        
        Args:
            messages: Modified messages from Rails
            **kwargs: Framework-specific arguments
            
        Returns:
            Framework-specific result
        """
        pass
    
    async def update_rails_state(self, original_messages: List[Message], 
                               modified_messages: List[Message], result: Any) -> None:
        """Update Rails state after processing.
        
        Override this method to update Rails counters/state based on the
        interaction outcome. Default implementation increments turn counter.
        
        Args:
            original_messages: Original input messages
            modified_messages: Messages after Rails injection
            result: Framework processing result
        """
        await self.rails.store.increment("turns")
        
        # Track message modifications
        if len(modified_messages) > len(original_messages):
            await self.rails.store.increment("injections")
    
    def convert_to_rails_message(self, message: Any) -> Message:
        """Convert framework-specific message to Rails Message format.
        
        Override this method if your framework uses a different message format.
        Default assumes messages are already dict-based.
        
        Args:
            message: Framework-specific message
            
        Returns:
            Rails-compatible Message (Dict[str, Any])
        """
        if isinstance(message, dict):
            return message
        
        # Framework-specific conversion logic should be implemented here
        raise NotImplementedError("Message conversion not implemented for this adapter")
    
    def convert_from_rails_message(self, message: Message) -> Any:
        """Convert Rails Message to framework-specific format.
        
        Override this method if your framework uses a different message format.
        Default returns the dict as-is.
        
        Args:
            message: Rails Message
            
        Returns:
            Framework-specific message format
        """
        return message
    
    async def __aenter__(self):
        """Context manager entry."""
        self._context_active = True
        await self.rails.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._context_active = False
        await self.rails.__aexit__(exc_type, exc_val, exc_tb)


class SimpleRailsAdapter(BaseRailsAdapter):
    """Simple Rails adapter for framework-agnostic usage.
    
    This adapter provides a minimal implementation that can be used
    with any framework or as a standalone Rails wrapper.
    
    Usage:
        rails = Rails()
        rails.when(condition).inject(message)
        
        adapter = SimpleRailsAdapter(rails)
        result = await adapter.run(messages, processor=my_function)
    """
    
    async def process_messages(self, messages: List[Message], 
                             processor: Optional[Callable] = None, **kwargs) -> Any:
        """Process messages with optional custom processor.
        
        Args:
            messages: Messages to process
            processor: Optional callable to process messages
            **kwargs: Additional arguments for processor
            
        Returns:
            Result from processor or messages unchanged
        """
        if processor:
            if asyncio.iscoroutinefunction(processor):
                return await processor(messages, **kwargs)
            else:
                return processor(messages, **kwargs)
        
        # Default: return messages unchanged
        return messages


class ProcessorAdapter(SimpleRailsAdapter):
    """Adapter that wraps a processor function."""
    
    def __init__(self, rails: Optional[Rails] = None, processor: Optional[Callable] = None):
        super().__init__(rails)
        self.processor = processor
    
    async def process_messages(self, messages: List[Message], **kwargs) -> Any:
        """Process messages using the stored processor."""
        return await super().process_messages(messages, processor=self.processor, **kwargs)


def create_adapter(rails: Optional[Rails] = None, 
                   processor: Optional[Callable] = None) -> ProcessorAdapter:
    """Factory function to create a simple Rails adapter.
    
    This is a convenience function for quickly creating adapters without
    subclassing.
    
    Args:
        rails: Rails instance to use
        processor: Function to process messages
        
    Returns:
        Configured ProcessorAdapter
        
    Example:
        def my_processor(messages):
            # Process messages here
            return processed_messages
            
        adapter = create_adapter(processor=my_processor)
        result = await adapter.run(messages)
    """
    return ProcessorAdapter(rails, processor)