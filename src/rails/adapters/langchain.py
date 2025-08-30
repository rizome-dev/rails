"""LangChain adapter for Rails integration.

This module provides seamless integration between Rails and LangChain,
allowing Rails conditional message injection to work with any LangChain chain,
agent, or chat model.
"""

import asyncio
from typing import Any

try:
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
    from langchain_core.runnables import Runnable
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Graceful degradation when LangChain is not installed
    BaseMessage = Any
    Runnable = Any
    LANGCHAIN_AVAILABLE = False

from ..core import Rails
from ..types import Message
from .base import BaseRailsAdapter


class LangChainAdapter(BaseRailsAdapter):
    """Rails adapter for LangChain integration.
    
    This adapter allows you to wrap any LangChain runnable (chains, agents, chat models)
    with Rails conditional message injection capabilities.
    
    Usage:
        from langchain_openai import ChatOpenAI
        from rails.adapters import LangChainAdapter
        
        # Set up Rails rules
        rails = Rails()
        rails.when(lambda s: s.get_counter_sync('turns') >= 3).inject({
            "role": "system", 
            "content": "You've been chatting for a while. Try to wrap up."
        })
        
        # Create LangChain model
        llm = ChatOpenAI()
        
        # Wrap with Rails
        adapter = LangChainAdapter(rails, llm)
        
        # Use as normal, but with Rails injection
        messages = [{"role": "user", "content": "Hello!"}]
        result = await adapter.run(messages)
    """

    def __init__(self, rails: Rails | None = None, runnable: Runnable | None = None):
        """Initialize the LangChain adapter.
        
        Args:
            rails: Rails instance for message injection
            runnable: LangChain runnable (chain, agent, chat model, etc.)
        """
        super().__init__(rails)
        self.runnable = runnable

        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install it with: pip install langchain-core"
            )

    async def process_messages(self, messages: list[Message],
                             runnable: Runnable | None = None, **kwargs) -> Any:
        """Process messages through LangChain runnable.
        
        Args:
            messages: Rails-processed messages
            runnable: Optional runnable to use (overrides instance runnable)
            **kwargs: Additional arguments for the runnable
            
        Returns:
            LangChain runnable result
        """
        target_runnable = runnable or self.runnable

        if target_runnable is None:
            raise ValueError("No runnable provided. Pass one to __init__ or process_messages")

        # Convert Rails messages to LangChain messages
        lc_messages = [self.convert_from_rails_message(msg) for msg in messages]

        # Run through LangChain
        if asyncio.iscoroutinefunction(target_runnable.invoke):
            result = await target_runnable.ainvoke(lc_messages, **kwargs)
        else:
            result = target_runnable.invoke(lc_messages, **kwargs)

        return result

    def convert_to_rails_message(self, message: BaseMessage | dict[str, Any]) -> Message:
        """Convert LangChain message to Rails message format.
        
        Args:
            message: LangChain BaseMessage or dict
            
        Returns:
            Rails Message (dict format)
        """
        if isinstance(message, dict):
            return message

        # Convert LangChain BaseMessage to dict
        if hasattr(message, 'type') and hasattr(message, 'content'):
            return {
                "role": self._get_role_from_type(message.type),
                "content": message.content
            }

        # Fallback for unknown message types
        return {"role": "user", "content": str(message)}

    def convert_from_rails_message(self, message: Message) -> BaseMessage:
        """Convert Rails message to LangChain BaseMessage.
        
        Args:
            message: Rails Message (dict format)
            
        Returns:
            LangChain BaseMessage
        """
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "system":
            return SystemMessage(content=content)
        elif role == "assistant" or role == "ai":
            return AIMessage(content=content)
        else:  # user, human, or any other role
            return HumanMessage(content=content)

    def _get_role_from_type(self, message_type: str) -> str:
        """Map LangChain message type to Rails role.
        
        Args:
            message_type: LangChain message type
            
        Returns:
            Rails message role
        """
        type_to_role = {
            "system": "system",
            "ai": "assistant",
            "human": "user",
            "assistant": "assistant",
            "user": "user"
        }
        return type_to_role.get(message_type, "user")

    async def update_rails_state(self, original_messages: list[Message],
                               modified_messages: list[Message], result: Any) -> None:
        """Update Rails state after LangChain processing.
        
        Args:
            original_messages: Original input messages
            modified_messages: Messages after Rails injection
            result: LangChain processing result
        """
        await super().update_rails_state(original_messages, modified_messages, result)

        # Track LangChain-specific metrics
        if hasattr(result, 'usage_metadata'):
            await self.rails.store.increment("langchain_tokens",
                                           result.usage_metadata.get("total_tokens", 0))


def create_langchain_adapter(runnable: Runnable,
                           rails: Rails | None = None) -> LangChainAdapter:
    """Factory function to create a LangChain Rails adapter.
    
    Args:
        runnable: LangChain runnable to wrap
        rails: Optional Rails instance
        
    Returns:
        Configured LangChainAdapter
        
    Example:
        from langchain_openai import ChatOpenAI
        from rails import Rails
        from rails.adapters import create_langchain_adapter
        
        # Set up Rails
        rails = Rails()
        rails.when(lambda s: s.get_counter_sync('turns') >= 5).inject({
            "role": "system", 
            "content": "Let's start wrapping up this conversation."
        })
        
        # Create adapter
        llm = ChatOpenAI()
        adapter = create_langchain_adapter(llm, rails)
        
        # Use it
        result = await adapter.run([
            {"role": "user", "content": "What's the weather like?"}
        ])
    """
    return LangChainAdapter(rails, runnable)


# Decorator for wrapping LangChain runnables with Rails
def with_rails(rails: Rails | None = None):
    """Decorator to wrap LangChain runnables with Rails.
    
    Args:
        rails: Rails instance to use
        
    Returns:
        Decorator function
        
    Example:
        rails = Rails()
        rails.when(condition).inject(message)
        
        @with_rails(rails)
        def my_chain():
            return ChatOpenAI() | StrOutputParser()
        
        # Now my_chain includes Rails injection
        result = await my_chain.run(messages)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            runnable = func(*args, **kwargs)
            return LangChainAdapter(rails, runnable)
        return wrapper
    return decorator
