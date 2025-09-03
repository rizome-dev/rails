"""LangChain adapter for Rails integration.

This module provides seamless integration between Rails and LangChain,
allowing Rails conditional message injection to work with any LangChain chain,
agent, or chat model.
"""

import asyncio
import inspect
from typing import Any, Optional
from functools import wraps

try:
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
        FunctionMessage,
    )
    from langchain_core.runnables import Runnable
    from langchain_core.runnables.config import RunnableConfig
    from langchain_core.agents import AgentExecutor

    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Graceful degradation when LangChain is not installed
    BaseMessage = Any
    Runnable = Any
    RunnableConfig = Any
    AgentExecutor = Any

    # Create dummy message classes for when LangChain isn't available
    class AIMessage:
        def __init__(self, content: str, **kwargs):
            self.content = content

    class HumanMessage:
        def __init__(self, content: str, **kwargs):
            self.content = content

    class SystemMessage:
        def __init__(self, content: str, **kwargs):
            self.content = content

    class ToolMessage:
        def __init__(self, content: str, tool_call_id: str = "", **kwargs):
            self.content = content
            self.tool_call_id = tool_call_id

    class FunctionMessage:
        def __init__(self, content: str, name: str = "", **kwargs):
            self.content = content
            self.name = name

    LANGCHAIN_AVAILABLE = False

from ..core import Rails, rails_context
from ..types import Message, Role, RailState
from .base import BaseAdapter


class WrappedLangChainRunnable:
    """Transparent wrapper for LangChain runnables with Rails integration."""

    def __init__(self, runnable: Any, rails: Rails, adapter: "LangChainAdapter"):
        """Initialize the wrapped runnable.

        Args:
            runnable: The LangChain runnable to wrap
            rails: Rails instance for lifecycle management
            adapter: Parent adapter for message conversion
        """
        self._wrapped = runnable
        self._rails = rails
        self._adapter = adapter

    def __getattr__(self, name: str) -> Any:
        """Proxy all attribute access to the wrapped runnable."""
        attr = getattr(self._wrapped, name)

        # Intercept key methods
        if name in ["invoke", "ainvoke", "stream", "astream", "batch", "abatch"]:
            return self._create_interceptor(attr, name)

        return attr

    def __repr__(self) -> str:
        """Represent as the wrapped object."""
        return repr(self._wrapped)

    def __str__(self) -> str:
        """String representation."""
        return str(self._wrapped)

    def _create_interceptor(self, original_method: Any, method_name: str) -> Any:
        """Create an interceptor for a specific method."""

        if asyncio.iscoroutinefunction(original_method):

            @wraps(original_method)
            async def async_interceptor(
                input_data: Any, config: Optional[Any] = None, **kwargs
            ):
                return await self._process_with_rails_async(
                    input_data, original_method, config, **kwargs
                )

            return async_interceptor
        else:

            @wraps(original_method)
            def sync_interceptor(
                input_data: Any, config: Optional[Any] = None, **kwargs
            ):
                return self._process_with_rails_sync(
                    input_data, original_method, config, **kwargs
                )

            return sync_interceptor

    async def _process_with_rails_async(
        self, input_data: Any, method: Any, config: Optional[Any] = None, **kwargs
    ) -> Any:
        """Process input through Rails before calling the async method."""
        # Set Rails context
        token = rails_context.set(self._rails)

        try:
            # Handle different input types
            messages = await self._prepare_messages(input_data)

            # Process through Rails
            if messages:
                processed = await self._rails.process(messages)

                # Update counters
                await self._rails.store.increment("turns")
                if len(processed) > len(messages):
                    await self._rails.store.increment(
                        "injections", len(processed) - len(messages)
                    )

                # Convert back to LangChain format
                input_data = await self._adapter.from_rails_messages(processed)

            # Call original method
            result = await method(input_data, config=config, **kwargs)

            # Track tokens if available
            if hasattr(result, "usage_metadata"):
                await self._rails.store.increment(
                    "tokens", result.usage_metadata.get("total_tokens", 0)
                )
            elif isinstance(result, dict) and "usage_metadata" in result:
                await self._rails.store.increment(
                    "tokens", result["usage_metadata"].get("total_tokens", 0)
                )

            return result

        finally:
            rails_context.reset(token)

    def _process_with_rails_sync(
        self, input_data: Any, method: Any, config: Optional[Any] = None, **kwargs
    ) -> Any:
        """Process input through Rails before calling the sync method."""
        # Run async code in a thread to avoid event loop conflicts
        import concurrent.futures
        import threading

        async def _process(input_data_inner, method_inner, config_inner, kwargs_inner):
            token = rails_context.set(self._rails)

            try:
                # Handle different input types
                messages = await self._prepare_messages(input_data_inner)

                # Process through Rails
                processed_input = input_data_inner
                if messages:
                    processed = await self._rails.process(messages)

                    # Update counters
                    await self._rails.store.increment("turns")
                    if len(processed) > len(messages):
                        await self._rails.store.increment(
                            "injections", len(processed) - len(messages)
                        )

                    # Convert back to LangChain format
                    processed_input = await self._adapter.from_rails_messages(processed)

                # Call original method
                result = method_inner(
                    processed_input, config=config_inner, **kwargs_inner
                )

                # Track tokens if available
                if hasattr(result, "usage_metadata"):
                    await self._rails.store.increment(
                        "tokens", result.usage_metadata.get("total_tokens", 0)
                    )
                elif isinstance(result, dict) and "usage_metadata" in result:
                    await self._rails.store.increment(
                        "tokens", result["usage_metadata"].get("total_tokens", 0)
                    )

                return result

            finally:
                rails_context.reset(token)

        # Run in thread pool to avoid event loop issues
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run, _process(input_data, method, config, kwargs)
            )
            return future.result()

    async def _prepare_messages(self, input_data: Any) -> Optional[list[Message]]:
        """Convert input data to Rails messages."""
        if isinstance(input_data, list):
            # List of messages
            return await self._adapter.to_rails_messages(input_data)
        elif isinstance(input_data, dict) and "messages" in input_data:
            # Dict with messages key
            return await self._adapter.to_rails_messages(input_data["messages"])
        elif isinstance(input_data, str):
            # Simple string input
            return [Message(role=Role.USER, content=input_data)]
        elif hasattr(input_data, "messages"):
            # Object with messages attribute
            return await self._adapter.to_rails_messages(input_data.messages)
        else:
            # Can't process - return None to skip Rails processing
            return None


class LangChainAdapter(BaseAdapter):
    """Rails adapter for LangChain integration.

    This adapter wraps LangChain runnables to automatically inject Rails
    message processing. Once wrapped, the runnable behaves exactly like
    the original but with automatic Rails integration.

    Usage:
        from langchain_openai import ChatOpenAI
        from rails.adapters import LangChainAdapter

        # Set up Rails rules
        rails = Rails()
        rails.when(counter("turns") >= 3).inject(
            system("Consider wrapping up the conversation.")
        )

        # Create adapter and wrap the model
        adapter = LangChainAdapter(rails)
        llm = ChatOpenAI()
        wrapped_llm = adapter.wrap(llm)

        # Use exactly like the original - Rails injection happens automatically!
        result = wrapped_llm.invoke([{"role": "user", "content": "Hello!"}])
    """

    framework_name: str = "langchain"

    def __init__(self, rails: Rails | None = None):
        """Initialize the LangChain adapter.

        Args:
            rails: Rails instance for message injection
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install it with: pip install langchain-core"
            )
        super().__init__(rails=rails or Rails(), framework_name="langchain")

    async def to_rails_messages(self, framework_messages: list[Any]) -> list[Message]:
        """Convert LangChain messages to Rails format.

        Args:
            framework_messages: Messages in LangChain format

        Returns:
            Messages in Rails format
        """
        rails_messages = []
        for msg in framework_messages:
            if isinstance(msg, dict):
                # Dict format
                role_str = msg.get("role", "user")
                content = msg.get("content", "")

                # Map role strings to Role enum
                if role_str in ["user", "human"]:
                    role = Role.USER
                elif role_str in ["assistant", "ai"]:
                    role = Role.ASSISTANT
                elif role_str == "system":
                    role = Role.SYSTEM
                elif role_str == "tool":
                    role = Role.TOOL if hasattr(Role, "TOOL") else Role.USER
                elif role_str == "function":
                    role = Role.FUNCTION if hasattr(Role, "FUNCTION") else Role.USER
                else:
                    role = Role.USER  # Default fallback

                rails_messages.append(Message(role=role, content=content))
            elif hasattr(msg, "type") and hasattr(msg, "content"):
                # LangChain BaseMessage
                role = self._get_rails_role_from_type(msg.type)
                rails_messages.append(Message(role=role, content=msg.content))
            else:
                # Unknown format
                rails_messages.append(Message(role=Role.USER, content=str(msg)))

        return rails_messages

    async def from_rails_messages(self, rails_messages: list[Message]) -> list[Any]:
        """Convert Rails messages to LangChain format.

        Args:
            rails_messages: Messages in Rails format

        Returns:
            Messages in LangChain format
        """
        # If LangChain isn't available, return dict format
        if not LANGCHAIN_AVAILABLE:
            return [
                {"role": msg.role.value, "content": msg.content}
                for msg in rails_messages
            ]

        lc_messages = []
        for msg in rails_messages:
            if msg.role == Role.SYSTEM:
                lc_messages.append(SystemMessage(content=msg.content))
            elif msg.role == Role.ASSISTANT:
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == Role.USER:
                lc_messages.append(HumanMessage(content=msg.content))
            elif hasattr(Role, "TOOL") and msg.role == Role.TOOL:
                lc_messages.append(ToolMessage(content=msg.content, tool_call_id=""))
            elif hasattr(Role, "FUNCTION") and msg.role == Role.FUNCTION:
                lc_messages.append(FunctionMessage(content=msg.content, name=""))
            else:
                # Default to human message
                lc_messages.append(HumanMessage(content=msg.content))

        return lc_messages

    def _get_rails_role_from_type(self, message_type: str) -> Role:
        """Map LangChain message type to Rails Role enum.

        Args:
            message_type: LangChain message type

        Returns:
            Rails Role enum
        """
        type_to_role = {
            "system": Role.SYSTEM,
            "ai": Role.ASSISTANT,
            "assistant": Role.ASSISTANT,
            "human": Role.USER,
            "user": Role.USER,
            "tool": Role.TOOL if hasattr(Role, "TOOL") else Role.USER,
            "function": Role.FUNCTION if hasattr(Role, "FUNCTION") else Role.USER,
        }
        return type_to_role.get(message_type, Role.USER)

    async def wrap(self, runnable: Any) -> WrappedLangChainRunnable:
        """Wrap a LangChain runnable with Rails integration.

        The wrapped object behaves exactly like the original runnable,
        but automatically processes messages through Rails.

        Args:
            runnable: LangChain runnable to wrap

        Returns:
            Wrapped runnable with automatic Rails integration

        Example:
            llm = ChatOpenAI()
            wrapped = await adapter.wrap(llm)
            # Now use wrapped exactly like llm, but with Rails!
            result = wrapped.invoke(messages)
        """
        # Ensure Rails is initialized
        if self.rails.state == RailState.INITIALIZED:
            await self.rails.__aenter__()

        # Register store access for tools
        await self.register_store_access(self.rails.store)

        return WrappedLangChainRunnable(runnable, self.rails, self)

    def wrap_sync(self, runnable: Any) -> WrappedLangChainRunnable:
        """Synchronous version of wrap for convenience.

        Args:
            runnable: LangChain runnable to wrap

        Returns:
            Wrapped runnable with automatic Rails integration
        """
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # We're in an event loop, use run_coroutine_threadsafe
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.wrap(runnable))
                return future.result()
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.wrap(runnable))
            finally:
                loop.close()


def create_langchain_adapter(rails: Rails | None = None) -> LangChainAdapter:
    """Factory function to create a LangChain Rails adapter.

    Args:
        rails: Optional Rails instance

    Returns:
        Configured LangChainAdapter

    Example:
        from langchain_openai import ChatOpenAI
        from rails import Rails, counter, system
        from rails.adapters import create_langchain_adapter

        # Set up Rails
        rails = Rails()
        rails.when(counter("turns") >= 5).inject(
            system("Let's start wrapping up this conversation.")
        )

        # Create adapter and wrap model
        adapter = create_langchain_adapter(rails)
        llm = ChatOpenAI()
        wrapped = adapter.wrap_sync(llm)

        # Use exactly like the original!
        result = wrapped.invoke([
            {"role": "user", "content": "What's the weather like?"}
        ])
    """
    return LangChainAdapter(rails)


# Decorator for wrapping LangChain runnables with Rails
def with_rails(rails: Rails | None = None):
    """Decorator to wrap LangChain runnables with Rails.

    Args:
        rails: Rails instance to use

    Returns:
        Decorator function

    Example:
        from rails import Rails, counter, system

        rails = Rails()
        rails.when(counter("errors") > 0).inject(
            system("An error occurred. Please be more careful.")
        )

        @with_rails(rails)
        def create_chain():
            return ChatOpenAI() | StrOutputParser()

        # Now the chain includes Rails injection automatically
        chain = create_chain()
        result = chain.invoke("Hello!")
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            runnable = func(*args, **kwargs)
            adapter = LangChainAdapter(rails)
            return adapter.wrap_sync(runnable)

        return wrapper

    return decorator
