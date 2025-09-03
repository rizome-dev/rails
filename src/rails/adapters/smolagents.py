"""SmolAgents adapter for Rails integration.

This module provides seamless integration between Rails and SmolAgents,
allowing Rails conditional message injection to work with SmolAgents agents and tools.
"""

import asyncio
import inspect
from typing import Any, Optional
from functools import wraps

try:
    from smolagents import CodeAgent, ToolCallingAgent

    SMOLAGENTS_AVAILABLE = True
except ImportError:
    # Graceful degradation when SmolAgents is not installed
    CodeAgent = Any
    ToolCallingAgent = Any
    SMOLAGENTS_AVAILABLE = False

from ..core import Rails, rails_context
from ..types import Message, Role, RailState
from .base import BaseAdapter


class WrappedSmolAgent:
    """Transparent wrapper for SmolAgents agents with Rails integration."""

    def __init__(self, agent: Any, rails: Rails, adapter: "SmolAgentsAdapter"):
        """Initialize the wrapped agent.

        Args:
            agent: The SmolAgents agent to wrap
            rails: Rails instance for lifecycle management
            adapter: Parent adapter for message conversion
        """
        self._wrapped = agent
        self._rails = rails
        self._adapter = adapter

    def __getattr__(self, name: str) -> Any:
        """Proxy all attribute access to the wrapped agent."""
        attr = getattr(self._wrapped, name)

        # Intercept the run method
        if name == "run":
            return self._create_run_interceptor(attr)

        return attr

    def __repr__(self) -> str:
        """Represent as the wrapped object."""
        return repr(self._wrapped)

    def __str__(self) -> str:
        """String representation."""
        return str(self._wrapped)

    def _create_run_interceptor(self, original_run: Any) -> Any:
        """Create an interceptor for the run method."""

        @wraps(original_run)
        def run_interceptor(task: str, **kwargs):
            """Intercept run() calls and inject Rails processing."""
            return self._process_with_rails(task, original_run, **kwargs)

        return run_interceptor

    def _process_with_rails(self, task: str, method: Any, **kwargs) -> Any:
        """Process task through Rails before calling the original run method."""
        # Run async code in a thread to avoid event loop conflicts
        import concurrent.futures

        async def _process(task_inner, method_inner, kwargs_inner):
            token = rails_context.set(self._rails)

            try:
                # Create messages from task
                messages = [Message(role=Role.USER, content=task_inner)]

                # Update counters BEFORE processing (so conditions can see the current run)
                await self._rails.store.increment("turns")
                await self._rails.store.increment("agent_runs")

                # Process through Rails
                processed = await self._rails.process(messages)

                if len(processed) > len(messages):
                    await self._rails.store.increment(
                        "injections", len(processed) - len(messages)
                    )

                # Build enhanced task with injected context
                enhanced_task = self._build_enhanced_task(
                    task_inner, processed, messages
                )

                # Call original method with enhanced task
                result = method_inner(enhanced_task, **kwargs_inner)

                # Track metrics
                if hasattr(result, "tool_calls") or "```" in str(result):
                    await self._rails.store.increment("tool_calls")

                # More flexible code generation detection
                result_str = str(result)
                if (
                    "```python" in result_str
                    or "```code" in result_str
                    or "def " in result_str
                    or "class " in result_str
                ):
                    await self._rails.store.increment("code_generations")

                return result

            finally:
                rails_context.reset(token)

        # Run in thread pool to avoid event loop issues
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, _process(task, method, kwargs))
            return future.result()

    def _build_enhanced_task(
        self,
        original_task: str,
        processed_messages: list[Message],
        original_messages: list[Message],
    ) -> str:
        """Build an enhanced task string with Rails injections.

        Args:
            original_task: The original task string
            processed_messages: Messages after Rails processing
            original_messages: Original messages before Rails

        Returns:
            Enhanced task string with any injected context
        """
        # Check for injected messages
        if len(processed_messages) > len(original_messages):
            injected = processed_messages[len(original_messages) :]

            # Build context from injected messages
            context_parts = []
            for msg in injected:
                if msg.role == Role.SYSTEM:
                    context_parts.append(f"System: {msg.content}")
                elif msg.role == Role.ASSISTANT:
                    context_parts.append(f"Assistant: {msg.content}")
                else:
                    context_parts.append(msg.content)

            if context_parts:
                context = "\n".join(context_parts)
                return f"{context}\n\nTask: {original_task}"

        return original_task


class WrappedCodeAgent(WrappedSmolAgent):
    """Specialized wrapper for SmolAgents CodeAgent with enhanced tracking."""

    def _process_with_rails(self, task: str, method: Any, **kwargs) -> Any:
        """Enhanced processing with code-specific tracking."""
        result = super()._process_with_rails(task, method, **kwargs)

        # Additional code-specific tracking
        result_str = str(result)

        # Track metrics in a thread
        import concurrent.futures

        async def _track():
            if "def " in result_str or "class " in result_str:
                await self._rails.store.increment("python_functions")

            if "import " in result_str:
                await self._rails.store.increment("imports_used")

            if "Error" in result_str or "Exception" in result_str:
                await self._rails.store.increment("errors_encountered")

        # Run in thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(asyncio.run, _track()).result()

        return result


class SmolAgentsAdapter(BaseAdapter):
    """Rails adapter for SmolAgents integration.

    This adapter wraps SmolAgents agents to automatically inject Rails
    message processing. Once wrapped, the agent behaves exactly like
    the original but with automatic Rails integration.

    Usage:
        from smolagents import CodeAgent
        from rails.adapters import SmolAgentsAdapter

        # Set up Rails rules
        rails = Rails()
        rails.when(counter("tool_calls") >= 3).inject(
            system("You've used several tools. Consider if you have enough information.")
        )

        # Create adapter and wrap agent
        adapter = SmolAgentsAdapter(rails)
        agent = CodeAgent(tools=[], model="gpt-4")
        wrapped = adapter.wrap(agent)

        # Use exactly like the original - Rails injection happens automatically!
        result = wrapped.run("Analyze this data and create a visualization")
    """

    framework_name: str = "smolagents"

    def __init__(self, rails: Rails | None = None):
        """Initialize the SmolAgents adapter.

        Args:
            rails: Rails instance for message injection
        """
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError(
                "SmolAgents is not installed. Install it with: pip install smolagents"
            )
        super().__init__(rails=rails or Rails(), framework_name="smolagents")

    async def to_rails_messages(self, framework_messages: list[Any]) -> list[Message]:
        """Convert SmolAgents format messages to Rails format.

        Args:
            framework_messages: Messages in SmolAgents dict format

        Returns:
            Messages in Rails format
        """
        rails_messages = []
        for msg in framework_messages:
            if isinstance(msg, dict):
                role_str = msg.get("role", "user")
                content = msg.get("content", "")

                # Map role strings to Role enum
                if role_str in ["user", "human"]:
                    role = Role.USER
                elif role_str in ["assistant", "ai"]:
                    role = Role.ASSISTANT
                elif role_str == "system":
                    role = Role.SYSTEM
                else:
                    role = Role.USER  # Default fallback

                rails_messages.append(Message(role=role, content=content))
            elif isinstance(msg, str):
                # Plain string is a user message
                rails_messages.append(Message(role=Role.USER, content=msg))
            else:
                # Handle other message formats if needed
                rails_messages.append(Message(role=Role.USER, content=str(msg)))

        return rails_messages

    async def from_rails_messages(self, rails_messages: list[Message]) -> list[Any]:
        """Convert Rails messages to SmolAgents dict format.

        Args:
            rails_messages: Messages in Rails format

        Returns:
            Messages in SmolAgents dict format
        """
        framework_messages = []
        for msg in rails_messages:
            framework_messages.append({"role": msg.role.value, "content": msg.content})
        return framework_messages

    async def wrap(self, agent: Any) -> WrappedSmolAgent:
        """Wrap a SmolAgents agent with Rails integration.

        The wrapped agent behaves exactly like the original,
        but automatically processes tasks through Rails.

        Args:
            agent: SmolAgents agent to wrap

        Returns:
            Wrapped agent with automatic Rails integration

        Example:
            agent = CodeAgent(tools=[], model="gpt-4")
            wrapped = await adapter.wrap(agent)
            # Now use wrapped exactly like agent, but with Rails!
            result = wrapped.run("Create a data analysis script")
        """
        # Ensure Rails is initialized
        if self.rails.state == RailState.INITIALIZED:
            await self.rails.__aenter__()

        # Register store access for tools
        await self.register_store_access(self.rails.store)

        # Return specialized wrapper for CodeAgent if applicable
        if hasattr(agent, "__class__") and "code" in agent.__class__.__name__.lower():
            return WrappedCodeAgent(agent, self.rails, self)

        return WrappedSmolAgent(agent, self.rails, self)

    def wrap_sync(self, agent: Any) -> WrappedSmolAgent:
        """Synchronous version of wrap for convenience.

        Args:
            agent: SmolAgents agent to wrap

        Returns:
            Wrapped agent with automatic Rails integration
        """
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # We're in an event loop, use run_coroutine_threadsafe
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.wrap(agent))
                return future.result()
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.wrap(agent))
            finally:
                loop.close()


class CodeAgentAdapter(SmolAgentsAdapter):
    """Specialized Rails adapter for SmolAgents CodeAgent.

    This adapter provides enhanced tracking for code generation tasks.

    Usage:
        from smolagents import CodeAgent
        from rails.adapters import CodeAgentAdapter

        # Set up Rails for code-specific rules
        rails = Rails()
        rails.when(counter("code_generations") >= 2).inject(
            system("You've generated multiple code snippets. Explain how they work together.")
        )

        # Create adapter and wrap agent
        adapter = CodeAgentAdapter(rails)
        agent = CodeAgent(tools=[], model="gpt-4")
        wrapped = adapter.wrap_sync(agent)

        # Use exactly like the original!
        result = wrapped.run("Create a fibonacci function")
    """

    async def wrap(self, agent: Any) -> WrappedCodeAgent:
        """Wrap a CodeAgent with specialized tracking.

        Args:
            agent: CodeAgent to wrap

        Returns:
            Wrapped agent with code-specific Rails integration
        """
        # Ensure Rails is initialized
        if self.rails.state == RailState.INITIALIZED:
            await self.rails.__aenter__()

        # Register store access for tools
        await self.register_store_access(self.rails.store)

        return WrappedCodeAgent(agent, self.rails, self)


def create_smolagents_adapter(rails: Rails | None = None) -> SmolAgentsAdapter:
    """Factory function to create a SmolAgents Rails adapter.

    Args:
        rails: Optional Rails instance

    Returns:
        Configured SmolAgentsAdapter

    Example:
        from smolagents import CodeAgent
        from rails import Rails, counter, system
        from rails.adapters import create_smolagents_adapter

        # Set up Rails
        rails = Rails()
        rails.when(counter("turns") >= 3).inject(
            system("Consider breaking down the task into smaller steps.")
        )

        # Create adapter and wrap agent
        adapter = create_smolagents_adapter(rails)
        agent = CodeAgent(tools=[], model="gpt-4")
        wrapped = adapter.wrap_sync(agent)

        # Use exactly like the original!
        result = wrapped.run("Analyze sales data and create visualizations")
    """
    return SmolAgentsAdapter(rails)


def create_code_agent_adapter(rails: Rails | None = None) -> CodeAgentAdapter:
    """Factory function to create a specialized CodeAgent adapter.

    Args:
        rails: Optional Rails instance

    Returns:
        Configured CodeAgentAdapter with enhanced tracking
    """
    return CodeAgentAdapter(rails)


# Decorator for wrapping SmolAgents agents with Rails
def with_rails(rails: Rails | None = None):
    """Decorator to wrap SmolAgents agent creation with Rails.

    Args:
        rails: Rails instance to use

    Returns:
        Decorator function

    Example:
        from rails import Rails, counter, system

        rails = Rails()
        rails.when(counter("errors") > 0).inject(
            system("An error was encountered. Please handle it gracefully.")
        )

        @with_rails(rails)
        def create_agent():
            return CodeAgent(tools=[], model="gpt-4")

        # Now the agent includes Rails injection automatically
        agent = create_agent()
        result = agent.run("Create a data analysis script")
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            agent = func(*args, **kwargs)
            # Auto-detect agent type
            if (
                hasattr(agent, "__class__")
                and "code" in agent.__class__.__name__.lower()
            ):
                adapter = CodeAgentAdapter(rails)
            else:
                adapter = SmolAgentsAdapter(rails)
            return adapter.wrap_sync(agent)

        return wrapper

    return decorator
