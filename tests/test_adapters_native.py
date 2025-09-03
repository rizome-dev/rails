"""Test adapter functionality for LangChain and SmolAgents."""

import asyncio
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Any, List, Dict

from rails import Rails, Message, Role, current_rails, counter
from rails.conditions import LambdaCondition

# Patch the availability checks before importing adapters
with patch("rails.adapters.langchain.LANGCHAIN_AVAILABLE", True):
    from rails.adapters.langchain import LangChainAdapter, WrappedLangChainRunnable

with patch("rails.adapters.smolagents.SMOLAGENTS_AVAILABLE", True):
    from rails.adapters.smolagents import (
        SmolAgentsAdapter,
        WrappedSmolAgent,
        CodeAgentAdapter,
    )


# Mock LangChain classes
class MockLangChainRunnable:
    """Mock LangChain runnable for testing."""

    def __init__(self):
        self.invoke_count = 0
        self.ainvoke_count = 0
        self.last_input = None

    def invoke(self, input_data: Any, config=None, **kwargs):
        """Mock synchronous invoke."""
        self.invoke_count += 1
        self.last_input = input_data
        return {
            "content": f"Response to {len(input_data)} messages",
            "usage_metadata": {"total_tokens": 100},
        }

    async def ainvoke(self, input_data: Any, config=None, **kwargs):
        """Mock asynchronous invoke."""
        self.ainvoke_count += 1
        self.last_input = input_data
        return {
            "content": f"Async response to {len(input_data)} messages",
            "usage_metadata": {"total_tokens": 150},
        }


# Mock SmolAgents classes
class MockSmolAgent:
    """Mock SmolAgents agent for testing."""

    def __init__(self):
        self.run_count = 0
        self.last_task = None

    def run(self, task: str, **kwargs):
        """Mock run method."""
        self.run_count += 1
        self.last_task = task
        # Simulate code generation
        if "fibonacci" in task.lower():
            return "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        return f"Completed task: {task[:50]}"


class MockCodeAgent(MockSmolAgent):
    """Mock SmolAgents CodeAgent for testing."""

    def __init__(self, tools=None, model=None):
        super().__init__()
        self.tools = tools or []
        self.model = model


@pytest.mark.asyncio
@patch("rails.adapters.langchain.LANGCHAIN_AVAILABLE", True)
async def test_langchain_wrapper_sync():
    """Test LangChain wrapper with synchronous invoke."""
    # Setup
    rails = Rails()
    counter = 0

    # Add injection rule
    async def inject_after_3(store):
        count = await store.get_counter("turns", 0)
        return count >= 3

    rails.add_rule(
        condition=LambdaCondition(func=inject_after_3),
        action=lambda msgs: msgs
        + [Message(role=Role.SYSTEM, content="Injected message")],
        name="inject_test",
    )

    # Create adapter and wrap runnable
    adapter = LangChainAdapter(rails)
    mock_runnable = MockLangChainRunnable()

    async with rails:
        wrapped = await adapter.wrap(mock_runnable)

        # Verify wrapper is returned
        assert isinstance(wrapped, WrappedLangChainRunnable)

        # Test invoke - should work exactly like the original
        messages = [{"role": "user", "content": "Hello"}]
        result = wrapped.invoke(messages)

        # Verify method was called
        assert mock_runnable.invoke_count == 1
        assert result["content"] == "Response to 1 messages"

        # Test multiple calls to trigger injection
        for i in range(3):
            result = wrapped.invoke([{"role": "user", "content": f"Message {i}"}])

        # After 3 turns, injection should occur
        assert await rails.store.get_counter("turns") >= 3
        assert await rails.store.get_counter("injections") > 0


@pytest.mark.asyncio
@patch("rails.adapters.langchain.LANGCHAIN_AVAILABLE", True)
async def test_langchain_wrapper_async():
    """Test LangChain wrapper with asynchronous ainvoke."""
    # Setup
    rails = Rails()

    # Add injection rule
    rails.add_rule(
        condition=counter("turns") >= 2,
        action=lambda msgs: msgs
        + [Message(role=Role.SYSTEM, content="Async injection")],
        name="async_inject",
    )

    # Create adapter and wrap runnable
    adapter = LangChainAdapter(rails)
    mock_runnable = MockLangChainRunnable()

    async with rails:
        wrapped = await adapter.wrap(mock_runnable)

        # Test ainvoke
        messages = [{"role": "user", "content": "Async hello"}]
        result = await wrapped.ainvoke(messages)

        # Verify async method was called
        assert mock_runnable.ainvoke_count == 1
        assert "Async response" in result["content"]

        # Test token tracking
        assert await rails.store.get_counter("tokens") > 0


@pytest.mark.asyncio
@patch("rails.adapters.smolagents.SMOLAGENTS_AVAILABLE", True)
async def test_smolagents_wrapper():
    """Test SmolAgents wrapper."""
    # Setup
    rails = Rails()

    # Add injection rule for tool usage
    rails.add_rule(
        condition=counter("agent_runs") >= 2,
        action=lambda msgs: msgs
        + [Message(role=Role.SYSTEM, content="Consider breaking down the task")],
        name="guide_breakdown",
    )

    # Create adapter and wrap agent
    adapter = SmolAgentsAdapter(rails)
    mock_agent = MockSmolAgent()

    async with rails:
        wrapped = await adapter.wrap(mock_agent)

        # Verify wrapper is returned
        assert isinstance(wrapped, WrappedSmolAgent)

        # Test run method
        result = wrapped.run("Create a simple function")

        # Verify method was called
        assert mock_agent.run_count == 1
        assert "Completed task" in result

        # Test injection after multiple runs
        result = wrapped.run("Another task")

        # Check that injection occurred
        assert await rails.store.get_counter("agent_runs") >= 2
        assert (
            "Consider breaking down" in mock_agent.last_task
            or await rails.store.get_counter("injections") > 0
        )


@pytest.mark.asyncio
@patch("rails.adapters.smolagents.SMOLAGENTS_AVAILABLE", True)
async def test_code_agent_specialized_tracking():
    """Test CodeAgent specialized tracking."""
    # Setup
    rails = Rails()

    # Add code-specific rule
    rails.add_rule(
        condition=counter("code_generations") >= 1,
        action=lambda msgs: msgs
        + [Message(role=Role.SYSTEM, content="Add docstrings to your code")],
        name="docstring_reminder",
    )

    # Create specialized adapter
    adapter = CodeAgentAdapter(rails)
    mock_agent = MockCodeAgent()

    async with rails:
        wrapped = await adapter.wrap(mock_agent)

        # Test code generation
        result = wrapped.run("Create a fibonacci function")

        # Verify code was generated and tracked
        assert "fibonacci" in result
        assert await rails.store.get_counter("code_generations") > 0
        assert await rails.store.get_counter("python_functions") > 0


@pytest.mark.asyncio
@patch("rails.adapters.langchain.LANGCHAIN_AVAILABLE", True)
async def test_rails_context_in_tools():
    """Test that current_rails() works within tool execution."""
    # Setup
    rails = Rails()
    tool_called = False
    rails_accessible = False

    def mock_tool():
        nonlocal tool_called, rails_accessible
        tool_called = True
        try:
            r = current_rails()
            rails_accessible = r == rails
        except RuntimeError:
            rails_accessible = False
        return "Tool result"

    # Create adapter with mock runnable that calls the tool
    adapter = LangChainAdapter(rails)
    mock_runnable = Mock()

    def invoke_with_tool(messages, **kwargs):
        # Simulate tool call during processing
        mock_tool()
        return {"content": "Response with tool"}

    mock_runnable.invoke = invoke_with_tool

    async with rails:
        wrapped = await adapter.wrap(mock_runnable)

        # Execute
        result = wrapped.invoke([{"role": "user", "content": "Use tool"}])

        # Verify tool was called and Rails was accessible
        assert tool_called
        assert rails_accessible


@pytest.mark.asyncio
@patch("rails.adapters.langchain.LANGCHAIN_AVAILABLE", True)
@patch("rails.adapters.smolagents.SMOLAGENTS_AVAILABLE", True)
async def test_message_conversion():
    """Test message conversion between Rails and framework formats."""
    # Test LangChain conversion
    lc_adapter = LangChainAdapter(Rails())

    # Rails to LangChain
    rails_messages = [
        Message(role=Role.USER, content="User message"),
        Message(role=Role.ASSISTANT, content="Assistant message"),
        Message(role=Role.SYSTEM, content="System message"),
    ]

    lc_messages = await lc_adapter.from_rails_messages(rails_messages)
    assert len(lc_messages) == 3

    # LangChain to Rails
    dict_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "system", "content": "Be helpful"},
    ]

    converted = await lc_adapter.to_rails_messages(dict_messages)
    assert len(converted) == 3
    assert converted[0].role == Role.USER
    assert converted[1].role == Role.ASSISTANT
    assert converted[2].role == Role.SYSTEM

    # Test SmolAgents conversion
    sa_adapter = SmolAgentsAdapter(Rails())

    # Rails to SmolAgents
    sa_messages = await sa_adapter.from_rails_messages(rails_messages)
    assert len(sa_messages) == 3
    assert all(isinstance(m, dict) for m in sa_messages)

    # SmolAgents to Rails
    converted_sa = await sa_adapter.to_rails_messages(sa_messages)
    assert len(converted_sa) == 3


@pytest.mark.asyncio
@patch("rails.adapters.langchain.LANGCHAIN_AVAILABLE", True)
@patch("rails.adapters.smolagents.SMOLAGENTS_AVAILABLE", True)
async def test_wrap_sync_convenience():
    """Test synchronous wrap convenience method."""
    rails = Rails()

    # Test LangChain
    lc_adapter = LangChainAdapter(rails)
    mock_runnable = MockLangChainRunnable()
    wrapped = lc_adapter.wrap_sync(mock_runnable)
    assert isinstance(wrapped, WrappedLangChainRunnable)

    # Test SmolAgents
    sa_adapter = SmolAgentsAdapter(rails)
    mock_agent = MockSmolAgent()
    wrapped = sa_adapter.wrap_sync(mock_agent)
    assert isinstance(wrapped, WrappedSmolAgent)


@pytest.mark.asyncio
@patch("rails.adapters.langchain.LANGCHAIN_AVAILABLE", True)
@patch("rails.adapters.smolagents.SMOLAGENTS_AVAILABLE", True)
async def test_decorator_pattern():
    """Test the @with_rails decorator pattern."""
    rails = Rails()

    # Test with LangChain
    from rails.adapters.langchain import with_rails as lc_with_rails

    @lc_with_rails(rails)
    def create_langchain_model():
        return MockLangChainRunnable()

    wrapped_model = create_langchain_model()
    assert hasattr(wrapped_model, "invoke")
    assert hasattr(wrapped_model, "_wrapped")

    # Test with SmolAgents
    from rails.adapters.smolagents import with_rails as sa_with_rails

    @sa_with_rails(rails)
    def create_agent():
        return MockSmolAgent()

    wrapped_agent = create_agent()
    assert hasattr(wrapped_agent, "run")
    assert hasattr(wrapped_agent, "_wrapped")


@pytest.mark.asyncio
@patch("rails.adapters.smolagents.SMOLAGENTS_AVAILABLE", True)
async def test_complex_injection_scenario():
    """Test complex scenario with multiple injections and conditions."""
    rails = Rails()

    # Add multiple rules
    rails.add_rule(
        condition=counter("turns") == 2,
        action=lambda msgs: msgs
        + [Message(role=Role.SYSTEM, content="First injection")],
        name="first",
    )

    rails.add_rule(
        condition=counter("turns") == 4,
        action=lambda msgs: msgs
        + [Message(role=Role.SYSTEM, content="Second injection")],
        name="second",
    )

    # Create and wrap agent
    adapter = SmolAgentsAdapter(rails)
    mock_agent = MockSmolAgent()

    async with rails:
        wrapped = await adapter.wrap(mock_agent)

        # Run multiple times
        tasks_executed = []
        for i in range(5):
            result = wrapped.run(f"Task {i}")
            tasks_executed.append(mock_agent.last_task)

        # Check injections occurred at right times
        assert await rails.store.get_counter("turns") == 5
        assert await rails.store.get_counter("injections") >= 2

        # Verify injections were added to tasks
        assert any("First injection" in task for task in tasks_executed)
        assert any("Second injection" in task for task in tasks_executed)


@pytest.mark.asyncio
@patch("rails.adapters.langchain.LANGCHAIN_AVAILABLE", True)
async def test_error_handling():
    """Test error handling in native wrappers."""
    rails = Rails()

    # Create runnable that raises an error
    mock_runnable = Mock()
    mock_runnable.invoke = Mock(side_effect=ValueError("Test error"))

    adapter = LangChainAdapter(rails)

    async with rails:
        wrapped = await adapter.wrap(mock_runnable)

        # Should propagate the error
        with pytest.raises(ValueError, match="Test error"):
            wrapped.invoke([{"role": "user", "content": "Test"}])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
