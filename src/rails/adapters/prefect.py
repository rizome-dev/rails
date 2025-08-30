"""Prefect integration for Rails lifecycle orchestration."""

import inspect
from collections.abc import Callable
from typing import Any

from loguru import logger

from ..core import Rails, rails_context
from ..types import Message, Role
from .base import BaseAdapter


class PrefectAdapter(BaseAdapter):
    """Rails adapter for Prefect workflows.
    
    This adapter enables Rails lifecycle orchestration within Prefect flows and tasks,
    allowing sophisticated state management and conditional message injection for
    AI agents running in Prefect workflows.
    
    Usage:
        from prefect import flow, task
        from rails import Rails
        from rails.adapters.prefect import PrefectAdapter
        from rails.conditions import counter, queue
        
        # Create Rails with lifecycle rules
        rails = Rails()
        rails.add_rule(
            condition=counter("tasks_completed") >= 5,
            action=system("Great progress! 5 tasks completed.")
        )
        rails.add_rule(
            condition=queue("errors").length > 0,
            action=system("Errors detected. Consider error recovery.")
        )
        
        # Create adapter
        adapter = PrefectAdapter(rails=rails)
        
        @task
        @adapter.task_decorator
        async def ai_task(messages: list[Message]) -> list[Message]:
            # Task automatically has access to Rails store
            rails = current_rails()
            await rails.store.increment("tasks_completed")
            
            # Process messages (Rails rules automatically applied)
            return messages
        
        @flow
        @adapter.flow_decorator
        async def ai_workflow():
            # Flow has Rails lifecycle management
            messages = [Message(role=Role.USER, content="Start workflow")]
            
            # Tasks will have Rails context
            result = await ai_task(messages)
            
            return result
    """

    framework_name: str = "prefect"
    enable_state_tracking: bool = True
    enable_artifact_logging: bool = True

    async def to_rails_messages(self, framework_messages: list[Any]) -> list[Message]:
        """Convert Prefect task inputs to Rails messages."""
        rails_messages = []

        for msg in framework_messages:
            if isinstance(msg, Message):
                rails_messages.append(msg)
            elif isinstance(msg, dict):
                rails_messages.append(Message(
                    role=Role(msg.get("role", "user")),
                    content=msg.get("content", "")
                ))
            else:
                # Assume it's a string or other content
                rails_messages.append(Message(
                    role=Role.USER,
                    content=str(msg)
                ))

        return rails_messages

    async def from_rails_messages(self, rails_messages: list[Message]) -> list[Any]:
        """Convert Rails messages back to Prefect format."""
        # Return as dicts for Prefect artifact storage
        return [msg.model_dump() for msg in rails_messages]

    def task_decorator(self, func: Callable) -> Callable:
        """Decorator to add Rails lifecycle to Prefect tasks.
        
        Args:
            func: Prefect task function
            
        Returns:
            Wrapped function with Rails integration
        """
        if inspect.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                # Set Rails context for this task
                token = rails_context.set(self.rails)

                try:
                    # Track task start
                    if self.enable_state_tracking:
                        await self.rails.store.increment("prefect_tasks_started")
                        await self.rails.store.set("current_task", func.__name__)

                    # Process messages if first arg is messages
                    if args and isinstance(args[0], list):
                        messages = await self.process_messages(args[0])
                        args = (messages,) + args[1:]

                    # Execute task
                    result = await func(*args, **kwargs)

                    # Track task completion
                    if self.enable_state_tracking:
                        await self.rails.store.increment("prefect_tasks_completed")

                    # Process result if it's messages
                    if isinstance(result, list) and result and isinstance(result[0], (dict, Message)):
                        result = await self.process_messages(result)

                    return result

                except Exception as e:
                    # Track errors
                    await self.rails.store.increment("prefect_task_errors")
                    await self.rails.store.push_queue("error_queue", {
                        "task": func.__name__,
                        "error": str(e)
                    })
                    raise

                finally:
                    # Reset context
                    rails_context.reset(token)
        else:
            def wrapper(*args, **kwargs):
                # Sync version - limited Rails support
                token = rails_context.set(self.rails)
                try:
                    result = func(*args, **kwargs)
                    self.rails.store.increment_sync("prefect_tasks_completed")
                    return result
                except Exception:
                    self.rails.store.increment_sync("prefect_task_errors")
                    raise
                finally:
                    rails_context.reset(token)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    def flow_decorator(self, func: Callable) -> Callable:
        """Decorator to add Rails lifecycle to Prefect flows.
        
        Args:
            func: Prefect flow function
            
        Returns:
            Wrapped function with Rails integration
        """
        if inspect.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                # Initialize Rails lifecycle for flow
                async with self.rails:
                    # Track flow start
                    if self.enable_state_tracking:
                        await self.rails.store.increment("prefect_flows_started")
                        await self.rails.store.set("current_flow", func.__name__)

                    try:
                        # Execute flow with Rails context
                        result = await func(*args, **kwargs)

                        # Track flow completion
                        if self.enable_state_tracking:
                            await self.rails.store.increment("prefect_flows_completed")

                        # Log artifacts if enabled
                        if self.enable_artifact_logging:
                            metrics = await self.rails.emit_metrics()
                            logger.info(f"Rails metrics for flow {func.__name__}: {metrics}")

                        return result

                    except Exception:
                        # Track flow errors
                        await self.rails.store.increment("prefect_flow_errors")
                        raise
        else:
            def wrapper(*args, **kwargs):
                # Sync version
                import asyncio

                async def run():
                    async with self.rails:
                        self.rails.store.increment_sync("prefect_flows_started")
                        return func(*args, **kwargs)

                return asyncio.run(run())

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    def create_rails_task(self, name: str = "rails_process") -> Callable:
        """Create a Prefect task that processes messages through Rails.
        
        Args:
            name: Task name
            
        Returns:
            Prefect task function
            
        Example:
            from prefect import flow
            
            rails_task = adapter.create_rails_task("inject_context")
            
            @flow
            async def my_flow():
                messages = [...]
                processed = await rails_task(messages)
                return processed
        """
        async def rails_task(messages: list[Message]) -> list[Message]:
            """Process messages through Rails lifecycle orchestration."""
            return await self.process_messages(messages)

        rails_task.__name__ = name
        return rails_task

    def create_state_hook(self) -> Callable:
        """Create a Prefect state hook that updates Rails store.
        
        Returns:
            State hook function
            
        Example:
            from prefect import task
            
            @task(on_completion=[adapter.create_state_hook()])
            async def my_task():
                ...
        """
        async def state_hook(task, task_run, state):
            """Update Rails store based on Prefect state changes."""
            rails = rails_context.get()
            if rails:
                await rails.store.set(f"prefect_task_{task.name}_state", state.type)

                if state.is_completed():
                    await rails.store.increment("prefect_successful_tasks")
                elif state.is_failed():
                    await rails.store.increment("prefect_failed_tasks")
                    await rails.store.push_queue("failed_tasks", task.name)

        return state_hook


def rails_task(rails: Rails, name: str = "rails_lifecycle") -> Callable:
    """Create a standalone Prefect task with Rails lifecycle.
    
    Args:
        rails: Rails instance
        name: Task name
        
    Returns:
        Prefect task function
        
    Example:
        from prefect import flow, task
        from rails import Rails
        from rails.adapters.prefect import rails_task
        
        rails = Rails()
        # Configure rails...
        
        process_messages = rails_task(rails, "process_messages")
        
        @flow
        async def my_flow():
            messages = [...]
            result = await process_messages(messages)
            return result
    """
    adapter = PrefectAdapter(rails=rails)

    # Try to import Prefect and create a proper task
    try:
        from prefect import task

        @task(name=name)
        async def rails_lifecycle_task(messages: list[Message]) -> list[Message]:
            """Process messages through Rails lifecycle orchestration."""
            return await adapter.process_messages(messages)

        return rails_lifecycle_task

    except ImportError:
        # Prefect not installed, return plain function
        logger.warning("Prefect not installed, returning plain function")
        return adapter.create_rails_task(name)


def rails_flow(rails: Rails, name: str = "rails_flow") -> Callable:
    """Create a Prefect flow with Rails lifecycle management.
    
    Args:
        rails: Rails instance
        name: Flow name
        
    Returns:
        Flow decorator
        
    Example:
        from rails import Rails
        from rails.adapters.prefect import rails_flow
        
        rails = Rails()
        # Configure rails...
        
        @rails_flow(rails, "ai_workflow")
        async def my_workflow():
            # Flow has Rails lifecycle
            ...
    """
    adapter = PrefectAdapter(rails=rails)

    def decorator(func):
        # Try to import Prefect
        try:
            from prefect import flow

            # Apply Prefect flow decorator first
            prefect_flow = flow(name=name)(func)

            # Then apply Rails decorator
            return adapter.flow_decorator(prefect_flow)

        except ImportError:
            # Prefect not installed, just apply Rails decorator
            logger.warning("Prefect not installed, applying Rails decorator only")
            return adapter.flow_decorator(func)

    return decorator
