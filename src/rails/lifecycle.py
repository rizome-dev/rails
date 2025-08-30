"""
Composable lifecycle management for Rails.

This module provides the foundation for modular, composable lifecycle functions
inspired by the reference implementation patterns. It enables tools and workflows
to define setup/cleanup logic that can be easily composed and managed.
"""

import asyncio
import inspect
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class LifecycleFunction:
    """Represents a modular lifecycle function with setup and cleanup phases."""

    name: str
    func: Callable
    priority: int = 0
    dependencies: list[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class LifecycleRegistry:
    """Registry for managing lifecycle functions and their execution order."""

    def __init__(self):
        self._functions: dict[str, LifecycleFunction] = {}
        self._active_contexts: dict[str, Any] = {}

    def register(self, name: str, func: Callable, priority: int = 0,
                 dependencies: list[str] = None) -> None:
        """Register a lifecycle function."""
        lifecycle_func = LifecycleFunction(
            name=name,
            func=func,
            priority=priority,
            dependencies=dependencies or []
        )
        self._functions[name] = lifecycle_func

    def get_execution_order(self) -> list[str]:
        """Get lifecycle functions in execution order based on priorities and dependencies."""
        # Simple topological sort for dependencies + priority ordering
        functions = list(self._functions.values())

        # Sort by priority first (higher priority = earlier execution)
        functions.sort(key=lambda f: f.priority, reverse=True)

        # TODO: Add proper dependency resolution for complex cases
        return [f.name for f in functions]

    def get_function(self, name: str) -> LifecycleFunction | None:
        """Get a lifecycle function by name."""
        return self._functions.get(name)

    def list_functions(self) -> list[str]:
        """List all registered lifecycle function names."""
        return list(self._functions.keys())


# Global lifecycle registry
_lifecycle_registry = LifecycleRegistry()


def lifecycle_function(name: str | None = None, priority: int = 0,
                      dependencies: list[str] = None):
    """
    Decorator for creating modular lifecycle functions.
    
    Lifecycle functions should be async context managers that yield control
    to the main execution and handle cleanup on exit.
    
    Args:
        name: Function name (defaults to function __name__)
        priority: Execution priority (higher = earlier, default 0)
        dependencies: List of function names this depends on
        
    Usage:
        @lifecycle_function(priority=10)
        async def setup_database(rails):
            # Setup phase
            connection = await create_db_connection()
            rails.store.set_sync('db_connection', connection)
            
            yield  # Main execution happens here
            
            # Cleanup phase
            await connection.close()
            
        # Use with Rails
        rails.with_lifecycle(setup_database).when(condition).inject(message)
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__

        # Register the function
        _lifecycle_registry.register(
            name=func_name,
            func=func,
            priority=priority,
            dependencies=dependencies
        )

        # Mark function as lifecycle function
        func._is_lifecycle_function = True
        func._lifecycle_name = func_name
        func._lifecycle_priority = priority
        func._lifecycle_dependencies = dependencies or []

        return func

    return decorator


@asynccontextmanager
async def with_lifecycle_functions(rails_instance, lifecycle_funcs: list[str | Callable]):
    """
    Context manager for composing multiple lifecycle functions.
    
    Args:
        rails_instance: Rails instance to pass to lifecycle functions
        lifecycle_funcs: List of function names or callable functions
    """
    # Resolve function references
    resolved_funcs = []
    for func_ref in lifecycle_funcs:
        if isinstance(func_ref, str):
            lifecycle_func = _lifecycle_registry.get_function(func_ref)
            if lifecycle_func is None:
                raise ValueError(f"Lifecycle function '{func_ref}' not found")
            resolved_funcs.append(lifecycle_func.func)
        else:
            resolved_funcs.append(func_ref)

    # Start all lifecycle functions
    contexts = []
    try:
        for func in resolved_funcs:
            if inspect.isasyncgenfunction(func):
                # Async generator function - use as context manager
                ctx = func(rails_instance)
                await ctx.__aenter__()
                contexts.append(ctx)
            else:
                # Regular async function - call it
                result = await func(rails_instance)
                contexts.append(result)

        # Yield control for main execution
        yield

    finally:
        # Cleanup in reverse order
        for ctx in reversed(contexts):
            try:
                if hasattr(ctx, '__aexit__'):
                    await ctx.__aexit__(None, None, None)
            except Exception as e:
                # Log error but continue cleanup
                print(f"Error in lifecycle cleanup: {e}")


class LifecycleManager:
    """Manager for orchestrating lifecycle functions and workflow execution."""

    def __init__(self, rails_instance):
        self.rails = rails_instance
        self._active_functions: list[str] = []
        self._execution_context: dict[str, Any] = {}

    async def execute_with_lifecycle(self, lifecycle_funcs: list[str | Callable],
                                   workflow: Callable, *args, **kwargs):
        """
        Execute a workflow with specified lifecycle functions.
        
        Args:
            lifecycle_funcs: List of lifecycle functions to activate
            workflow: Workflow function to execute
            *args, **kwargs: Arguments to pass to workflow
        """
        async with with_lifecycle_functions(self.rails, lifecycle_funcs):
            # Set active functions for context
            self._active_functions = [
                f.__name__ if callable(f) else f for f in lifecycle_funcs
            ]

            try:
                if inspect.iscoroutinefunction(workflow):
                    return await workflow(*args, **kwargs)
                else:
                    return workflow(*args, **kwargs)
            finally:
                self._active_functions.clear()

    def get_active_functions(self) -> list[str]:
        """Get list of currently active lifecycle function names."""
        return self._active_functions.copy()

    def set_context(self, key: str, value: Any) -> None:
        """Set a value in the execution context."""
        self._execution_context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a value from the execution context."""
        return self._execution_context.get(key, default)

    def clear_context(self) -> None:
        """Clear the execution context."""
        self._execution_context.clear()


# Utility functions for common lifecycle patterns

@lifecycle_function(name="counter_tracker", priority=5)
async def counter_tracker_lifecycle(rails):
    """Basic counter tracking lifecycle function."""
    # Setup: Initialize tracking
    start_time = asyncio.get_event_loop().time()
    rails.store.set_sync('lifecycle_start_time', start_time)

    yield  # Main execution

    # Cleanup: Log final counts
    end_time = asyncio.get_event_loop().time()
    duration = end_time - start_time
    rails.store.set_sync('lifecycle_duration', duration)


@lifecycle_function(name="error_handler", priority=10)
async def error_handler_lifecycle(rails):
    """Error handling lifecycle function with automatic recovery."""
    try:
        yield  # Main execution
    except Exception as e:
        # Handle errors and potentially inject recovery messages
        await rails.store.increment('lifecycle_errors')
        rails.store.set_sync('last_error', str(e))

        # Could inject recovery message here
        # This is where tools could manipulate the rails instance
        recovery_message = {
            "role": "system",
            "content": f"Error occurred during execution: {e}. Attempting recovery..."
        }

        # Tools can access rails instance to add conditional recovery
        error_count = await rails.store.get_counter('lifecycle_errors', 0)
        if error_count >= 3:
            # Inject stop signal after too many errors
            rails.when(lambda s: s.get_counter_sync('lifecycle_errors') >= 3).inject({
                "role": "system",
                "content": "🛑 Multiple errors detected. Stopping execution for safety."
            })

        raise  # Re-raise the exception
