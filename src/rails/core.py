"""Core Rails implementation for lifecycle orchestration of AI agents."""

import inspect
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from .store import Store
from .types import Condition, Message, RailState

# Context variable for Rails instance access from tools
rails_context: ContextVar[Optional['Rails']] = ContextVar('rails_context', default=None)


def current_rails() -> 'Rails':
    """Get the current Rails instance from context.
    
    This allows tools to access the Rails store for lifecycle orchestration.
    
    Returns:
        Current Rails instance
        
    Raises:
        RuntimeError: If no Rails instance is active
        
    Usage:
        from rails import current_rails
        
        @tool
        def my_tool():
            rails = current_rails()
            await rails.store.push_queue("tasks", "new task")
    """
    rails = rails_context.get()
    if rails is None:
        raise RuntimeError("No Rails instance is currently active. Ensure Rails is initialized.")
    return rails


class Rule(BaseModel):
    """A lifecycle orchestration rule."""

    condition: Condition
    action: Callable  # Can be injector or workflow
    name: str | None = None
    priority: int = 0
    enabled: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def evaluate(self, store: Store) -> bool:
        """Evaluate if this rule should trigger."""
        if not self.enabled:
            return False
        return await self.condition.evaluate(store)

    async def execute(self, context: Any) -> Any:
        """Execute the rule action."""
        if inspect.iscoroutinefunction(self.action):
            return await self.action(context)
        else:
            return self.action(context)


class Rails(BaseModel):
    """Lifecycle orchestration layer for AI agents.
    
    Rails provides a shared state store that both Rails conditions and agent tools
    can access, enabling sophisticated feedback loops and lifecycle management.
    
    Usage:
        async with Rails() as rails:
            # Add lifecycle rules
            rails.add_rule(
                condition=QueueLength("tasks") > 5,
                action=inject_message(system("Focus on high priority tasks"))
            )
            
            # Process messages through Rails
            messages = await rails.process(messages)
    """

    store: Store = Field(default_factory=Store)
    rules: list[Rule] = Field(default_factory=list)
    state: RailState = Field(default=RailState.INITIALIZED)
    middleware_stack: list[Callable] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_rule(self, condition: Condition, action: Callable,
                 name: str | None = None, priority: int = 0) -> None:
        """Add a lifecycle orchestration rule.
        
        Args:
            condition: Condition to evaluate
            action: Action to take when condition is met
            name: Optional rule name for debugging
            priority: Rule priority (higher = evaluated first)
        """
        rule = Rule(
            condition=condition,
            action=action,
            name=name,
            priority=priority
        )
        self.rules.append(rule)
        # Sort by priority
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    async def process(self, messages: list[Message]) -> list[Message]:
        """Process messages through Rails lifecycle orchestration.
        
        Args:
            messages: Current message chain
            
        Returns:
            Modified message chain with any injections
        """
        self.state = RailState.EVALUATING
        result = messages.copy()

        # Set context if not already set
        current = rails_context.get()
        if current is None:
            token = rails_context.set(self)
        else:
            token = None

        try:
            # Evaluate all rules
            for rule in self.rules:
                if await rule.evaluate(self.store):
                    logger.debug(f"Rule '{rule.name or 'unnamed'}' triggered")
                    self.state = RailState.INJECTING

                    # Execute the action - it may modify messages or perform side effects
                    action_result = await rule.execute(result)
                    if action_result is not None:
                        # Action returned modified messages
                        result = action_result

            self.state = RailState.COMPLETED
            return result

        except Exception as e:
            self.state = RailState.ERROR
            logger.error(f"Error in Rails processing: {e}")
            raise
        finally:
            # Reset context if we set it
            if token is not None:
                rails_context.reset(token)


    async def __aenter__(self) -> 'Rails':
        """Context manager entry - set up Rails context."""
        # Set this instance in context for tool access
        token = rails_context.set(self)
        self._context_token = token

        # Restore store state if configured
        await self.store.restore()

        logger.info("Rails lifecycle orchestration initialized")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - clean up Rails context."""
        try:
            # Persist store state if configured
            await self.store.persist()

            # Reset context
            if hasattr(self, '_context_token'):
                rails_context.reset(self._context_token)

            logger.info("Rails lifecycle orchestration completed")
        except Exception as e:
            logger.error(f"Error during Rails cleanup: {e}")

    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware to the processing stack.
        
        Args:
            middleware: Async callable that processes messages
        """
        self.middleware_stack.append(middleware)

    async def process_with_middleware(self, messages: list[Message]) -> list[Message]:
        """Process messages through middleware stack then Rails rules.
        
        Args:
            messages: Input messages
            
        Returns:
            Processed messages
        """
        result = messages

        # Process through middleware stack
        for middleware in self.middleware_stack:
            if inspect.iscoroutinefunction(middleware):
                result = await middleware(result, self.store)
            else:
                result = middleware(result, self.store)

        # Then process through Rails rules
        return await self.process(result)

    def clear_rules(self) -> None:
        """Clear all rules."""
        self.rules.clear()

    def enable_rule(self, name: str) -> None:
        """Enable a rule by name."""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = True
                break

    def disable_rule(self, name: str) -> None:
        """Disable a rule by name."""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = False
                break

    def get_active_rules(self) -> list[Rule]:
        """Get all enabled rules."""
        return [r for r in self.rules if r.enabled]

    async def emit_metrics(self) -> dict[str, Any]:
        """Get Rails metrics for observability."""
        snapshot = await self.store.get_snapshot()
        return {
            "state": self.state.value,
            "total_rules": len(self.rules),
            "active_rules": len(self.get_active_rules()),
            "store_snapshot": snapshot,
            "middleware_count": len(self.middleware_stack)
        }

    def __str__(self) -> str:
        return f"Rails({len(self.rules)} rules, state={self.state.value})"

    def __repr__(self) -> str:
        return f"Rails(rules={len(self.rules)}, middleware={len(self.middleware_stack)}, state={self.state})"
