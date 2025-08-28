"""Core Rails implementation for message injection and workflow execution based on conditions."""

from typing import List, Union, Callable, Optional, Any
import asyncio
import inspect
from dataclasses import dataclass
from contextlib import asynccontextmanager

from .types import Message, Condition, Injector
from .store import Store
from .injectors import AppendInjector


@dataclass
class InjectionRule:
    """Represents a condition-injection rule pair."""
    condition: Condition
    message: Message
    injector: Injector
    name: Optional[str] = None


@dataclass  
class ExecutionRule:
    """Represents a condition-execution rule pair for workflows."""
    condition: Condition
    workflow: Callable
    args: tuple = ()
    kwargs: dict = None
    name: Optional[str] = None
    background: bool = False
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


# Global Rails registry for tool access
_current_rails: Optional['Rails'] = None


def current_rails() -> 'Rails':
    """
    Get the current Rails instance for tool access.
    
    This allows tools and workflows to easily access the Rails instance
    they're running within, enabling them to manipulate state, add rules, etc.
    
    Returns:
        Current Rails instance
        
    Raises:
        RuntimeError: If no Rails instance is currently active
        
    Usage:
        from rails import current_rails
        
        def my_tool():
            rails = current_rails()
            rails.store.increment('tool_calls')
            rails.when(condition).inject(message)
    """
    if _current_rails is None:
        raise RuntimeError("No Rails instance is currently active. Use Rails within a context manager or adapter.")
    return _current_rails


def set_current_rails(rails: Optional['Rails']) -> None:
    """Set the current Rails instance (internal use)."""
    global _current_rails
    _current_rails = rails


class Rails:
    """Main Rails class for conditional message injection.
    
    Rails allows you to define conditions and automatically inject
    specific messages when those conditions are met.
    
    Usage:
        rails = Rails()
        rails.when(lambda s: s.get_counter_sync('turns') >= 3).inject(message)
        modified_messages = await rails.check(messages)
    """
    
    def __init__(self) -> None:
        """Initialize Rails with empty rules and store."""
        self.store = Store()
        self._injection_rules: List[InjectionRule] = []
        self._execution_rules: List[ExecutionRule] = []
        self._current_condition: Optional[Condition] = None
        self._lifecycle_functions: List[Union[str, Callable]] = []
        self._lifecycle_manager = None
        
    def when(self, condition: Union[Condition, Callable[[Store], bool]]) -> 'Rails':
        """Add a condition to check for message injection.
        
        Args:
            condition: Condition callable or lambda that takes Store and returns bool
            
        Returns:
            Self for method chaining
            
        Examples:
            rails.when(lambda s: s.get_counter_sync('turns') >= 3)
            rails.when(counter_condition)
        """
        # Store the condition for the next inject() call
        self._current_condition = condition
        return self
        
    def inject(self, message: Message, strategy: Union[str, Injector] = 'append', 
               name: Optional[str] = None) -> 'Rails':
        """Define message to inject when the last condition is met.
        
        Args:
            message: Message to inject (framework-agnostic dict)
            strategy: Injection strategy ('append', 'prepend', 'replace_last') or Injector instance
            name: Optional name for this rule
            
        Returns:
            Self for method chaining
            
        Examples:
            rails.when(condition).inject({"role": "system", "content": "Help the user"})
            rails.when(condition).inject(message, strategy='prepend')
        """
        if self._current_condition is None:
            raise ValueError("inject() must be called after when(). Use when().inject() pattern.")
            
        # Convert string strategy to injector
        if isinstance(strategy, str):
            if strategy == 'append':
                from .injectors import append
                injector = append()
            elif strategy == 'prepend':
                from .injectors import prepend
                injector = prepend()
            elif strategy == 'replace_last':
                from .injectors import replace_last
                injector = replace_last()
            else:
                raise ValueError(f"Unknown strategy '{strategy}'. Use 'append', 'prepend', 'replace_last' or provide Injector instance.")
        else:
            injector = strategy
            
        # Create the rule
        rule = InjectionRule(
            condition=self._current_condition,
            message=message,
            injector=injector,
            name=name
        )
        self._injection_rules.append(rule)
        
        # Clear current condition for next rule
        self._current_condition = None
        
        return self
        
    def then(self, workflow: Callable, *args, background: bool = False, 
             name: Optional[str] = None, **kwargs) -> 'Rails':
        """Define workflow to execute when the last condition is met.
        
        Args:
            workflow: Function or workflow to execute
            *args: Positional arguments to pass to workflow
            background: Whether to execute in background (non-blocking)
            name: Optional name for this rule
            **kwargs: Keyword arguments to pass to workflow
            
        Returns:
            Self for method chaining
            
        Examples:
            rails.when(condition).then(my_workflow)
            rails.when(condition).then(lambda r: r.store.set('mode', 'debug'))
            rails.when(condition).then(start_background_task, background=True)
        """
        if self._current_condition is None:
            raise ValueError("then() must be called after when(). Use when().then() pattern.")
            
        # Create the execution rule
        rule = ExecutionRule(
            condition=self._current_condition,
            workflow=workflow,
            args=args,
            kwargs=kwargs,
            name=name,
            background=background
        )
        self._execution_rules.append(rule)
        
        # Clear current condition for next rule
        self._current_condition = None
        
        return self
        
    def with_lifecycle(self, *lifecycle_funcs: Union[str, Callable]) -> 'Rails':
        """Add lifecycle functions to be used with this Rails instance.
        
        Lifecycle functions provide composable setup/cleanup logic that can be
        mixed and matched based on the needs of your workflow.
        
        Args:
            *lifecycle_funcs: Lifecycle function names or callable functions
            
        Returns:
            Self for method chaining
            
        Examples:
            rails.with_lifecycle('database', 'queue').when(condition).inject(message)
            rails.with_lifecycle(my_lifecycle_func).when(condition).then(workflow)
            
            @lifecycle_function
            async def my_setup(rails):
                # setup code
                yield
                # cleanup code
        """
        self._lifecycle_functions.extend(lifecycle_funcs)
        return self
        
    async def check(self, messages: List[Message]) -> List[Message]:
        """Check conditions and inject messages or execute workflows if met.
        
        Args:
            messages: Current message chain to potentially modify
            
        Returns:
            Modified message chain with injections applied
        """
        result = messages.copy()
        
        # Handle injection rules
        for rule in self._injection_rules:
            try:
                if rule.condition(self.store):
                    result = rule.injector.inject(result, rule.message)
            except Exception as e:
                # Log error but continue with other rules
                # In production, you might want to use proper logging
                print(f"Error in injection condition {rule.name or 'unnamed'}: {e}")
                continue
        
        # Handle execution rules  
        for rule in self._execution_rules:
            try:
                if rule.condition(self.store):
                    await self._execute_workflow(rule)
            except Exception as e:
                # Log error but continue with other rules
                print(f"Error in execution condition {rule.name or 'unnamed'}: {e}")
                continue
                
        return result
        
    async def _execute_workflow(self, rule: ExecutionRule) -> None:
        """Execute a workflow from an execution rule."""
        try:
            if rule.background:
                # Execute in background (fire and forget)
                asyncio.create_task(self._run_workflow(rule))
            else:
                # Execute synchronously
                await self._run_workflow(rule)
        except Exception as e:
            print(f"Error executing workflow {rule.name or 'unnamed'}: {e}")
            
    async def _run_workflow(self, rule: ExecutionRule) -> Any:
        """Run a workflow with proper argument handling."""
        workflow = rule.workflow
        
        # Pass rails instance as first argument if workflow accepts it
        if inspect.signature(workflow).parameters:
            first_param = next(iter(inspect.signature(workflow).parameters.values()))
            if first_param.name in ['rails', 'r'] or first_param.annotation == 'Rails':
                args = (self,) + rule.args
            else:
                args = rule.args
        else:
            args = rule.args
        
        if inspect.iscoroutinefunction(workflow):
            return await workflow(*args, **rule.kwargs)
        else:
            return workflow(*args, **rule.kwargs)
        
    async def __aenter__(self) -> 'Rails':
        """Context manager entry for lifecycle management."""
        # Set this instance as the current Rails for global access
        set_current_rails(self)
        
        # Initialize lifecycle manager if we have lifecycle functions
        if self._lifecycle_functions:
            from .lifecycle import LifecycleManager
            self._lifecycle_manager = LifecycleManager(self)
        
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit for cleanup."""
        try:
            # Clear the global reference
            set_current_rails(None)
            
            # Cleanup lifecycle manager
            if self._lifecycle_manager:
                self._lifecycle_manager.clear_context()
                
            # Clear store
            await self.store.clear()
        finally:
            # Ensure global reference is cleared even if cleanup fails
            set_current_rails(None)
        
    def clear_rules(self) -> None:
        """Clear all injection and execution rules."""
        self._injection_rules.clear()
        self._execution_rules.clear()
        self._current_condition = None
        
    def get_injection_rules(self) -> List[InjectionRule]:
        """Get list of current injection rules."""
        return self._injection_rules.copy()
        
    def get_execution_rules(self) -> List[ExecutionRule]:
        """Get list of current execution rules."""
        return self._execution_rules.copy()
        
    def rule_count(self) -> int:
        """Get total number of active rules."""
        return len(self._injection_rules) + len(self._execution_rules)
        
    def injection_rule_count(self) -> int:
        """Get number of active injection rules."""
        return len(self._injection_rules)
        
    def execution_rule_count(self) -> int:
        """Get number of active execution rules."""  
        return len(self._execution_rules)
        
    # Convenience methods for common patterns
    def on_counter(self, counter_key: str, threshold: int, message: Message, 
                   comparison: str = ">=") -> 'Rails':
        """Convenience method for counter-based injection.
        
        Args:
            counter_key: Counter to check
            threshold: Threshold value
            message: Message to inject
            comparison: Comparison operator
            
        Returns:
            Self for chaining
        """
        from .conditions import CounterCondition
        condition = CounterCondition(counter_key, threshold, comparison)
        return self.when(condition).inject(message)
        
    def on_state(self, state_key: str, expected_value, message: Message) -> 'Rails':
        """Convenience method for state-based injection.
        
        Args:
            state_key: State key to check
            expected_value: Expected state value
            message: Message to inject
            
        Returns:
            Self for chaining
        """
        from .conditions import StateCondition
        condition = StateCondition(state_key, expected_value)
        return self.when(condition).inject(message)
        
    def __str__(self) -> str:
        total_rules = len(self._injection_rules) + len(self._execution_rules)
        return f"Rails({total_rules} rules: {len(self._injection_rules)} inject, {len(self._execution_rules)} execute)"
        
    def __repr__(self) -> str:
        return f"Rails(injection_rules={len(self._injection_rules)}, execution_rules={len(self._execution_rules)}, store_keys={len(self.store._state) + len(self.store._counters)}, lifecycle_funcs={len(self._lifecycle_functions)})"