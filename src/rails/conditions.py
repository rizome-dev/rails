"""Condition system for Rails message injection."""

from typing import Any, Callable
from .types import Condition
from .store import Store


class LambdaCondition:
    """Wrapper for lambda/function conditions."""
    
    def __init__(self, func: Callable[[Store], bool], name: str = "lambda_condition"):
        """Initialize lambda condition.
        
        Args:
            func: Function that takes Store and returns boolean
            name: Optional name for the condition
        """
        self.func = func
        self.name = name
        
    def __call__(self, store: Store) -> bool:
        """Execute the condition function."""
        return self.func(store)
        
    def __str__(self) -> str:
        return f"LambdaCondition({self.name})"


class CounterCondition:
    """Condition that checks counter values."""
    
    def __init__(self, counter_key: str, threshold: int, comparison: str = ">="):
        """Initialize counter condition.
        
        Args:
            counter_key: Counter key to check
            threshold: Threshold value to compare against
            comparison: Comparison operator (>=, <=, ==, >, <, !=)
        """
        self.counter_key = counter_key
        self.threshold = threshold
        self.comparison = comparison
        
    def __call__(self, store: Store) -> bool:
        """Check if counter meets threshold condition."""
        current_value = store.get_counter_sync(self.counter_key)
        
        if self.comparison == ">=":
            return current_value >= self.threshold
        elif self.comparison == "<=":
            return current_value <= self.threshold
        elif self.comparison == "==":
            return current_value == self.threshold
        elif self.comparison == ">":
            return current_value > self.threshold
        elif self.comparison == "<":
            return current_value < self.threshold
        elif self.comparison == "!=":
            return current_value != self.threshold
        else:
            raise ValueError(f"Unsupported comparison operator: {self.comparison}")
            
    def __str__(self) -> str:
        return f"CounterCondition({self.counter_key} {self.comparison} {self.threshold})"


class StateCondition:
    """Condition that checks state values."""
    
    def __init__(self, state_key: str, expected_value: Any, comparison: str = "=="):
        """Initialize state condition.
        
        Args:
            state_key: State key to check
            expected_value: Expected value to compare against
            comparison: Comparison operator (==, !=, in, not_in)
        """
        self.state_key = state_key
        self.expected_value = expected_value
        self.comparison = comparison
        
    def __call__(self, store: Store) -> bool:
        """Check if state meets condition."""
        current_value = store.get_sync(self.state_key)
        
        if self.comparison == "==":
            return current_value == self.expected_value
        elif self.comparison == "!=":
            return current_value != self.expected_value
        elif self.comparison == "in":
            return current_value in self.expected_value
        elif self.comparison == "not_in":
            return current_value not in self.expected_value
        else:
            raise ValueError(f"Unsupported comparison operator: {self.comparison}")
            
    def __str__(self) -> str:
        return f"StateCondition({self.state_key} {self.comparison} {self.expected_value})"


class AndCondition:
    """Condition that requires all sub-conditions to be true."""
    
    def __init__(self, *conditions: Condition):
        """Initialize AND condition.
        
        Args:
            *conditions: Conditions that must all be true
        """
        self.conditions = conditions
        
    def __call__(self, store: Store) -> bool:
        """Check if all conditions are true."""
        return all(condition(store) for condition in self.conditions)
        
    def __str__(self) -> str:
        return f"AndCondition({len(self.conditions)} conditions)"


class OrCondition:
    """Condition that requires at least one sub-condition to be true."""
    
    def __init__(self, *conditions: Condition):
        """Initialize OR condition.
        
        Args:
            *conditions: Conditions where at least one must be true
        """
        self.conditions = conditions
        
    def __call__(self, store: Store) -> bool:
        """Check if any condition is true."""
        return any(condition(store) for condition in self.conditions)
        
    def __str__(self) -> str:
        return f"OrCondition({len(self.conditions)} conditions)"


class NotCondition:
    """Condition that negates another condition."""
    
    def __init__(self, condition: Condition):
        """Initialize NOT condition.
        
        Args:
            condition: Condition to negate
        """
        self.condition = condition
        
    def __call__(self, store: Store) -> bool:
        """Check negated condition."""
        return not self.condition(store)
        
    def __str__(self) -> str:
        return f"NotCondition({self.condition})"


# Convenience functions for creating common conditions
def counter_at_least(counter_key: str, threshold: int) -> CounterCondition:
    """Create condition for counter >= threshold."""
    return CounterCondition(counter_key, threshold, ">=")


def counter_equals(counter_key: str, value: int) -> CounterCondition:
    """Create condition for counter == value."""
    return CounterCondition(counter_key, value, "==")


def state_equals(state_key: str, value: Any) -> StateCondition:
    """Create condition for state == value."""
    return StateCondition(state_key, value, "==")


def state_in(state_key: str, values: Any) -> StateCondition:
    """Create condition for state value in collection."""
    return StateCondition(state_key, values, "in")