"""Condition system for Rails lifecycle orchestration."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, field_validator

from .store import Store


class ComparisonOperator(str, Enum):
    """Supported comparison operators."""
    EQ = "=="
    NE = "!="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


class ConditionBase(BaseModel, ABC):
    """Base class for all conditions."""

    name: str | None = None
    description: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def evaluate(self, store: Store) -> bool:
        """Evaluate the condition against store state."""
        ...

    def describe(self) -> str:
        """Get human-readable description."""
        return self.description or str(self)


class CounterCondition(ConditionBase):
    """Condition that checks counter values."""

    counter_key: str
    threshold: int
    operator: ComparisonOperator = ComparisonOperator.GE

    @field_validator('counter_key')
    @classmethod
    def validate_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Counter key cannot be empty")
        return v

    async def evaluate(self, store: Store) -> bool:
        """Check if counter meets threshold."""
        current = await store.get_counter(self.counter_key)

        if self.operator == ComparisonOperator.EQ:
            return current == self.threshold
        elif self.operator == ComparisonOperator.NE:
            return current != self.threshold
        elif self.operator == ComparisonOperator.GT:
            return current > self.threshold
        elif self.operator == ComparisonOperator.GE:
            return current >= self.threshold
        elif self.operator == ComparisonOperator.LT:
            return current < self.threshold
        elif self.operator == ComparisonOperator.LE:
            return current <= self.threshold
        else:
            logger.warning(f"Unsupported operator {self.operator} for counter condition")
            return False

    def __str__(self) -> str:
        return f"Counter({self.counter_key} {self.operator.value} {self.threshold})"


class StateCondition(ConditionBase):
    """Condition that checks state values."""

    state_key: str
    expected_value: Any
    operator: ComparisonOperator = ComparisonOperator.EQ

    async def evaluate(self, store: Store) -> bool:
        """Check if state meets condition."""
        current = await store.get(self.state_key)

        if self.operator == ComparisonOperator.EQ:
            return current == self.expected_value
        elif self.operator == ComparisonOperator.NE:
            return current != self.expected_value
        elif self.operator == ComparisonOperator.IN:
            return current in self.expected_value
        elif self.operator == ComparisonOperator.NOT_IN:
            return current not in self.expected_value
        elif self.operator == ComparisonOperator.CONTAINS:
            return self.expected_value in current if current else False
        elif self.operator == ComparisonOperator.STARTS_WITH:
            return str(current).startswith(str(self.expected_value)) if current else False
        elif self.operator == ComparisonOperator.ENDS_WITH:
            return str(current).endswith(str(self.expected_value)) if current else False
        else:
            logger.warning(f"Unsupported operator {self.operator} for state condition")
            return False

    def __str__(self) -> str:
        return f"State({self.state_key} {self.operator.value} {self.expected_value})"


class StateExistsCondition(ConditionBase):
    """Condition that checks if a state key exists."""

    state_key: str

    async def evaluate(self, store: Store) -> bool:
        """Check if state key exists."""
        value = await store.get(self.state_key)
        return value is not None

    def __str__(self) -> str:
        return f"StateExists({self.state_key})"


class QueueCondition(ConditionBase):
    """Condition that checks queue state."""

    queue_name: str
    check_type: Literal["length", "empty", "contains", "head"]
    threshold: int | None = None
    expected_value: Any | None = None
    operator: ComparisonOperator = ComparisonOperator.GE

    @field_validator('queue_name')
    @classmethod
    def validate_queue_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Queue name cannot be empty")
        return v

    async def evaluate(self, store: Store) -> bool:
        """Check queue condition."""
        if self.check_type == "length":
            length = await store.queue_length(self.queue_name)
            if self.threshold is None:
                return False

            if self.operator == ComparisonOperator.EQ:
                return length == self.threshold
            elif self.operator == ComparisonOperator.GT:
                return length > self.threshold
            elif self.operator == ComparisonOperator.GE:
                return length >= self.threshold
            elif self.operator == ComparisonOperator.LT:
                return length < self.threshold
            elif self.operator == ComparisonOperator.LE:
                return length <= self.threshold
            else:
                return False

        elif self.check_type == "empty":
            length = await store.queue_length(self.queue_name)
            return length == 0

        elif self.check_type == "contains":
            queue = await store.get_queue(self.queue_name)
            return self.expected_value in queue

        elif self.check_type == "head":
            queue = await store.get_queue(self.queue_name)
            if not queue:
                return False
            return queue[0] == self.expected_value

        return False

    def __str__(self) -> str:
        if self.check_type == "length":
            return f"Queue({self.queue_name}.length {self.operator.value} {self.threshold})"
        elif self.check_type == "empty":
            return f"Queue({self.queue_name}.empty)"
        elif self.check_type == "contains":
            return f"Queue({self.queue_name} contains {self.expected_value})"
        elif self.check_type == "head":
            return f"Queue({self.queue_name}.head == {self.expected_value})"
        return f"Queue({self.queue_name} {self.check_type})"


class LambdaCondition(ConditionBase):
    """Condition that uses a custom function."""

    func: Callable[[Store], bool]
    func_name: str = "lambda"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def evaluate(self, store: Store) -> bool:
        """Evaluate custom function."""
        import inspect
        if inspect.iscoroutinefunction(self.func):
            return await self.func(store)
        else:
            return self.func(store)

    def __str__(self) -> str:
        return f"Lambda({self.func_name})"


class AndCondition(ConditionBase):
    """Composite condition requiring all sub-conditions to be true."""

    conditions: list[ConditionBase]

    def __init__(self, *conditions, **kwargs):
        """Allow passing conditions as positional arguments."""
        if conditions and not kwargs.get('conditions'):
            super().__init__(conditions=list(conditions))
        else:
            super().__init__(**kwargs)

    async def evaluate(self, store: Store) -> bool:
        """Check if all conditions are true."""
        for condition in self.conditions:
            if not await condition.evaluate(store):
                return False
        return True

    def __str__(self) -> str:
        return f"And({len(self.conditions)} conditions)"


class OrCondition(ConditionBase):
    """Composite condition requiring at least one sub-condition to be true."""

    conditions: list[ConditionBase]

    def __init__(self, *conditions, **kwargs):
        """Allow passing conditions as positional arguments."""
        if conditions and not kwargs.get('conditions'):
            super().__init__(conditions=list(conditions))
        else:
            super().__init__(**kwargs)

    async def evaluate(self, store: Store) -> bool:
        """Check if any condition is true."""
        for condition in self.conditions:
            if await condition.evaluate(store):
                return True
        return False

    def __str__(self) -> str:
        return f"Or({len(self.conditions)} conditions)"


class NotCondition(ConditionBase):
    """Condition that negates another condition."""

    condition: ConditionBase

    def __init__(self, condition: ConditionBase = None, **kwargs):
        """Allow passing condition as positional argument."""
        if condition is not None and 'condition' not in kwargs:
            super().__init__(condition=condition)
        else:
            super().__init__(**kwargs)

    async def evaluate(self, store: Store) -> bool:
        """Check negated condition."""
        return not await self.condition.evaluate(store)

    def __str__(self) -> str:
        return f"Not({self.condition})"


class AlwaysCondition(ConditionBase):
    """Condition that always evaluates to true."""

    async def evaluate(self, store: Store) -> bool:
        """Always returns true."""
        return True

    def __str__(self) -> str:
        return "Always()"


class NeverCondition(ConditionBase):
    """Condition that never evaluates to true."""

    async def evaluate(self, store: Store) -> bool:
        """Always returns false."""
        return False

    def __str__(self) -> str:
        return "Never()"


# Factory functions for common conditions
def counter(key: str) -> 'CounterBuilder':
    """Start building a counter condition."""
    return CounterBuilder(key)


def state(key: str) -> 'StateBuilder':
    """Start building a state condition."""
    return StateBuilder(key)


def queue(name: str) -> 'QueueBuilder':
    """Start building a queue condition."""
    return QueueBuilder(name)


class CounterBuilder:
    """Fluent builder for counter conditions."""

    def __init__(self, key: str):
        self.key = key

    def __eq__(self, value: int) -> CounterCondition:
        return CounterCondition(counter_key=self.key, threshold=value, operator=ComparisonOperator.EQ)

    def __ne__(self, value: int) -> CounterCondition:
        return CounterCondition(counter_key=self.key, threshold=value, operator=ComparisonOperator.NE)

    def __gt__(self, value: int) -> CounterCondition:
        return CounterCondition(counter_key=self.key, threshold=value, operator=ComparisonOperator.GT)

    def __ge__(self, value: int) -> CounterCondition:
        return CounterCondition(counter_key=self.key, threshold=value, operator=ComparisonOperator.GE)

    def __lt__(self, value: int) -> CounterCondition:
        return CounterCondition(counter_key=self.key, threshold=value, operator=ComparisonOperator.LT)

    def __le__(self, value: int) -> CounterCondition:
        return CounterCondition(counter_key=self.key, threshold=value, operator=ComparisonOperator.LE)


class StateBuilder:
    """Fluent builder for state conditions."""

    def __init__(self, key: str):
        self.key = key

    def __eq__(self, value: Any) -> StateCondition:
        return StateCondition(state_key=self.key, expected_value=value, operator=ComparisonOperator.EQ)

    def __ne__(self, value: Any) -> StateCondition:
        return StateCondition(state_key=self.key, expected_value=value, operator=ComparisonOperator.NE)

    def in_(self, values: Any) -> StateCondition:
        return StateCondition(state_key=self.key, expected_value=values, operator=ComparisonOperator.IN)

    def not_in(self, values: Any) -> StateCondition:
        return StateCondition(state_key=self.key, expected_value=values, operator=ComparisonOperator.NOT_IN)

    def contains(self, value: Any) -> StateCondition:
        return StateCondition(state_key=self.key, expected_value=value, operator=ComparisonOperator.CONTAINS)

    @property
    def exists(self) -> 'StateExistsCondition':
        """Check if state key exists."""
        return StateExistsCondition(state_key=self.key)


class QueueBuilder:
    """Fluent builder for queue conditions."""

    def __init__(self, name: str):
        self.name = name

    @property
    def empty(self) -> QueueCondition:
        return QueueCondition(queue_name=self.name, check_type="empty")

    @property
    def is_empty(self) -> QueueCondition:
        """Alias for empty."""
        return QueueCondition(queue_name=self.name, check_type="empty")

    @property
    def length(self) -> 'QueueLengthBuilder':
        return QueueLengthBuilder(self.name)

    def contains(self, value: Any) -> QueueCondition:
        return QueueCondition(queue_name=self.name, check_type="contains", expected_value=value)

    def head_equals(self, value: Any) -> QueueCondition:
        return QueueCondition(queue_name=self.name, check_type="head", expected_value=value)


class QueueLengthBuilder:
    """Builder for queue length conditions."""

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, value: int) -> QueueCondition:
        return QueueCondition(queue_name=self.name, check_type="length", threshold=value, operator=ComparisonOperator.EQ)

    def __gt__(self, value: int) -> QueueCondition:
        return QueueCondition(queue_name=self.name, check_type="length", threshold=value, operator=ComparisonOperator.GT)

    def __ge__(self, value: int) -> QueueCondition:
        return QueueCondition(queue_name=self.name, check_type="length", threshold=value, operator=ComparisonOperator.GE)

    def __lt__(self, value: int) -> QueueCondition:
        return QueueCondition(queue_name=self.name, check_type="length", threshold=value, operator=ComparisonOperator.LT)

    def __le__(self, value: int) -> QueueCondition:
        return QueueCondition(queue_name=self.name, check_type="length", threshold=value, operator=ComparisonOperator.LE)
