"""Type definitions for Rails lifecycle orchestration library."""

from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Role(str, Enum):
    """Message role types."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """Immutable message model for agent conversations."""

    role: Role
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    injected_by_rails: bool = Field(default=False)

    model_config = ConfigDict(frozen=True)

    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v


class StateEvent(BaseModel):
    """Event emitted when store state changes."""

    event_type: Literal["counter_increment", "state_set", "queue_push", "queue_pop"]
    key: str
    value: Any = None
    previous_value: Any = None
    timestamp: datetime = Field(default_factory=datetime.now)
    triggered_by: str | None = None  # Tool or condition that triggered


class RailState(str, Enum):
    """Rails lifecycle states."""
    INITIALIZED = "initialized"
    EVALUATING = "evaluating"
    INJECTING = "injecting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@runtime_checkable
class Condition(Protocol):
    """Protocol for condition evaluation."""

    @abstractmethod
    async def evaluate(self, store: 'Store') -> bool:
        """Evaluate condition against store state.
        
        Args:
            store: Current Rails store instance
            
        Returns:
            True if condition is met, False otherwise
        """
        ...

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of the condition."""
        ...


@runtime_checkable
class Injector(Protocol):
    """Protocol for message injection strategies."""

    @abstractmethod
    async def inject(self, messages: list[Message], new_message: Message) -> list[Message]:
        """Inject a message into the conversation.
        
        Args:
            messages: Current message chain
            new_message: Message to inject
            
        Returns:
            Modified message chain
        """
        ...

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of the injection strategy."""
        ...


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Protocol for integrating Rails with agent frameworks."""

    @abstractmethod
    async def register_store_access(self, store: 'Store') -> None:
        """Make Rails store accessible to agent tools.
        
        Args:
            store: Rails store instance to register
        """
        ...

    @abstractmethod
    async def wrap(self, agent: Any) -> Any:
        """Wrap an agent with Rails lifecycle management.
        
        Args:
            agent: Agent instance to wrap
            
        Returns:
            Wrapped agent with Rails integration
        """
        ...

    @abstractmethod
    async def process_messages(self, messages: list[Message]) -> list[Message]:
        """Process messages through the framework.
        
        Args:
            messages: Messages to process
            
        Returns:
            Processed messages
        """
        ...


class QueueConfig(BaseModel):
    """Configuration for a Rails queue."""

    max_size: int | None = None
    auto_dedup: bool = False
    fifo: bool = True  # True for FIFO, False for LIFO


class StoreConfig(BaseModel):
    """Configuration for the Rails store."""

    persist_on_exit: bool = False
    persistence_path: str | None = None
    emit_events: bool = True
    max_event_history: int = 1000
    default_queues: dict[str, QueueConfig] = Field(default_factory=dict)
