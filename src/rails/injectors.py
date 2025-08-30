"""Message injection strategies for Rails lifecycle orchestration."""

from abc import ABC, abstractmethod
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from .store import Store
from .types import Condition, Message, Role


class InjectorBase(BaseModel, ABC):
    """Base class for all message injectors."""

    name: str | None = None
    description: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def inject(self, messages: list[Message]) -> list[Message]:
        """Inject message(s) into the conversation.
        
        Args:
            messages: Current message chain
            
        Returns:
            Modified message chain
        """
        ...

    def describe(self) -> str:
        """Get human-readable description."""
        return self.description or self.__class__.__name__


class AppendInjector(InjectorBase):
    """Appends message to the end of the conversation."""

    message: Message

    async def inject(self, messages: list[Message]) -> list[Message]:
        """Append a message to the end."""
        result = messages.copy()
        result.append(self.message)
        return result


class PrependInjector(InjectorBase):
    """Prepends message to the beginning of the conversation."""

    message: Message

    async def inject(self, messages: list[Message]) -> list[Message]:
        """Prepend a message to the beginning."""
        result = messages.copy()
        result.insert(0, self.message)
        return result


class InsertInjector(InjectorBase):
    """Inserts message at a specific index."""

    message: Message
    index: int = 0

    async def inject(self, messages: list[Message]) -> list[Message]:
        """Insert a message at a specific index."""
        result = messages.copy()
        result.insert(self.index, self.message)
        return result


class ReplaceInjector(InjectorBase):
    """Replaces all messages with new messages."""

    messages: list[Message]

    async def inject(self, messages: list[Message]) -> list[Message]:
        """Replace all messages with new messages."""
        return self.messages.copy()


class ConditionalInjector(InjectorBase):
    """Conditionally inject based on Store state."""

    condition: Condition
    injector: InjectorBase

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def inject(self, messages: list[Message]) -> list[Message]:
        """Conditionally inject based on Store state."""
        from .core import current_rails
        try:
            rails = current_rails()
            store = rails.store
        except RuntimeError:
            store = Store()  # Fallback if no Rails context

        if await self.condition.evaluate(store):
            return await self.injector.inject(messages)
        return messages


class SystemInjector(InjectorBase):
    """Inject system message using template."""

    message: Message

    async def inject(self, messages: list[Message]) -> list[Message]:
        """Inject system message using template."""
        result = messages.copy()
        result.append(self.message)
        return result


class TemplateInjector(InjectorBase):
    """Injects messages using templates with store values."""

    template: str
    role: Role = Role.SYSTEM

    async def inject(self, messages: list[Message]) -> list[Message]:
        """Inject templated message."""
        # Get context from Rails store
        from .core import current_rails
        try:
            rails = current_rails()
            context_vars = await rails.store.get_snapshot()
            # Merge state for template
            template_context = {**context_vars.get('state', {}), **context_vars.get('counters', {})}
        except RuntimeError:
            template_context = {}

        # Use format_map which allows missing keys
        try:
            content = self.template.format_map(template_context)
        except KeyError:
            # Fallback: use the template as-is if keys are missing
            content = self.template

        msg = Message(role=self.role, content=content, injected_by_rails=True)

        result = messages.copy()
        result.append(msg)
        return result


# Factory functions for common patterns
def append_message(message: Message) -> Callable:
    """Factory for appending messages."""
    injector = AppendInjector(message=message)

    async def action(messages: list[Message]) -> list[Message]:
        return await injector.inject(messages)

    return action


def prepend_message(message: Message) -> Callable:
    """Factory for prepending messages."""
    injector = PrependInjector(message=message)

    async def action(messages: list[Message]) -> list[Message]:
        return await injector.inject(messages)

    return action


def system(content: str, position: str = "append") -> Callable:
    """Factory for system messages."""
    async def action(messages: list[Message]) -> list[Message]:
        msg = Message(role=Role.SYSTEM, content=content, injected_by_rails=True)
        result = messages.copy()
        if position == "prepend":
            result.insert(0, msg)
        else:
            result.append(msg)
        return result

    return action


def template(tpl: str, role: Role = Role.SYSTEM) -> Callable:
    """Factory for templated messages.
    
    Usage:
        rails.add_rule(
            condition=state("user_name").exists,
            action=template("Hello {user_name}!")
        )
    """
    injector = TemplateInjector(template=tpl, role=role)

    async def action(messages: list[Message]) -> list[Message]:
        return await injector.inject(messages)

    return action
