"""Injection strategies for Rails message manipulation."""

from typing import List, Dict, Any
from .types import Message, Injector


class AppendInjector:
    """Injector that appends message to the end of the message chain."""
    
    def __init__(self):
        """Initialize append injector."""
        pass
        
    def inject(self, messages: List[Message], new_message: Message) -> List[Message]:
        """Append message to end of chain.
        
        Args:
            messages: Current message chain
            new_message: Message to append
            
        Returns:
            Message chain with new message appended
        """
        return messages + [new_message]
        
    def __str__(self) -> str:
        return "AppendInjector"


class PrependInjector:
    """Injector that prepends message to the beginning of the message chain."""
    
    def __init__(self):
        """Initialize prepend injector."""
        pass
        
    def inject(self, messages: List[Message], new_message: Message) -> List[Message]:
        """Prepend message to beginning of chain.
        
        Args:
            messages: Current message chain
            new_message: Message to prepend
            
        Returns:
            Message chain with new message prepended
        """
        return [new_message] + messages
        
    def __str__(self) -> str:
        return "PrependInjector"


class InsertInjector:
    """Injector that inserts message at a specific position in the chain."""
    
    def __init__(self, position: int):
        """Initialize insert injector.
        
        Args:
            position: Position to insert at (0-based index)
        """
        self.position = position
        
    def inject(self, messages: List[Message], new_message: Message) -> List[Message]:
        """Insert message at specific position.
        
        Args:
            messages: Current message chain
            new_message: Message to insert
            
        Returns:
            Message chain with new message inserted
        """
        result = messages.copy()
        # Handle negative indices and out-of-bounds positions
        insert_pos = min(max(0, self.position), len(result))
        result.insert(insert_pos, new_message)
        return result
        
    def __str__(self) -> str:
        return f"InsertInjector(position={self.position})"


class ReplaceInjector:
    """Injector that replaces messages in the chain based on criteria."""
    
    def __init__(self, replace_last: bool = False, replace_all: bool = False, 
                 filter_func=None):
        """Initialize replace injector.
        
        Args:
            replace_last: Replace only the last message
            replace_all: Replace all messages with the new message
            filter_func: Function to filter which messages to replace
        """
        self.replace_last = replace_last
        self.replace_all = replace_all
        self.filter_func = filter_func
        
    def inject(self, messages: List[Message], new_message: Message) -> List[Message]:
        """Replace messages based on strategy.
        
        Args:
            messages: Current message chain
            new_message: Message to use as replacement
            
        Returns:
            Message chain with replacements applied
        """
        if self.replace_all:
            return [new_message]
            
        if self.replace_last and messages:
            return messages[:-1] + [new_message]
            
        if self.filter_func:
            result = []
            for msg in messages:
                if self.filter_func(msg):
                    result.append(new_message)
                else:
                    result.append(msg)
            return result
            
        # Default behavior: append if no specific replacement strategy
        return messages + [new_message]
        
    def __str__(self) -> str:
        strategy = []
        if self.replace_all:
            strategy.append("replace_all")
        if self.replace_last:
            strategy.append("replace_last")
        if self.filter_func:
            strategy.append("filter_func")
        return f"ReplaceInjector({', '.join(strategy)})"


class ConditionalInjector:
    """Injector that conditionally applies another injector based on message content."""
    
    def __init__(self, base_injector: Injector, condition_func):
        """Initialize conditional injector.
        
        Args:
            base_injector: Underlying injector to use
            condition_func: Function that takes (messages, new_message) -> bool
        """
        self.base_injector = base_injector
        self.condition_func = condition_func
        
    def inject(self, messages: List[Message], new_message: Message) -> List[Message]:
        """Conditionally apply injection.
        
        Args:
            messages: Current message chain
            new_message: Message to potentially inject
            
        Returns:
            Original or modified message chain
        """
        if self.condition_func(messages, new_message):
            return self.base_injector.inject(messages, new_message)
        return messages
        
    def __str__(self) -> str:
        return f"ConditionalInjector({self.base_injector})"


# Convenience factory functions
def append() -> AppendInjector:
    """Create an append injector."""
    return AppendInjector()


def prepend() -> PrependInjector:
    """Create a prepend injector."""
    return PrependInjector()


def insert_at(position: int) -> InsertInjector:
    """Create an insert injector for specific position."""
    return InsertInjector(position)


def replace_last() -> ReplaceInjector:
    """Create injector that replaces the last message."""
    return ReplaceInjector(replace_last=True)


def replace_all() -> ReplaceInjector:
    """Create injector that replaces all messages."""
    return ReplaceInjector(replace_all=True)


def replace_where(filter_func) -> ReplaceInjector:
    """Create injector that replaces messages matching filter."""
    return ReplaceInjector(filter_func=filter_func)