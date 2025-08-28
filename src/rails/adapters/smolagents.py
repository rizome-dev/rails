"""SmolaAgents adapter for Rails integration.

This module provides seamless integration between Rails and SmolaAgents,
allowing Rails conditional message injection to work with SmolaAgents agents and tools.
"""

from typing import List, Dict, Any, Optional, Union
import asyncio

try:
    from smolagents import CodeAgent, ToolCallingAgent
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    # Graceful degradation when SmolaAgents is not installed
    CodeAgent = Any
    ToolCallingAgent = Any
    SMOLAGENTS_AVAILABLE = False

from .base import BaseRailsAdapter
from ..core import Rails
from ..types import Message


class SmolaAgentsAdapter(BaseRailsAdapter):
    """Rails adapter for SmolaAgents integration.
    
    This adapter allows you to wrap SmolaAgents agents with Rails conditional
    message injection capabilities, enabling sophisticated conversation flow control.
    
    Usage:
        from smolagents import CodeAgent
        from rails.adapters import SmolaAgentsAdapter
        
        # Set up Rails rules
        rails = Rails()
        rails.when(lambda s: s.get_counter_sync('tool_calls') >= 3).inject({
            "role": "system",
            "content": "You've used several tools. Consider if you have enough information to answer."
        })
        
        # Create SmolaAgents agent
        agent = CodeAgent(tools=[], model="gpt-4")
        
        # Wrap with Rails
        adapter = SmolaAgentsAdapter(rails, agent)
        
        # Use with Rails injection
        result = await adapter.run("Analyze this data and create a visualization")
    """
    
    def __init__(self, rails: Optional[Rails] = None, agent: Optional[Any] = None):
        """Initialize the SmolaAgents adapter.
        
        Args:
            rails: Rails instance for message injection
            agent: SmolaAgents agent instance
        """
        super().__init__(rails)
        self.agent = agent
        
        if not SMOLAGENTS_AVAILABLE:
            raise ImportError(
                "SmolaAgents is not installed. Install it with: pip install smolagents"
            )
    
    async def process_messages(self, messages: List[Message], 
                             agent: Optional[Any] = None,
                             task: Optional[str] = None,
                             **kwargs) -> Any:
        """Process messages through SmolaAgents agent.
        
        Args:
            messages: Rails-processed messages
            agent: Optional agent to use (overrides instance agent)
            task: Task string for single-task execution
            **kwargs: Additional arguments for the agent
            
        Returns:
            SmolaAgents agent result
        """
        target_agent = agent or self.agent
        
        if target_agent is None:
            raise ValueError("No agent provided. Pass one to __init__ or process_messages")
        
        # Handle single task execution (most common SmolaAgents pattern)
        if task:
            # Inject Rails messages as system context
            system_messages = [msg for msg in messages if msg.get("role") == "system"]
            if system_messages:
                # Combine system messages into agent context
                context = "\n".join([msg["content"] for msg in system_messages])
                enhanced_task = f"Context: {context}\n\nTask: {task}"
            else:
                enhanced_task = task
            
            # Run the agent
            result = target_agent.run(enhanced_task, **kwargs)
            return result
        
        # Handle conversation-style interaction
        else:
            # Convert Rails messages to SmolaAgents format
            conversation = self._build_conversation(messages)
            
            # For conversation, we typically run the last user message
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if user_messages:
                last_user_message = user_messages[-1]["content"]
                
                # Add system context if present
                system_messages = [msg for msg in messages if msg.get("role") == "system"]
                if system_messages:
                    context = "\n".join([msg["content"] for msg in system_messages])
                    enhanced_message = f"Context: {context}\n\nUser: {last_user_message}"
                else:
                    enhanced_message = last_user_message
                
                result = target_agent.run(enhanced_message, **kwargs)
                return result
            else:
                raise ValueError("No user message found in conversation")
    
    def _build_conversation(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Build SmolaAgents conversation format from Rails messages.
        
        Args:
            messages: Rails messages
            
        Returns:
            SmolaAgents conversation format
        """
        conversation = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Map Rails roles to SmolaAgents format
            if role in ["user", "human"]:
                conversation.append({"role": "user", "content": content})
            elif role in ["assistant", "ai"]:
                conversation.append({"role": "assistant", "content": content})
            elif role == "system":
                # System messages are handled separately in SmolaAgents
                conversation.append({"role": "system", "content": content})
        
        return conversation
    
    async def update_rails_state(self, original_messages: List[Message], 
                               modified_messages: List[Message], result: Any) -> None:
        """Update Rails state after SmolaAgents processing.
        
        Args:
            original_messages: Original input messages
            modified_messages: Messages after Rails injection
            result: SmolaAgents processing result
        """
        await super().update_rails_state(original_messages, modified_messages, result)
        
        # Track SmolaAgents-specific metrics
        # Check if tools were used (basic heuristic)
        if hasattr(result, 'tool_calls') or "```" in str(result):
            await self.rails.store.increment("tool_calls")
        
        # Track if code was generated
        if "```python" in str(result) or "```code" in str(result):
            await self.rails.store.increment("code_generations")


class CodeAgentAdapter(SmolaAgentsAdapter):
    """Specialized Rails adapter for SmolaAgents CodeAgent.
    
    This adapter provides additional functionality specific to CodeAgent,
    including tracking code execution and managing coding context.
    
    Usage:
        from smolagents import CodeAgent
        from rails.adapters import CodeAgentAdapter
        
        # Set up Rails for code-specific rules
        rails = Rails()
        rails.when(lambda s: s.get_counter_sync('code_generations') >= 2).inject({
            "role": "system",
            "content": "You've generated code multiple times. Consider explaining your approach."
        })
        
        agent = CodeAgent(tools=[], model="gpt-4")
        adapter = CodeAgentAdapter(rails, agent)
        
        result = await adapter.run("Create a function to calculate fibonacci numbers")
    """
    
    async def update_rails_state(self, original_messages: List[Message], 
                               modified_messages: List[Message], result: Any) -> None:
        """Update Rails state with CodeAgent-specific tracking.
        
        Args:
            original_messages: Original input messages
            modified_messages: Messages after Rails injection
            result: CodeAgent processing result
        """
        await super().update_rails_state(original_messages, modified_messages, result)
        
        # Track code-specific patterns
        result_str = str(result)
        
        # Track different types of code generation
        if "def " in result_str or "class " in result_str:
            await self.rails.store.increment("python_functions")
        
        if "import " in result_str:
            await self.rails.store.increment("imports_used")
        
        if "Error" in result_str or "Exception" in result_str:
            await self.rails.store.increment("errors_encountered")


def create_smolagents_adapter(agent: Any, 
                            rails: Optional[Rails] = None) -> SmolaAgentsAdapter:
    """Factory function to create a SmolaAgents Rails adapter.
    
    Args:
        agent: SmolaAgents agent to wrap
        rails: Optional Rails instance
        
    Returns:
        Configured SmolaAgentsAdapter
        
    Example:
        from smolagents import CodeAgent
        from rails import Rails
        from rails.adapters import create_smolagents_adapter
        
        # Set up Rails
        rails = Rails()
        rails.when(lambda s: s.get_counter_sync('turns') >= 3).inject({
            "role": "system",
            "content": "Consider if you need to break down the task into smaller steps."
        })
        
        # Create adapter
        agent = CodeAgent(tools=[], model="gpt-4")
        adapter = create_smolagents_adapter(agent, rails)
        
        # Use it
        result = await adapter.run(task="Analyze sales data and create visualizations")
    """
    # Return specialized adapter for CodeAgent
    if hasattr(agent, 'tools') and 'code' in str(type(agent).__name__).lower():
        return CodeAgentAdapter(rails, agent)
    
    return SmolaAgentsAdapter(rails, agent)


# Decorator for wrapping SmolaAgents agents with Rails
def with_rails(rails: Optional[Rails] = None):
    """Decorator to wrap SmolaAgents agent creation with Rails.
    
    Args:
        rails: Rails instance to use
        
    Returns:
        Decorator function
        
    Example:
        rails = Rails()
        rails.when(condition).inject(message)
        
        @with_rails(rails)
        def create_agent():
            return CodeAgent(tools=[], model="gpt-4")
        
        # Now the agent includes Rails injection
        adapter = create_agent()
        result = await adapter.run("Create a data analysis script")
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            agent = func(*args, **kwargs)
            return create_smolagents_adapter(agent, rails)
        return wrapper
    return decorator