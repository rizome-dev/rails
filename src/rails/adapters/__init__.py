"""Rails framework adapters for seamless integration with popular agent frameworks.

This package provides adapters that allow Rails conditional message injection
to work seamlessly with various agent frameworks like LangChain, SmolAgents,
and more. Each adapter maintains the framework-agnostic nature of Rails while
providing framework-specific optimizations.

Usage Examples:

LangChain Integration:
    from rails import Rails
    from rails.adapters import LangChainAdapter
    from langchain_openai import ChatOpenAI
    
    rails = Rails()
    rails.when(lambda s: s.get_counter_sync('turns') >= 3).inject({
        "role": "system",
        "content": "Let's wrap up this conversation."
    })
    
    llm = ChatOpenAI()
    adapter = LangChainAdapter(rails, llm)
    
    result = await adapter.run([
        {"role": "user", "content": "Hello!"}
    ])

SmolAgents Integration:
    from rails import Rails
    from rails.adapters import SmolAgentsAdapter
    from smolagents import CodeAgent
    
    rails = Rails()
    rails.when(lambda s: s.get_counter_sync('tool_calls') >= 2).inject({
        "role": "system",
        "content": "You've used tools. Consider explaining your approach."
    })
    
    agent = CodeAgent(tools=[], model="gpt-4")
    adapter = SmolAgentsAdapter(rails, agent)
    
    result = await adapter.run("Create a data visualization")

Custom Framework Integration:
    from rails.adapters import BaseAdapter
    
    class MyFrameworkAdapter(BaseAdapter):
        async def process_messages(self, messages, **kwargs):
            # Your framework-specific logic here
            return my_framework.process(messages)
    
    adapter = MyFrameworkAdapter(rails=rails)
    result = await adapter.process_messages(messages)
"""

from .base import BaseAdapter, GenericAdapter, MiddlewareAdapter, create_adapter

# Import adapters with graceful degradation for optional dependencies
try:
    from .langchain import LangChainAdapter, create_langchain_adapter
    from .langchain import with_rails as langchain_with_rails
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from .smolagents import (
        CodeAgentAdapter,
        SmolAgentsAdapter,
        create_smolagents_adapter,
    )
    from .smolagents import with_rails as smolagents_with_rails
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False


__all__ = [
    # Base adapter classes
    "BaseAdapter",
    "GenericAdapter",
    "MiddlewareAdapter",
    "create_adapter",
]

# Add framework-specific exports if available
if LANGCHAIN_AVAILABLE:
    __all__.extend([
        "LangChainAdapter",
        "create_langchain_adapter",
        "langchain_with_rails",
    ])

if SMOLAGENTS_AVAILABLE:
    __all__.extend([
        "SmolAgentsAdapter",
        "CodeAgentAdapter",
        "create_smolagents_adapter",
        "smolagents_with_rails",
    ])


def get_available_adapters() -> dict:
    """Get information about available framework adapters.
    
    Returns:
        Dictionary mapping framework names to adapter availability and classes
        
    Example:
        available = get_available_adapters()
        if available['langchain']['available']:
            adapter_class = available['langchain']['adapter']
    """
    adapters = {
        'base': {
            'available': True,
            'adapter': BaseAdapter,
            'description': 'Base adapter pattern for custom frameworks'
        },
        'generic': {
            'available': True,
            'adapter': GenericAdapter,
            'description': 'Generic adapter for framework-agnostic usage'
        }
    }

    if LANGCHAIN_AVAILABLE:
        adapters['langchain'] = {
            'available': True,
            'adapter': LangChainAdapter,
            'description': 'Adapter for LangChain chains, agents, and chat models'
        }
    else:
        adapters['langchain'] = {
            'available': False,
            'adapter': None,
            'description': 'LangChain adapter (requires: pip install langchain-core)',
            'install_hint': 'pip install langchain-core'
        }

    if SMOLAGENTS_AVAILABLE:
        adapters['smolagents'] = {
            'available': True,
            'adapter': SmolAgentsAdapter,
            'description': 'Adapter for SmolAgents agents'
        }
    else:
        adapters['smolagents'] = {
            'available': False,
            'adapter': None,
            'description': 'SmolAgents adapter (requires: pip install smolagents)',
            'install_hint': 'pip install smolagents'
        }

    return adapters


def print_adapter_status():
    """Print the status of all available adapters."""
    adapters = get_available_adapters()

    print("Rails Framework Adapters:")
    print("=" * 50)

    for name, info in adapters.items():
        status = "✓ Available" if info['available'] else "✗ Not Available"
        print(f"{name:12} {status:15} - {info['description']}")

        if not info['available'] and 'install_hint' in info:
            print(f"{'':12} {'':15}   Install: {info['install_hint']}")

    print()


# Convenience function for quick adapter creation
def auto_adapter(framework_object, rails=None):
    """Automatically create the appropriate adapter for a framework object.
    
    This function inspects the provided framework object and returns the
    appropriate Rails adapter.
    
    Args:
        framework_object: Framework-specific object (chain, agent, etc.)
        rails: Optional Rails instance
        
    Returns:
        Appropriate Rails adapter
        
    Example:
        from langchain_openai import ChatOpenAI
        from rails import Rails
        from rails.adapters import auto_adapter
        
        rails = Rails()
        llm = ChatOpenAI()
        
        # Automatically creates LangChainAdapter
        adapter = auto_adapter(llm, rails)
    """
    obj_type = str(type(framework_object))

    # LangChain detection
    if LANGCHAIN_AVAILABLE and 'langchain' in obj_type.lower():
        return LangChainAdapter(rails, framework_object)

    # SmolAgents detection
    if SMOLAGENTS_AVAILABLE and ('smolagents' in obj_type.lower() or 'agent' in obj_type.lower()):
        return create_smolagents_adapter(framework_object, rails)

    # Default to generic adapter
    return GenericAdapter(rails=rails or Rails())
