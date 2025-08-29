# Rails | Put your agents on rails

**Production-grade lifecycle management for AI agents - inject context and execute workflows when conditions are met**

Rails provides a comprehensive, framework-agnostic system for surgical control over AI agent behavior. Inject messages, execute workflows, compose lifecycle functions, and orchestrate complex patterns - all triggered by your custom conditions. No forced lifecycle phases, maximum flexibility.

```bash
pip install agent-rails
# or
pdm add agent-rails
```

**Built by: [Rizome Labs](https://rizome.dev) | Contact: [hi@rizome.dev](mailto:hi@rizome.dev)**

## Quick Start

```python
import asyncio
from rails import Rails, current_rails, lifecycle_function

# Define a tool that accesses Rails
def api_tool(data):
    rails = current_rails()  # Access Rails from within tools
    rails.store.increment_sync('api_calls')
    return {"processed": True, "calls": rails.store.get_counter_sync('api_calls')}

# Define composable lifecycle function
@lifecycle_function(priority=10)
async def setup_monitoring(rails):
    rails.store.set_sync('monitoring_active', True)
    print("🔧 Monitoring started")
    yield  # Main execution happens here
    print("🔧 Monitoring stopped")

# Define workflow function
async def error_recovery(rails):
    print("🔄 Running error recovery workflow")
    rails.store.set_counter_sync('errors', 0)

async def main():
    # Create Rails with lifecycle functions
    rails = Rails().with_lifecycle(setup_monitoring)
    
    # Message injection: When errors occur, inject helpful message
    rails.when(lambda s: s.get_counter_sync('errors') >= 2).inject({
        "role": "system", 
        "content": "Multiple errors detected. Switching to recovery mode."
    })
    
    # Workflow execution: When API calls exceed limit, run background optimization
    rails.when(lambda s: s.get_counter_sync('api_calls') >= 5).then(
        lambda r: print("⚡ Running optimization..."), background=True
    )
    
    # Error recovery workflow
    rails.when(lambda s: s.get_counter_sync('errors') >= 3).then(error_recovery)
    
    # Use Rails with automatic lifecycle management
    async with rails:
        messages = [{"role": "user", "content": "Help me process data"}]
        
        # Simulate tool calls and errors
        for i in range(6):
            result = api_tool({"item": i})
            if i > 3: rails.store.increment_sync('errors')
            
            # Check conditions - both inject messages AND execute workflows
            messages = await rails.check(messages)
            
            # Display any Rails messages
            for msg in messages:
                if "system" in msg.get("role", ""):
                    print(f"💬 Rails: {msg['content']}")

asyncio.run(main())
```

# Documentation

## Core Features

### 1. Message Injection - `when().inject()`

Conditionally inject messages into agent conversations:

```python
from rails import Rails, CounterCondition, StateCondition

rails = Rails()

# Lambda conditions for custom logic
rails.when(lambda s: s.get_counter_sync('errors') >= 3).inject({
    "role": "system",
    "content": "Multiple errors detected. Switching to recovery mode."
})

# Built-in condition types for common patterns
rails.when(CounterCondition('attempts', 5, '>=')).inject(retry_message)
rails.when(StateCondition('mode', 'expert')).inject(expert_guidance)

# Multiple injection strategies
rails.when(condition).inject(message)  # append (default)
rails.when(condition).inject(message, strategy='prepend')  # add to start
rails.when(condition).inject(message, strategy='replace_last')  # replace last

# Convenience methods
rails.on_counter('turns', 10, stop_message)
rails.on_state('debug_mode', True, debug_message)

# Apply all conditions and get enhanced messages
enhanced_messages = await rails.check(original_messages)
```

### 2. Workflow Execution - `when().then()`

Execute functions and workflows when conditions are met:

```python
from rails import Rails

rails = Rails()

# Execute function when condition met
async def error_recovery(rails):
    print("Running error recovery...")
    rails.store.set_counter_sync('errors', 0)
    
rails.when(lambda s: s.get_counter_sync('errors') >= 3).then(error_recovery)

# Execute in background (non-blocking)
rails.when(condition).then(
    optimization_workflow, 
    background=True,  # runs asynchronously
    name="background_optimizer"
)

# Pass additional arguments
rails.when(condition).then(
    custom_workflow, 
    arg1, arg2,  # positional args
    param1="value",  # keyword args
    background=False
)

# Lambda functions for simple workflows
rails.when(condition).then(lambda r: r.store.set('recovery_mode', True))
```

### 3. Composable Lifecycle Functions - `@lifecycle_function`

Create modular, reusable lifecycle components:

```python
from rails import Rails, lifecycle_function

@lifecycle_function(name="database", priority=10)
async def database_lifecycle(rails):
    # Setup phase
    connection = await create_db_connection()
    rails.store.set_sync('db_connection', connection)
    print("🔧 Database connected")
    
    yield  # Main execution happens here
    
    # Cleanup phase
    await connection.close()
    print("🔧 Database disconnected")

@lifecycle_function(name="monitoring", priority=5)
async def monitoring_lifecycle(rails):
    # Setup monitoring
    rails.store.set_sync('monitoring_active', True)
    
    # Add conditional rules during setup
    rails.when(lambda s: s.get_counter_sync('errors') >= 5).inject({
        "role": "system",
        "content": "🚨 High error rate detected!"
    })
    
    yield
    
    # Cleanup monitoring
    rails.store.set_sync('monitoring_active', False)

# Compose multiple lifecycle functions
rails = Rails().with_lifecycle('database', 'monitoring', custom_lifecycle)

# Or compose with function references
rails = Rails().with_lifecycle(database_lifecycle, monitoring_lifecycle)
```

### 4. Global Tool Access - `current_rails()`

Tools can access and manipulate the Rails instance they're running within:

```python
from rails import current_rails

def api_client_tool(endpoint, data):
    """Tool that tracks usage and adds conditional behaviors."""
    rails = current_rails()  # Access the active Rails instance
    
    # Track API usage
    rails.store.increment_sync('api_calls')
    rails.store.increment_sync(f'api_calls_{endpoint}')
    
    # Tool can add conditional rules based on its state
    if rails.store.get_counter_sync('api_calls') >= 8:
        rails.when(lambda s: s.get_counter_sync('api_calls') >= 10).inject({
            "role": "system",
            "content": "⚠️ API rate limit approaching. Consider throttling."
        })
    
    # Simulate API call
    result = call_external_api(endpoint, data)
    
    # Track errors
    if not result.get('success'):
        rails.store.increment_sync('api_errors')
    
    return result

# Tools automatically access Rails when called within Rails context
async with Rails() as rails:
    result = api_client_tool('process', {'data': 'value'})  # Works seamlessly
```

## Advanced Features

### 5. Background Execution & Orchestration

Execute complex workflows with proper concurrency and orchestration:

```python
from rails import Rails, WorkflowOrchestrator, execute_background_workflow

async def complex_data_processing(rails):
    # Long-running data processing
    await process_large_dataset()
    return {"processed": True}

async def quick_validation(rails):  
    # Quick validation task
    return {"valid": True}

# Background execution
rails = Rails()

async with rails:
    # Execute single workflow in background
    task_id = await execute_background_workflow(
        complex_data_processing, 
        rails_instance=rails,
        task_id="data_proc_001"
    )
    
    # Continue other work while background task runs
    messages = await rails.check(messages)
    
    # Advanced orchestration
    orchestrator = WorkflowOrchestrator(rails, max_concurrent=3)
    
    async with orchestrator.orchestration_context():
        # Conditional pipeline - steps run based on conditions
        pipeline_steps = [
            (lambda s: True, initialize_system),
            (lambda s: s.get_counter_sync('items') > 0, process_items),
            (lambda s: s.get_sync('validation_required'), validate_results),
            (lambda s: True, finalize_results)
        ]
        
        pipeline_results = await orchestrator.execute_conditional_pipeline(pipeline_steps)
        
        # Parallel execution
        parallel_workflows = [complex_data_processing, quick_validation, cleanup_temp_data]
        parallel_results = await orchestrator.execute_parallel_workflows(
            parallel_workflows, 
            wait_all=True
        )
```

### 6. Built-in Condition Types

Rails provides powerful condition primitives:

```python
from rails import (
    CounterCondition, StateCondition, LambdaCondition,
    AndCondition, OrCondition, NotCondition
)

# Counter conditions with operators
rails.when(CounterCondition('api_calls', 10, '>=')).inject(rate_limit_msg)
rails.when(CounterCondition('errors', 0, '==')).inject(success_msg)

# State conditions
rails.when(StateCondition('mode', 'production')).inject(prod_warning)
rails.when(StateCondition('user_tier', 'premium')).inject(premium_features)

# Logical combinations
complex_condition = AndCondition([
    CounterCondition('attempts', 3, '>='),
    StateCondition('auto_retry', True)
])
rails.when(complex_condition).then(auto_retry_workflow)

# Custom logic with lambda conditions
rails.when(LambdaCondition(lambda s: 
    s.get_counter_sync('success_rate') / s.get_counter_sync('total_attempts') < 0.5
)).inject(low_success_warning)
```

## Framework Integration

### 7. Framework Adapters

Rails includes adapters for seamless integration with popular agent frameworks:

```python
from rails.adapters import create_adapter, BaseRailsAdapter

# Generic adapter with any processing function
def my_agent_processor(messages):
    # Your agent processing logic here
    return {"role": "assistant", "content": "Processed with Rails!"}

adapter = create_adapter(rails, my_agent_processor)

# Context manager handles Rails lifecycle automatically
async with adapter as active_adapter:
    result = await active_adapter.run(messages)
```

#### LangChain Integration

```python
from rails.adapters import LangChainAdapter
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Set up Rails with conditions
rails = Rails()
rails.when(lambda s: s.get_counter_sync('messages') >= 3).inject({
    "role": "system",
    "content": "This conversation is getting long. Let me summarize..."
})

# Create LangChain adapter
llm = ChatOpenAI(model="gpt-4")
adapter = LangChainAdapter(rails, llm)

# Rails automatically applies before LangChain processing
messages = [HumanMessage(content="Help me debug this code")]
result = await adapter.run(messages)
```

#### SmolAgents Integration

```python
from rails.adapters import SmolAgentsAdapter
from smolagents import Agent

# Set up Rails with agent-specific conditions
rails = Rails()
rails.when(lambda s: s.get_counter_sync('tool_calls') >= 5).inject({
    "role": "system", 
    "content": "I notice I'm using many tools. Let me focus on the core task."
})

# Create SmolAgents adapter  
agent = Agent(model="gpt-4", tools=[web_search, calculator])
adapter = SmolAgentsAdapter(rails, agent)

result = await adapter.run("Research the latest AI developments and calculate ROI")
```

#### Custom Framework Adapter

```python
from rails.adapters import BaseRailsAdapter

class MyFrameworkAdapter(BaseRailsAdapter):
    def __init__(self, rails, my_agent):
        super().__init__(rails)
        self.agent = my_agent
    
    async def process_messages(self, messages, **kwargs):
        # Convert Rails messages to your framework format
        framework_messages = self.convert_messages(messages)
        
        # Process with your framework
        result = await self.agent.process(framework_messages, **kwargs)
        
        # Update Rails state based on result
        await self.rails.store.increment('framework_calls')
        
        return result
    
    async def update_rails_state(self, original_messages, modified_messages, result):
        # Custom state updates beyond the default
        await super().update_rails_state(original_messages, modified_messages, result)
        
        # Track framework-specific metrics
        if result.get('tool_used'):
            await self.rails.store.increment('tool_usage')

# Usage
adapter = MyFrameworkAdapter(rails, my_custom_agent)
result = await adapter.run(messages)
```

### 8. State Management

Rails provides comprehensive thread-safe state management:

```python
from rails import Rails

async with Rails() as rails:
    # COUNTERS - for tracking numeric values
    rails.store.increment_sync('api_calls')  # increment by 1
    rails.store.increment_sync('errors', 5)  # increment by custom amount
    rails.store.set_counter_sync('retries', 0)  # set to specific value
    
    current_calls = rails.store.get_counter_sync('api_calls', default=0)
    has_errors = rails.store.get_counter_sync('errors') > 0
    
    # STATE VALUES - for storing arbitrary data  
    rails.store.set_sync('user_tier', 'premium')
    rails.store.set_sync('config', {'debug': True, 'retries': 3})
    rails.store.set_sync('last_error', None)
    
    user_tier = rails.store.get_sync('user_tier', 'standard')
    config = rails.store.get_sync('config', {})
    has_config = rails.store.exists_sync('config')
    
    # ASYNC VERSIONS - for use in async contexts
    await rails.store.increment('async_counter')
    await rails.store.set('async_state', 'value')
    value = await rails.store.get('async_state')
    
    # BULK OPERATIONS
    await rails.store.clear()  # clear all state
    rails.store.delete_sync('old_key')  # remove specific key
```

### 9. Context Manager & Lifecycle

Rails supports both manual and automatic lifecycle management:

```python
# Automatic lifecycle with context manager (recommended)
async with Rails() as rails:
    # Rails instance available globally via current_rails()
    rails.when(condition).inject(message)
    rails.when(condition).then(workflow)
    result = await rails.check(messages)
    # Cleanup handled automatically on exit

# Manual lifecycle management
rails = Rails()
try:
    rails_instance = await rails.__aenter__()  # Manual setup
    # Use rails...
    result = await rails.check(messages)
finally:
    await rails.__aexit__(None, None, None)  # Manual cleanup

# With lifecycle functions
async with Rails().with_lifecycle('database', 'monitoring') as rails:
    # All lifecycle functions activated automatically
    result = await rails.check(messages)
    # Lifecycle functions cleaned up in reverse order
```

# Installation & Development

## Installation

```bash
# Core Rails package
pip install agent-rails

# Or with PDM
pdm add agent-rails

# With optional framework dependencies
pip install agent-rails[adapters]  # includes langchain, smolagents
pip install agent-rails[dev]       # includes development tools

# Framework-specific installation
pip install "agent-rails[adapters]" langchain
pip install "agent-rails[adapters]" smolagents
```

## Development

```bash
# Clone and set up development environment
git clone https://github.com/rizome-dev/rails
cd rails
pdm install --dev

# Run tests
pdm run test
pdm run test-cov  # with coverage report

# Code quality
pdm run lint      # ruff linting
pdm run format    # black formatting  
pdm run typecheck # mypy type checking

# Build and publish (maintainers only)
pdm run build    # build wheel and sdist
pdm run check    # verify built packages
```

---

## Examples & Community  

- **Enhanced Example**: [`enhanced_example.py`](https://github.com/rizome-dev/rails/blob/main/examples/enhanced_example.py) - Full demonstration of all capabilities
- **Adapter Examples**: [`adapter_example.py`](https://github.com/rizome-dev/rails/blob/main/examples/adapter_example.py) - Framework integration patterns  
- **Documentation**: [GitHub Repository](https://github.com/rizome-dev/rails)
- **Issues & Feature Requests**: [GitHub Issues](https://github.com/rizome-dev/rails/issues)

**Built with ❤️ by [Rizome Labs, Inc.](https://rizome.dev)**
