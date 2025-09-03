# Rails | Lifecycle Orchestration for AI Agents

**Production-grade lifecycle orchestration for AI agents - monitor execution state and inject contextual guidance at critical moments**

Rails provides a framework-agnostic orchestration layer that creates a bidirectional communication channel between your agents and their lifecycle. Through a shared state store accessible to both Rails conditions and agent tools, Rails enables sophisticated feedback loops and intervention patterns.

```bash
pip install agent-rails
# or
pdm add agent-rails

pdm add agent-rails[smolagents] # smolagents adapter
pdm add agent-rails[langchain] # langchain adapter
pdm add agent-rails[all] # all adapters
```

**Built by: [Rizome Labs](https://rizome.dev) | Contact: [hi@rizome.dev](mailto:hi@rizome.dev)**

## Quick Start

```python
import os
import asyncio
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
from rails import Rails, current_rails, Message, Role, state, system
from rails.adapters import SmolAgentsAdapter

model = LiteLLMModel(
    model_id="openrouter/google/gemini-2.5-flash-lite",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, stream_outputs=True)

async def main():
    # Create Rails instance
    rails = Rails()
    
    # Setup DEBUG_MODE using new Rails rules
    await rails.store.set('DEBUG_MODE', True)
    
    # Add debug injection rule using new syntax
    rails.add_rule(
        condition=state('DEBUG_MODE') == True,
        action=system("Running in DEBUG mode. Please log all important details."),
        name="debug_mode_injection"
    )

    # Create adapter and wrap the agent
    adapter = SmolAgentsAdapter(rails)
    
    async with rails:
        wrapped_agent = await adapter.wrap(agent)
        
        # Use the agent exactly as you normally would!
        try:
            result = wrapped_agent.run("Research the latest AI developments and calculate ROI")
            print(f"Agent result: {str(result)[:100]}...")
            
            # Check Rails metrics
            turns = await rails.store.get_counter("agent_runs")
            injections = await rails.store.get_counter("injections", 0)
            print(f"\n📊 Rails Metrics: {turns} runs, {injections} injections")
        except Exception as e:
            print(f"⚠️ Agent error (expected without valid API): {str(e)[:60]}...")
```

## Architectural Principles

Rails implements a **bidirectional shared state model** that enables sophisticated lifecycle orchestration:

```
Agent Tools (Write State) ←→ Rails Store (Monitor State) ←→ Rails Conditions (Inject Context)
```

## Core Components

### 1. Fluent Condition Builders

Rails provides intuitive condition builders for common patterns:

```python
from rails import Rails, counter, state, queue

rails = Rails()

# Counter conditions with comparison operators
rails.add_rule(
    condition=counter("api_calls") >= 10,
    action=lambda msgs: msgs + [{"role": "system", "content": "API limit approaching"}]
)

# State conditions with equality checks
rails.add_rule(
    condition=state("mode") == "production",
    action=lambda msgs: msgs + [{"role": "system", "content": "In production - be careful"}]
)

# Queue conditions for task management
rails.add_rule(
    condition=queue("errors").is_empty,
    action=lambda msgs: msgs + [{"role": "system", "content": "All errors resolved!"}]
)

# Composite conditions
from rails import AndCondition, OrCondition, NotCondition

complex_condition = AndCondition(
    counter("attempts") >= 3,
    state("retry_enabled") == True
)
rails.add_rule(condition=complex_condition, action=retry_handler)
```

### 2. Shared State Store

The Rails store provides thread-safe state management accessible to both Rails and agent tools:

```python
from rails import Rails, current_rails

async with Rails() as rails:
    # Counters - for tracking numeric values
    await rails.store.increment("api_calls")  # +1
    await rails.store.increment("errors", 5)  # +5
    await rails.store.reset_counter("retries")
    count = await rails.store.get_counter("api_calls")
    
    # State values - for arbitrary data
    await rails.store.set("user_tier", "premium")
    await rails.store.set("config", {"debug": True, "timeout": 30})
    tier = await rails.store.get("user_tier", default="standard")
    
    # Queues - for task management (FIFO by default)
    await rails.store.push_queue("tasks", "process_data")
    await rails.store.push_queue("tasks", "generate_report")
    task = await rails.store.pop_queue("tasks")  # "process_data"
    pending = await rails.store.queue_length("tasks")  # 1
    all_tasks = await rails.store.get_queue("tasks")  # ["generate_report"]
    
    # Synchronous versions for use in tools
    rails.store.increment_sync("tool_calls")
    rails.store.set_sync("last_tool", "calculator")
    value = rails.store.get_sync("last_tool")
    count = rails.store.get_counter_sync("tool_calls")  # Get counter synchronously
    rails.store.push_queue_sync("tasks", "new_task")     # Push to queue synchronously
```

### 3. Tool Integration with `current_rails()`

Tools can access the Rails instance they're running within:

```python
from rails import current_rails

def data_processing_tool(data):
    """Tool that participates in lifecycle orchestration."""
    rails = current_rails()  # Get active Rails instance
    
    # Track tool usage
    rails.store.increment_sync('tool_calls')
    rails.store.increment_sync(f'tool_calls_data_processing')
    
    # Add tasks to queue for later processing
    if data.get('requires_validation'):
        rails.store.push_queue_sync('validation_queue', data['id'])
    
    # Update state based on tool results
    try:
        result = process_data(data)
        rails.store.increment_sync('successful_processing')
    except Exception as e:
        rails.store.increment_sync('processing_errors')
        rails.store.push_queue_sync('error_log', str(e))
        result = None
    
    # Check if we should slow down
    if rails.store.get_counter_sync('processing_errors') > 5:
        rails.store.set_sync('mode', 'careful')
    
    return result

# Tools automatically access Rails when called within Rails context
async with Rails() as rails:
    # Tool can now use current_rails() to access the store
    result = data_processing_tool({'data': 'value', 'requires_validation': True})
```

### 4. Message Injection System

Rails uses a functional approach to message transformation:

```python
from rails import Rails, Message, Role
from rails import AppendInjector, PrependInjector, ReplaceInjector
from rails import system, template

rails = Rails()

# Simple function-based injection
rails.add_rule(
    condition=counter("errors") > 0,
    action=lambda msgs: msgs + [Message(role=Role.SYSTEM, content="Error detected")]
)

# Using injector classes
error_injector = AppendInjector(
    message=Message(role=Role.SYSTEM, content="Please review the errors")
)
rails.add_rule(
    condition=counter("errors") >= 3,
    action=error_injector.inject
)

# Factory functions for common patterns
rails.add_rule(
    condition=state("mode") == "debug",
    action=system("Debug mode active - verbose output enabled")
)

# Template injection with store values
rails.add_rule(
    condition=state("user_name").exists,
    action=template("Hello {user_name}, you have {api_calls} API calls remaining")
)

# Process messages through all rules
messages = [Message(role=Role.USER, content="Hello")]
processed = await rails.process(messages)
```

### 5. Event Streaming & Observability

Rails emits events for all state changes, enabling monitoring and debugging:

```python
from rails import Rails

rails = Rails()

# Subscribe to events
async def event_handler(event):
    print(f"Event: {event.event_type} - {event.key} = {event.value}")

rails.store.subscribe_events(event_handler)

# All state changes emit events
await rails.store.increment("counter", triggered_by="user_action")
await rails.store.set("state", "active", triggered_by="system")
await rails.store.push_queue("tasks", "item", triggered_by="tool")

# Stream events for real-time monitoring
async for event in rails.store.event_stream():
    if event.event_type == "counter_increment":
        print(f"Counter {event.key} changed: {event.previous_value} → {event.value}")

# Get metrics snapshot
metrics = await rails.emit_metrics()
print(f"Active rules: {metrics['active_rules']}")
print(f"Store snapshot: {metrics['store_snapshot']}")
```

## Framework Integration

### Framework Adapters

Rails provides transparent adapters that wrap your existing agents and models, automatically injecting Rails lifecycle management without changing how you use them:

```python
from rails import Rails
from rails.adapters import create_adapter

# Your existing agent function
def my_agent(messages):
    # Your agent logic here
    return {"role": "assistant", "content": "Response"}

rails = Rails()
adapter = create_adapter(rails)

async with rails:
    wrapped_agent = await adapter.wrap(my_agent)
    # Use wrapped_agent exactly like the original!
    result = wrapped_agent(messages)  # Rails processes transparently
```

### LangChain Integration

```python
from rails import Rails, counter, system
from rails.adapters import LangChainAdapter
from langchain_openai import ChatOpenAI

rails = Rails()

# Add Rails conditions
rails.add_rule(
    condition=counter("turns") >= 5,
    action=system("This conversation is getting long. Consider summarizing."),
    name="conversation_limit"
)

# Create adapter and wrap the model
adapter = LangChainAdapter(rails)
llm = ChatOpenAI(model="gpt-4")

async with rails:
    wrapped_llm = await adapter.wrap(llm)
    
    # Use exactly like the original - Rails magic happens automatically!
    messages = [{"role": "user", "content": "Hello, let's chat!"}]
    result = wrapped_llm.invoke(messages)  # Rails processes transparently
```

### Custom Framework Adapter

```python
from rails.adapters import BaseAdapter

class MyFrameworkAdapter(BaseAdapter):
    def __init__(self, rails, agent):
        super().__init__(rails)
        self.agent = agent
    
    async def process_messages(self, messages, **kwargs):
        # Apply Rails processing
        processed = await self.rails.process(messages)
        
        # Convert to framework format
        framework_messages = self.to_framework_format(processed)
        
        # Process with framework
        result = await self.agent.process(framework_messages)
        
        # Update Rails state
        await self.rails.store.increment("framework_calls")
        
        return result
```

## Advanced Patterns

### Queue-Based Task Management

```python
from rails import Rails, current_rails, queue

rails = Rails()

# Tool adds tasks to queue
def task_manager_tool(action, task=None):
    rails = current_rails()
    
    if action == "add":
        rails.store.push_queue_sync("tasks", task)
    elif action == "complete":
        completed = rails.store.pop_queue_sync("tasks")
        rails.store.increment_sync("completed_tasks")
        return completed
    
    return rails.store.get_queue_sync("tasks")

# Rails monitors queue and provides guidance
rails.add_rule(
    condition=queue("tasks").length > 5,
    action=lambda msgs: msgs + [{
        "role": "system",
        "content": "Multiple tasks pending. Focus on completion before adding more."
    }]
)

rails.add_rule(
    condition=queue("tasks").is_empty & (counter("idle_turns") > 2),
    action=lambda msgs: msgs + [{
        "role": "system", 
        "content": "No pending tasks. Consider asking the user for next steps."
    }]
)
```

### Error Recovery Pattern

```python
from rails import Rails, current_rails, counter

rails = Rails()

# Tool reports errors
def api_tool(endpoint):
    rails = current_rails()
    
    try:
        result = call_api(endpoint)
        rails.store.reset_counter_sync("consecutive_errors")
        return result
    except Exception as e:
        rails.store.increment_sync("errors")
        rails.store.increment_sync("consecutive_errors")
        rails.store.push_queue_sync("error_log", {
            "endpoint": endpoint,
            "error": str(e),
            "timestamp": datetime.now()
        })
        
        if rails.store.get_counter_sync("consecutive_errors") >= 3:
            rails.store.set_sync("mode", "recovery")
        
        return None

# Rails provides recovery guidance
rails.add_rule(
    condition=state("mode") == "recovery",
    action=lambda msgs: msgs + [{
        "role": "system",
        "content": "In recovery mode. Try alternative approaches or ask for help."
    }]
)
```

### Progress Tracking

```python
from rails import Rails, current_rails

rails = Rails()

# Tools update progress
def step_tool(step_name, status):
    rails = current_rails()
    
    rails.store.set_sync(f"step_{step_name}", status)
    
    if status == "complete":
        rails.store.increment_sync("completed_steps")
        total = rails.store.get_counter_sync("total_steps", 10)
        completed = rails.store.get_counter_sync("completed_steps")
        
        if completed == total:
            rails.store.set_sync("workflow_status", "complete")
    
    return {"step": step_name, "status": status}

# Rails provides progress updates
rails.add_rule(
    condition=counter("completed_steps") % 5 == 0,  # Every 5 steps
    action=lambda msgs: msgs + [{
        "role": "system",
        "content": f"Good progress! {rails.store.get_counter_sync('completed_steps')} steps completed."
    }]
)
```

## Configuration

### Store Configuration

```python
from rails import Rails, StoreConfig, QueueConfig

config = StoreConfig(
    persist_on_exit=True,
    persistence_path="./rails_state.json",
    emit_events=True,
    max_event_history=1000,
    default_queues={
        "tasks": QueueConfig(max_size=100, fifo=True, auto_dedup=True),
        "errors": QueueConfig(max_size=50, fifo=False),  # LIFO for errors
    }
)

rails = Rails(store=Store(config=config))
```

### Middleware Stack

```python
from rails import Rails

rails = Rails()

# Add middleware for processing
async def logging_middleware(messages, store):
    await store.increment("middleware_calls")
    print(f"Processing {len(messages)} messages")
    return messages

async def metric_middleware(messages, store):
    start = time.time()
    result = messages
    duration = time.time() - start
    await store.set("last_processing_time", duration)
    return result

rails.add_middleware(logging_middleware)
rails.add_middleware(metric_middleware)

# Process through middleware stack
result = await rails.process_with_middleware(messages)
```

## Installation & Development

### Installation

```bash
# Core Rails package
pip install agent-rails

# Or with PDM
pdm add agent-rails

# With optional framework dependencies
pip install agent-rails[adapters]  # includes framework adapters
pip install agent-rails[dev]       # includes development tools
```

### Development

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

## API Reference

### Rails

- `Rails()` - Create Rails instance
- `add_rule(condition, action, name=None, priority=0)` - Add orchestration rule
- `process(messages)` - Process messages through rules
- `process_with_middleware(messages)` - Process through middleware stack
- `add_middleware(middleware)` - Add middleware function
- `emit_metrics()` - Get metrics snapshot

### Store

**Async Methods:**
- `increment(key, amount=1)` - Increment counter
- `get_counter(key, default=0)` - Get counter value
- `reset_counter(key)` - Reset counter to zero
- `set(key, value)` - Set state value
- `get(key, default=None)` - Get state value
- `delete(key)` - Delete state key
- `push_queue(queue, item)` - Add item to queue
- `pop_queue(queue)` - Remove and return item from queue
- `get_queue(queue)` - Get all queue items
- `queue_length(queue)` - Get queue length
- `clear_queue(queue)` - Clear all items from queue
- `get_snapshot()` - Get complete state snapshot
- `clear()` - Clear all state

**Synchronous Methods (for use in tools):**
- `increment_sync(key, amount=1)` - Increment counter synchronously
- `get_counter_sync(key, default=0)` - Get counter value synchronously
- `get_sync(key, default=None)` - Get state value synchronously
- `set_sync(key, value)` - Set state value synchronously
- `push_queue_sync(queue, item)` - Add item to queue synchronously

### Conditions

- `counter(key)` - Create counter condition builder
- `state(key)` - Create state condition builder  
- `queue(name)` - Create queue condition builder
- `AndCondition(*conditions)` - All conditions must be true
- `OrCondition(*conditions)` - Any condition must be true
- `NotCondition(condition)` - Negate condition
- `AlwaysCondition()` - Always true
- `NeverCondition()` - Always false

### Injectors

- `AppendInjector(message)` - Append message to end
- `PrependInjector(message)` - Prepend message to start
- `InsertInjector(message, index)` - Insert at index
- `ReplaceInjector(messages)` - Replace all messages
- `system(content, position="append")` - System message factory
- `template(template, role=Role.SYSTEM)` - Template message factory

---

**Built with ❤️ by [Rizome Labs, Inc.](https://rizome.dev)**
