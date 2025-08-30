# Rails | Lifecycle Orchestration for AI Agents

**Production-grade lifecycle orchestration for AI agents - monitor execution state and inject contextual guidance at critical moments**

Rails provides a framework-agnostic orchestration layer that creates a bidirectional communication channel between your agents and their lifecycle. Through a shared state store accessible to both Rails conditions and agent tools, Rails enables sophisticated feedback loops and intervention patterns.

```bash
pip install agent-rails
# or
pdm add agent-rails
```

**Built by: [Rizome Labs](https://rizome.dev) | Contact: [hi@rizome.dev](mailto:hi@rizome.dev)**

## Quick Start

```python
import asyncio
from rails import Rails, current_rails, counter, state, queue

# Define a tool that accesses Rails
def fetch_data(data):
    rails = current_rails()  # Access Rails from within tools
    rails.store.increment_sync('api_calls')
    
    # Tool can push tasks to queues
    if data.get('needs_processing'):
        rails.store.push_queue_sync('tasks', data['item'])
    
    return {"processed": True, "calls": rails.store.get_counter_sync('api_calls')}

async def main():
    rails = Rails()
    
    # Fluent condition builders with message injection
    rails.add_rule(
        condition=counter("errors") >= 2,
        action=lambda msgs: msgs + [{
            "role": "system", 
            "content": "Multiple errors detected. Switching to recovery mode."
        }]
    )
    
    # Queue-based orchestration
    rails.add_rule(
        condition=queue("tasks").length > 5,
        action=lambda msgs: msgs + [{
            "role": "system",
            "content": "Task queue is growing. Focus on completion before taking new tasks."
        }]
    )
    
    # State-based guidance
    rails.add_rule(
        condition=state("mode") == "exploration",
        action=lambda msgs: msgs + [{
            "role": "system",
            "content": "In exploration mode - be thorough but watch for diminishing returns."
        }]
    )
    
    async with rails:  # Context manager handles lifecycle
        messages = [{"role": "user", "content": "Help me process data"}]
        
        # Simulate tool calls and state changes
        for i in range(6):
            result = fetch_data({"item": i, "needs_processing": i > 2})
            if i > 3: 
                await rails.store.increment('errors')
            
            # Process messages through Rails conditions
            from rails import Message, Role
            rail_messages = [Message(role=Role(m["role"].upper()), content=m["content"]) for m in messages]
            processed = await rails.process(rail_messages)
            
            # Display any injected guidance
            for msg in processed[len(rail_messages):]:
                print(f"💬 Rails: {msg.content}")

asyncio.run(main())
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

Rails provides adapters for seamless integration with popular frameworks:

```python
from rails import Rails
from rails.adapters import create_adapter, BaseAdapter

# Generic adapter for any processing function
def my_agent(messages):
    # Your agent logic here
    return {"role": "assistant", "content": "Response"}

rails = Rails()
adapter = create_adapter(rails, my_agent)

async with adapter:
    result = await adapter.process_messages(messages)
```

### LangChain Integration

```python
from rails import Rails, counter
from rails.adapters import LangChainAdapter
from langchain_openai import ChatOpenAI

rails = Rails()

# Add Rails conditions
rails.add_rule(
    condition=counter("messages") >= 5,
    action=lambda msgs: msgs + [{
        "role": "system",
        "content": "This conversation is getting long. Consider summarizing."
    }]
)

# Create adapter
llm = ChatOpenAI(model="gpt-4")
adapter = LangChainAdapter(rails, llm)

# Rails processes messages before LangChain
result = await adapter.run(messages)
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
