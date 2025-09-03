<!-- You have been trained on a corpus that was cutoff at sometime before the current date. You must constantly ground yourself in the current date, to avoid utilizing outdated or wrong information. You have access to a powerful Web Search tool, as well as others, which can assist in grounding yourself in the current SOTA & Usage Guidelines for the task at hand. -->
<!-- Current Date: 2025-09-03 16:52:15 UTC -->

# QWEN.md

This file is managed by Rizome CLI. Do not edit directly.
Update RIZOME.md and run 'rizome sync' instead.

Rails is a Python library for production-grade lifecycle management of AI agents. It provides conditional message injection, workflow execution, and composable lifecycle functions triggered by custom conditions.

## Common Instructions

### Development Commands

#### Setup & Installation
```bash
# Install with PDM (preferred)
pdm install           # Install core dependencies
pdm install --dev     # Install with development dependencies
pdm venv create       # Create virtual environment if needed

# Alternative: pip
pip install -e .                    # Editable install
pip install -e ".[dev]"            # With dev dependencies
pip install -e ".[adapters]"       # With framework adapters
```

#### Running Tests
```bash
# Run all tests
pdm run test

# Run tests with coverage
pdm run test-cov

# Run specific test file
pdm run pytest tests/test_rails.py -v

# Run specific test function
pdm run pytest tests/test_rails.py::TestRails::test_basic_message_injection -v

# Run tests with specific marker or pattern
pdm run pytest -k "injection" -v
```

#### Code Quality
```bash
# Linting (using ruff)
pdm run lint

# Formatting (using black)
pdm run format                # Apply formatting
pdm run format-check          # Check formatting without changes

# Type checking (using mypy)
pdm run typecheck

# Clean build artifacts
pdm run clean
```

#### Building & Distribution
```bash
pdm run build         # Build wheel and sdist
pdm run check         # Verify built packages with twine
```

### Architecture Overview

#### Core Components

**Rails (`src/rails/core.py`)**
- Central orchestrator managing conditional rules and lifecycle
- Maintains global registry for `current_rails()` access
- Handles both injection rules and execution rules
- Context manager for automatic lifecycle management

**Store (`src/rails/store.py`)**
- Thread-safe state management with counters and arbitrary values
- Provides both sync and async interfaces
- Supports bulk operations and existence checks

**Conditions (`src/rails/conditions.py`)**
- Base `Condition` protocol with implementations:
  - `LambdaCondition`: Custom logic via lambda functions
  - `CounterCondition`: Numeric comparisons on counters
  - `StateCondition`: Value comparisons on state
  - Logical operators: `AndCondition`, `OrCondition`, `NotCondition`

**Injectors (`src/rails/injectors.py`)**
- Message manipulation strategies:
  - `AppendInjector`: Add to end
  - `PrependInjector`: Add to start
  - `InsertInjector`: Insert at index
  - `ReplaceInjector`: Replace messages
  - `ConditionalInjector`: Apply based on conditions

**Lifecycle (`src/rails/lifecycle.py`)**
- `@lifecycle_function` decorator for composable lifecycle components
- `LifecycleRegistry` for managing registered functions
- `LifecycleManager` for orchestrating setup/teardown
- Built-in lifecycle functions for common patterns

**Execution (`src/rails/execution.py`)**
- `BackgroundExecutor` for async workflow execution
- `WorkflowOrchestrator` for complex execution patterns
- Support for parallel, sequential, and conditional pipelines
- Global executor management with proper cleanup

**Adapters (`src/rails/adapters/`)**
- `BaseAdapter`: Abstract base for framework integration  
- `LangChainAdapter`: Transparent wrapper for LangChain runnables (chains, models)
- `SmolAgentsAdapter`: Transparent wrapper for SmolAgents agents  
- `CodeAgentAdapter`: Specialized wrapper for SmolAgents CodeAgent with enhanced tracking
- `GenericAdapter` and `MiddlewareAdapter` for custom integrations
- `create_adapter()` factory function for any processing function
- `auto_adapter()` automatically detects framework and creates appropriate adapter

#### Key Design Patterns

1. **Fluent Interface**: `rails.when(condition).inject(message)` chains
2. **Transparent Wrappers**: Adapters intercept methods via `__getattr__` while preserving original API
3. **Context Variables**: `current_rails()` for global access within tools
4. **Thread Pool Execution**: Sync methods run async Rails code in separate threads to avoid event loop conflicts
5. **Generator-based Lifecycle**: Using `yield` for setup/teardown phases
6. **Strategy Pattern**: Different injection and execution strategies
7. **Decorator Pattern**: `@lifecycle_function` for modular components and `@with_rails` for adapters

### Testing Guidelines

- All new features require corresponding tests in `tests/`
- Use `pytest.mark.asyncio` for async test functions
- Mock external dependencies and API calls with proper patching
- Test both success and failure conditions
- Verify thread safety for Store operations
- Test lifecycle management with context managers
- For adapter tests: Use `@patch` to mock `FRAMEWORK_AVAILABLE` flags
- Mock framework classes to avoid requiring actual framework installations
- Test wrapper behavior: method interception, context management, metrics tracking

### Common Development Tasks

#### Adding a New Condition Type
1. Implement the `Condition` protocol in `conditions.py`
2. Add convenience helper function if appropriate
3. Update `__all__` in `conditions.py`
4. Add tests in `test_rails.py`

#### Adding a New Injector Strategy
1. Implement the `Injector` protocol in `injectors.py`
2. Add convenience helper function
3. Update `__all__` in `injectors.py`
4. Add integration test showing usage

#### Creating a Framework Adapter
Modern adapters use transparent wrapping:
1. Extend `BaseAdapter` in `adapters/base.py`
2. Implement `wrap()` method that returns a wrapper class
3. Wrapper class uses `__getattr__` to proxy all methods to original object
4. Intercept key methods (e.g., `invoke`, `run`) to inject Rails processing
5. Use thread pools to run async Rails code in sync methods
6. Add example in `examples/adapters_demo.py` or create new example file
7. For official adapters (LangChain, SmolAgents), add to `src/rails/adapters/`

#### Adding a Lifecycle Function
1. Use `@lifecycle_function` decorator
2. Implement setup before `yield`, cleanup after
3. Register in `LifecycleRegistry` if built-in
4. Document priority and dependencies

#### Working with the Store
- Always use async methods in async contexts
- Use sync methods (ending in `_sync`) in tools or synchronous code
- Queue operations are FIFO by default, configure in `QueueConfig` for LIFO
- Counter operations are atomic and thread-safe
- State values support any JSON-serializable data

#### Creating Tools with Rails Integration
```python
from rails import current_rails

def my_tool(data):
    rails = current_rails()  # Access Rails from within tool
    rails.store.increment_sync('tool_calls')
    rails.store.push_queue_sync('tasks', data['id'])
    # Tool logic here
    return result
```

### Important Conventions

- **Async-first design**: Prefer async methods, provide sync wrappers where needed
- **Type hints**: All public APIs must have complete type annotations  
- **Error handling**: Use specific exceptions, never silent failures
- **Thread safety**: Store operations must be thread-safe
- **Documentation**: All public functions need docstrings with examples
- **Testing**: New features require tests before merging
- **Backwards compatibility**: Follow semantic versioning
- **Message format**: Use `Message(role=Role.X, content="...")` for type safety
- **Condition evaluation**: Conditions must implement async `evaluate(store)` method
- **Context management**: Always use `async with Rails()` for proper lifecycle

## QWEN-Specific Instructions

Qwen-specific instructions
