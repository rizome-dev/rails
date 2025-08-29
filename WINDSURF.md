<!-- Current Date: 2025-08-29 14:36:30 UTC -->

# WINDSURF.md

This file is managed by Rizome CLI. Do not edit directly.
Update RIZOME.md and run 'rizome sync' instead.

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
- `BaseRailsAdapter`: Abstract base for framework integration
- `LangChainAdapter`: Integration with LangChain
- `SmolaAgentsAdapter`: Integration with SmolaAgents
- Generic `create_adapter()` for any processing function

#### Key Design Patterns

1. **Fluent Interface**: `rails.when(condition).inject(message)` chains
2. **Context Variables**: `current_rails()` for global access within tools
3. **Generator-based Lifecycle**: Using `yield` for setup/teardown phases
4. **Strategy Pattern**: Different injection and execution strategies
5. **Decorator Pattern**: `@lifecycle_function` for modular components

### Testing Guidelines

- All new features require corresponding tests in `tests/`
- Use `pytest.mark.asyncio` for async test functions
- Mock external dependencies and API calls
- Test both success and failure conditions
- Verify thread safety for Store operations
- Test lifecycle management with context managers

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
1. Extend `BaseRailsAdapter` in `adapters/`
2. Implement `process_messages()` for framework-specific logic
3. Override `update_rails_state()` if needed
4. Add example in `examples/adapters.py`

#### Adding a Lifecycle Function
1. Use `@lifecycle_function` decorator
2. Implement setup before `yield`, cleanup after
3. Register in `LifecycleRegistry` if built-in
4. Document priority and dependencies

### Important Conventions

- **Async-first design**: Prefer async methods, provide sync wrappers where needed
- **Type hints**: All public APIs must have complete type annotations
- **Error handling**: Use specific exceptions, never silent failures
- **Thread safety**: Store operations must be thread-safe
- **Documentation**: All public functions need docstrings with examples
- **Testing**: New features require tests before merging
- **Backwards compatibility**: Follow semantic versioning

## WINDSURF-Specific Instructions

Windsurf-specific instructions
