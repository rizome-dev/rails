"""Rails - Lifecycle orchestration for AI agents."""

__version__ = "0.2.0"

# Core
# Conditions
from .conditions import (
    AlwaysCondition,
    # Composite conditions
    AndCondition,
    ComparisonOperator,
    # Base classes
    ConditionBase,
    # Concrete conditions
    CounterCondition,
    LambdaCondition,
    NeverCondition,
    NotCondition,
    OrCondition,
    QueueCondition,
    StateCondition,
    # Builder functions
    counter,
    queue,
    state,
)
from .core import Rails, Rule, current_rails, rails_context

# Injectors
from .injectors import (
    # Concrete injectors
    AppendInjector,
    ConditionalInjector,
    # Base classes
    InjectorBase,
    InsertInjector,
    PrependInjector,
    ReplaceInjector,
    SystemInjector,
    TemplateInjector,
    # Factory functions
    append_message,
    prepend_message,
    system,
    template,
)
from .store import QueueConfig, Store, StoreConfig

# Types
from .types import (
    Condition,
    FrameworkAdapter,
    Injector,
    Message,
    RailState,
    Role,
    StateEvent,
)

# Lifecycle (if still present)
try:
    from .lifecycle import (
        LifecycleFunction,
        LifecycleManager,
        LifecycleRegistry,
        lifecycle_function,
        with_lifecycle_functions,
    )
except ImportError:
    # Lifecycle module may be optional
    pass

# Execution (if still present)
try:
    from .execution import (
        BackgroundExecutor,
        ExecutionTask,
        WorkflowOrchestrator,
        background_execution_context,
        execute_background_workflow,
        get_background_executor,
    )
except ImportError:
    # Execution module may be optional
    pass

# Adapters
from . import adapters
from .adapters.base import (
    BaseAdapter,
    GenericAdapter,
    MiddlewareAdapter,
    create_adapter,
    create_middleware_stack,
    rails_middleware,
)

__all__ = [
    # Version
    "__version__",

    # Core
    "Rails",
    "Rule",
    "current_rails",
    "rails_context",

    # Store
    "Store",
    "StoreConfig",
    "QueueConfig",

    # Types
    "Message",
    "Role",
    "StateEvent",
    "RailState",
    "Condition",
    "Injector",
    "FrameworkAdapter",

    # Conditions
    "ConditionBase",
    "ComparisonOperator",
    "CounterCondition",
    "StateCondition",
    "QueueCondition",
    "LambdaCondition",
    "AndCondition",
    "OrCondition",
    "NotCondition",
    "AlwaysCondition",
    "NeverCondition",
    "counter",
    "state",
    "queue",

    # Injectors
    "InjectorBase",
    "AppendInjector",
    "PrependInjector",
    "InsertInjector",
    "ReplaceInjector",
    "ConditionalInjector",
    "SystemInjector",
    "TemplateInjector",
    "append_message",
    "prepend_message",
    "system",
    "template",

    # Adapters
    "adapters",
    "BaseAdapter",
    "MiddlewareAdapter",
    "GenericAdapter",
    "create_adapter",
    "create_middleware_stack",
    "rails_middleware",
]
