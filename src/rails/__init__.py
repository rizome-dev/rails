"""Rails - Conditional message injection and workflow execution for AI agents."""

__version__ = "0.1.0"

from .core import Rails, InjectionRule, ExecutionRule, current_rails
from .store import Store
from .types import Message, Condition, Injector
from .conditions import (
    LambdaCondition, CounterCondition, StateCondition,
    AndCondition, OrCondition, NotCondition,
    counter_at_least, counter_equals, state_equals, state_in
)
from .injectors import (
    AppendInjector, PrependInjector, InsertInjector, ReplaceInjector,
    ConditionalInjector, append, prepend, insert_at, replace_last, 
    replace_all, replace_where
)
from .lifecycle import (
    lifecycle_function, LifecycleFunction, LifecycleRegistry,
    with_lifecycle_functions, LifecycleManager,
    counter_tracker_lifecycle, error_handler_lifecycle
)
from .execution import (
    ExecutionTask, BackgroundExecutor, get_background_executor,
    execute_background_workflow, background_execution_context,
    WorkflowOrchestrator
)

__all__ = [
    # Core classes
    "Rails",
    "Store",
    "InjectionRule",
    "ExecutionRule",
    "current_rails",
    
    # Type definitions
    "Message",
    "Condition", 
    "Injector",
    
    # Condition classes
    "LambdaCondition",
    "CounterCondition", 
    "StateCondition",
    "AndCondition",
    "OrCondition",
    "NotCondition",
    
    # Condition helpers
    "counter_at_least",
    "counter_equals",
    "state_equals", 
    "state_in",
    
    # Injector classes
    "AppendInjector",
    "PrependInjector", 
    "InsertInjector",
    "ReplaceInjector",
    "ConditionalInjector",
    
    # Injector helpers
    "append",
    "prepend",
    "insert_at",
    "replace_last",
    "replace_all",
    "replace_where",
    
    # Lifecycle management
    "lifecycle_function",
    "LifecycleFunction",
    "LifecycleRegistry", 
    "with_lifecycle_functions",
    "LifecycleManager",
    "counter_tracker_lifecycle",
    "error_handler_lifecycle",
    
    # Background execution
    "ExecutionTask",
    "BackgroundExecutor",
    "get_background_executor",
    "execute_background_workflow",
    "background_execution_context",
    "WorkflowOrchestrator",
    
    # Adapters (note: specific adapters available based on installed dependencies)
    "adapters",
]

# Import adapters submodule
from . import adapters
