"""
Background execution capabilities for Rails with AnyIO task groups.

This module provides robust background execution patterns inspired by the
reference implementation, allowing workflows to run concurrently while
maintaining proper lifecycle management and error handling.
"""

import asyncio
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from anyio import CapacityLimiter
from loguru import logger


@dataclass
class ExecutionTask:
    """Represents a background execution task."""

    task_id: str
    workflow: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    error: Exception | None = None
    rails_instance: Optional['Rails'] = None


class BackgroundExecutor:
    """
    Background execution manager using AnyIO task groups.
    
    This class provides robust background execution capabilities with:
    - Concurrent task execution with limits
    - Task lifecycle tracking
    - Error handling and recovery
    - Integration with Rails lifecycle management
    """

    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent = max_concurrent_tasks
        self.task_limiter = CapacityLimiter(max_concurrent_tasks)
        self.active_tasks: dict[str, ExecutionTask] = {}
        self.completed_tasks: dict[str, ExecutionTask] = {}
        self._task_group = None
        self._executor_active = False

    async def start(self):
        """Start the background executor."""
        if self._executor_active:
            return

        self._executor_active = True
        logger.info(f"Started background executor with {self.max_concurrent} concurrent task limit")

    async def stop(self):
        """Stop the background executor and wait for tasks to complete."""
        if not self._executor_active:
            return

        self._executor_active = False

        # Wait for active tasks to complete (with timeout)
        if self.active_tasks:
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")

            # Give tasks 30 seconds to complete gracefully
            timeout_seconds = 30
            start_time = time.time()

            while self.active_tasks and (time.time() - start_time) < timeout_seconds:
                await asyncio.sleep(0.1)

            if self.active_tasks:
                logger.warning(f"Force stopping {len(self.active_tasks)} remaining tasks after {timeout_seconds}s timeout")

        logger.info("Background executor stopped")

    async def submit_task(self, task: ExecutionTask) -> str:
        """
        Submit a task for background execution.
        
        Args:
            task: ExecutionTask to execute
            
        Returns:
            Task ID for tracking
        """
        if not self._executor_active:
            raise RuntimeError("Background executor is not active. Call start() first.")

        self.active_tasks[task.task_id] = task

        # Start task execution in background
        asyncio.create_task(self._execute_task(task))

        logger.debug(f"Submitted background task {task.task_id}")
        return task.task_id

    async def _execute_task(self, task: ExecutionTask):
        """Execute a single task with proper lifecycle management."""
        try:
            async with self.task_limiter:
                task.status = "running"
                task.started_at = time.time()

                logger.debug(f"Starting execution of task {task.task_id}")

                # Set Rails context if available
                if task.rails_instance:
                    from .core import set_current_rails
                    set_current_rails(task.rails_instance)

                try:
                    # Execute the workflow
                    if asyncio.iscoroutinefunction(task.workflow):
                        result = await task.workflow(*task.args, **task.kwargs)
                    else:
                        result = task.workflow(*task.args, **task.kwargs)

                    task.result = result
                    task.status = "completed"
                    task.completed_at = time.time()

                    logger.debug(f"Completed task {task.task_id} in {task.completed_at - task.started_at:.2f}s")

                except Exception as e:
                    task.error = e
                    task.status = "failed"
                    task.completed_at = time.time()

                    logger.error(f"Task {task.task_id} failed: {e}")

                finally:
                    # Clean up Rails context
                    if task.rails_instance:
                        from .core import set_current_rails
                        set_current_rails(None)

        finally:
            # Move task from active to completed
            if task.task_id in self.active_tasks:
                self.active_tasks.pop(task.task_id)
                self.completed_tasks[task.task_id] = task

                # Clean up old completed tasks (keep last 100)
                if len(self.completed_tasks) > 100:
                    oldest_tasks = sorted(
                        self.completed_tasks.items(),
                        key=lambda x: x[1].completed_at or 0
                    )
                    for task_id, _ in oldest_tasks[:-100]:
                        self.completed_tasks.pop(task_id, None)

    async def get_task_status(self, task_id: str) -> ExecutionTask | None:
        """Get status of a specific task."""
        # Check active tasks first
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]

        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]

        return None

    async def wait_for_task(self, task_id: str, timeout: float = 60.0) -> tuple[bool, Any, Exception | None]:
        """
        Wait for a task to complete.
        
        Args:
            task_id: Task ID to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Tuple of (success, result, error)
        """
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            task = await self.get_task_status(task_id)

            if task is None:
                return False, None, Exception(f"Task {task_id} not found")

            if task.status == "completed":
                return True, task.result, None
            elif task.status == "failed":
                return False, None, task.error

            # Still running, wait a bit
            await asyncio.sleep(0.1)

        return False, None, Exception(f"Task {task_id} timeout after {timeout}s")

    def get_executor_status(self) -> dict[str, Any]:
        """Get comprehensive executor status."""
        active_task_info = {
            task_id: {
                "status": task.status,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "age_seconds": time.time() - task.created_at,
                "running_seconds": (time.time() - task.started_at) if task.started_at else 0,
            }
            for task_id, task in self.active_tasks.items()
        }

        return {
            "executor_active": self._executor_active,
            "max_concurrent": self.max_concurrent,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "active_task_details": active_task_info,
        }


# Global background executor instance
_background_executor: BackgroundExecutor | None = None


def get_background_executor() -> BackgroundExecutor:
    """Get the global background executor instance."""
    global _background_executor
    if _background_executor is None:
        _background_executor = BackgroundExecutor()
    return _background_executor


async def execute_background_workflow(workflow: Callable, *args, rails_instance=None,
                                    task_id: str | None = None, **kwargs) -> str:
    """
    Convenience function to execute a workflow in the background.
    
    Args:
        workflow: Workflow function to execute
        *args: Arguments to pass to workflow
        rails_instance: Rails instance for context (optional)
        task_id: Custom task ID (auto-generated if None)
        **kwargs: Keyword arguments to pass to workflow
        
    Returns:
        Task ID for tracking
    """
    executor = get_background_executor()

    # Ensure executor is started
    if not executor._executor_active:
        await executor.start()

    # Generate task ID if not provided
    if task_id is None:
        task_id = f"bg_{str(uuid.uuid4())[:8]}"

    # Create task
    task = ExecutionTask(
        task_id=task_id,
        workflow=workflow,
        args=args,
        kwargs=kwargs,
        rails_instance=rails_instance
    )

    return await executor.submit_task(task)


@asynccontextmanager
async def background_execution_context(max_concurrent: int = 10):
    """
    Context manager for background execution.
    
    This ensures proper startup and cleanup of the background executor.
    
    Args:
        max_concurrent: Maximum number of concurrent tasks
        
    Usage:
        async with background_execution_context() as executor:
            task_id = await execute_background_workflow(my_workflow)
            result = await executor.wait_for_task(task_id)
    """
    executor = BackgroundExecutor(max_concurrent)

    try:
        await executor.start()
        yield executor
    finally:
        await executor.stop()


class WorkflowOrchestrator:
    """
    High-level orchestrator for complex workflow patterns.
    
    This class provides patterns for common workflow orchestration needs,
    like conditional pipelines, parallel execution, and error recovery.
    """

    def __init__(self, rails_instance, max_concurrent: int = 5):
        self.rails = rails_instance
        self.executor = BackgroundExecutor(max_concurrent)
        self._active = False

    async def start(self):
        """Start the orchestrator."""
        if not self._active:
            await self.executor.start()
            self._active = True

    async def stop(self):
        """Stop the orchestrator."""
        if self._active:
            await self.executor.stop()
            self._active = False

    async def execute_conditional_pipeline(self, conditions_and_workflows: list[tuple[Callable, Callable]]) -> dict[str, Any]:
        """
        Execute workflows conditionally in sequence.
        
        Args:
            conditions_and_workflows: List of (condition_func, workflow_func) tuples
            
        Returns:
            Results dictionary with task IDs and outcomes
        """
        results = {}

        for i, (condition, workflow) in enumerate(conditions_and_workflows):
            try:
                # Check condition
                if condition(self.rails.store):
                    task_id = f"pipeline_{i}_{uuid.uuid4().hex[:6]}"

                    task = ExecutionTask(
                        task_id=task_id,
                        workflow=workflow,
                        rails_instance=self.rails
                    )

                    await self.executor.submit_task(task)
                    success, result, error = await self.executor.wait_for_task(task_id)

                    results[task_id] = {
                        "success": success,
                        "result": result,
                        "error": error,
                        "step": i
                    }

                    # If this step failed and it's critical, stop pipeline
                    if not success:
                        logger.error(f"Pipeline step {i} failed: {error}")
                        break

                else:
                    results[f"skipped_{i}"] = {"skipped": True, "condition_met": False}

            except Exception as e:
                results[f"error_{i}"] = {"error": str(e), "step": i}
                break

        return results

    async def execute_parallel_workflows(self, workflows: list[Callable], wait_all: bool = True) -> dict[str, Any]:
        """
        Execute multiple workflows in parallel.
        
        Args:
            workflows: List of workflow functions
            wait_all: Whether to wait for all workflows to complete
            
        Returns:
            Results dictionary with task IDs and outcomes
        """
        if not self._active:
            await self.start()

        # Submit all tasks
        task_ids = []
        for i, workflow in enumerate(workflows):
            task_id = f"parallel_{i}_{uuid.uuid4().hex[:6]}"

            task = ExecutionTask(
                task_id=task_id,
                workflow=workflow,
                rails_instance=self.rails
            )

            await self.executor.submit_task(task)
            task_ids.append(task_id)

        # Wait for results
        results = {}
        for task_id in task_ids:
            if wait_all:
                success, result, error = await self.executor.wait_for_task(task_id)
                results[task_id] = {"success": success, "result": result, "error": error}
            else:
                # Just get current status without waiting
                task_status = await self.executor.get_task_status(task_id)
                results[task_id] = {"status": task_status.status if task_status else "unknown"}

        return results

    @asynccontextmanager
    async def orchestration_context(self):
        """Context manager for orchestrator lifecycle."""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()
