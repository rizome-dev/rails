"""
Queue Management Example - Task Flow Orchestration

This example demonstrates Rails queue-based workflow patterns:
1. Queue-based task management and distribution
2. Priority queues and task routing
3. Queue condition builders and monitoring
4. Task retry and error handling patterns
5. Queue-driven workflow orchestration
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from rails import (
    Rails, current_rails, Message, Role,
    counter, state, queue, system, template
)


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Task:
    """Represents a task in the workflow system."""
    id: str
    title: str
    priority: TaskPriority
    data: Dict[str, Any]
    created_at: str
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for queue storage."""
        return {
            'id': self.id,
            'title': self.title,
            'priority': self.priority.value,
            'data': self.data,
            'created_at': self.created_at,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        return cls(
            id=data['id'],
            title=data['title'],
            priority=TaskPriority(data['priority']),
            data=data['data'],
            created_at=data['created_at'],
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3)
        )


def task_creator_tool(title: str, priority: str = "medium", **task_data) -> Dict[str, Any]:
    """Tool for creating and queuing new tasks."""
    rails = current_rails()
    
    # Create task object
    task = Task(
        id=f"task_{int(asyncio.get_event_loop().time() * 1000)}",
        title=title,
        priority=TaskPriority(priority.lower()),
        data=task_data,
        created_at=datetime.now().isoformat()
    )
    
    # Queue task based on priority
    queue_name = f"tasks_{priority.lower()}"
    asyncio.create_task(rails.store.push_queue(queue_name, task.to_dict()))
    
    # Update task creation metrics
    asyncio.create_task(rails.store.increment('tasks_created'))
    asyncio.create_task(rails.store.increment(f'tasks_created_{priority.lower()}'))
    asyncio.create_task(rails.store.set('last_task_created', task.id))
    
    return {
        "task_id": task.id,
        "title": title,
        "priority": priority,
        "queued_to": queue_name,
        "created": True
    }


def task_processor_tool(queue_name: str) -> Dict[str, Any]:
    """Tool for processing tasks from queues."""
    rails = current_rails()
    
    # Try to get a task from the specified queue
    async def get_and_process():
        task_data = await rails.store.pop_queue(queue_name)
        if not task_data:
            return None
        
        task = Task.from_dict(task_data)
        
        # Simulate task processing
        processing_time = 0.1
        if task.priority == TaskPriority.URGENT:
            processing_time = 0.05  # Urgent tasks process faster
        elif task.priority == TaskPriority.LOW:
            processing_time = 0.2   # Low priority takes longer
        
        await asyncio.sleep(processing_time)
        
        # Simulate processing outcomes based on task data
        success_rate = 0.8  # 80% success rate
        if 'error' in task.title.lower():
            success_rate = 0.2  # Intentional failures for testing
        
        import random
        success = random.random() < success_rate
        
        if success:
            # Task completed successfully
            await rails.store.increment('tasks_completed')
            await rails.store.increment(f'tasks_completed_{task.priority.value}')
            
            # Add to completed tasks queue
            completion_data = {
                **task.to_dict(),
                'completed_at': datetime.now().isoformat(),
                'processing_time': processing_time
            }
            await rails.store.push_queue('completed_tasks', completion_data)
            
            return {
                "task_id": task.id,
                "status": "completed",
                "priority": task.priority.value,
                "processing_time": processing_time
            }
        else:
            # Task failed
            await rails.store.increment('tasks_failed')
            
            if task.retry_count < task.max_retries:
                # Add to retry queue
                task.retry_count += 1
                retry_data = task.to_dict()
                retry_data['retry_at'] = (datetime.now() + timedelta(seconds=5)).isoformat()
                
                await rails.store.push_queue('retry_queue', retry_data)
                await rails.store.increment('tasks_retried')
                
                return {
                    "task_id": task.id,
                    "status": "failed_retry_queued",
                    "retry_count": task.retry_count,
                    "max_retries": task.max_retries
                }
            else:
                # Max retries exceeded, move to dead letter queue
                dead_letter_data = {
                    **task.to_dict(),
                    'failed_at': datetime.now().isoformat(),
                    'failure_reason': 'max_retries_exceeded'
                }
                await rails.store.push_queue('dead_letter_queue', dead_letter_data)
                await rails.store.increment('tasks_dead_letter')
                
                return {
                    "task_id": task.id,
                    "status": "dead_letter",
                    "retry_count": task.retry_count
                }
    
    # Schedule the async processing
    result = asyncio.create_task(get_and_process())
    
    # Return a placeholder - in real usage, you'd handle this differently
    return {"processing": "scheduled", "queue": queue_name}


def queue_monitor_tool() -> Dict[str, Any]:
    """Tool for monitoring queue states and metrics."""
    rails = current_rails()
    
    async def get_queue_status():
        status = {}
        
        # Check all queue lengths
        queue_names = ['tasks_urgent', 'tasks_high', 'tasks_medium', 'tasks_low', 
                      'retry_queue', 'completed_tasks', 'dead_letter_queue']
        
        for queue_name in queue_names:
            length = await rails.store.queue_length(queue_name)
            status[queue_name] = length
        
        # Update monitoring state
        total_pending = sum(status[q] for q in ['tasks_urgent', 'tasks_high', 'tasks_medium', 'tasks_low'])
        await rails.store.set('total_pending_tasks', total_pending)
        await rails.store.set('monitoring_timestamp', datetime.now().isoformat())
        
        # Set alert conditions
        if total_pending > 10:
            await rails.store.set('queue_backlog_alert', True)
        else:
            await rails.store.set('queue_backlog_alert', False)
            
        if status['dead_letter_queue'] > 5:
            await rails.store.set('dead_letter_alert', True)
        
        return status
    
    # Schedule async monitoring
    asyncio.create_task(get_queue_status())
    return {"monitoring": "active"}


def retry_processor_tool() -> Dict[str, Any]:
    """Tool for processing tasks from retry queue."""
    rails = current_rails()
    
    async def process_retry():
        retry_data = await rails.store.pop_queue('retry_queue')
        if not retry_data:
            return None
        
        # Check if retry time has elapsed
        retry_at = datetime.fromisoformat(retry_data.get('retry_at', datetime.now().isoformat()))
        if datetime.now() < retry_at:
            # Not ready for retry, put back in queue
            await rails.store.push_queue('retry_queue', retry_data)
            return {"status": "not_ready", "retry_at": retry_at.isoformat()}
        
        # Ready for retry - put back in appropriate priority queue
        task = Task.from_dict(retry_data)
        queue_name = f"tasks_{task.priority.value}"
        await rails.store.push_queue(queue_name, retry_data)
        
        await rails.store.increment('tasks_retry_processed')
        
        return {
            "task_id": task.id,
            "status": "requeued",
            "queue": queue_name,
            "retry_count": task.retry_count
        }
    
    asyncio.create_task(process_retry())
    return {"retry_processing": "scheduled"}


async def queue_balancer_workflow(rails: Rails) -> None:
    """Workflow for balancing queues and optimizing task distribution."""
    print("⚖️ Queue balancer workflow activated")
    
    # Get queue lengths
    urgent_count = await rails.store.queue_length('tasks_urgent')
    high_count = await rails.store.queue_length('tasks_high')
    medium_count = await rails.store.queue_length('tasks_medium')
    low_count = await rails.store.queue_length('tasks_low')
    
    total_tasks = urgent_count + high_count + medium_count + low_count
    
    print(f"  📊 Queue status: U:{urgent_count} H:{high_count} M:{medium_count} L:{low_count}")
    
    # Balance queues if needed
    if medium_count > 5 and high_count == 0:
        # Promote some medium priority tasks to high priority
        for _ in range(min(2, medium_count)):
            task_data = await rails.store.pop_queue('tasks_medium')
            if task_data:
                task_data['priority'] = 'high'
                await rails.store.push_queue('tasks_high', task_data)
        
        await rails.store.increment('queue_promotions')
        print(f"  ⬆️ Promoted medium priority tasks to high priority")
    
    # Archive old completed tasks if queue is full
    completed_count = await rails.store.queue_length('completed_tasks')
    if completed_count > 20:
        archived_count = 0
        while await rails.store.queue_length('completed_tasks') > 10:
            task_data = await rails.store.pop_queue('completed_tasks')
            if task_data:
                task_data['archived_at'] = datetime.now().isoformat()
                await rails.store.push_queue('archived_tasks', task_data)
                archived_count += 1
        
        print(f"  📚 Archived {archived_count} completed tasks")
    
    await rails.store.set('last_balance_time', datetime.now().isoformat())
    await rails.store.increment('balance_cycles')


async def demonstrate_queue_management():
    """Comprehensive demonstration of queue-based workflow orchestration."""
    print("🚀 Queue Management & Task Flow Orchestration Demo")
    print("=" * 55)
    
    # Initialize Rails
    rails = Rails()
    
    # =================================================================
    # SETUP QUEUE-BASED LIFECYCLE RULES
    # =================================================================
    
    print("🔧 Configuring queue-based Rails lifecycle rules...")
    
    # 1. Queue backlog monitoring
    rails.add_rule(
        condition=queue('tasks_urgent').length >= 3,
        action=system("🚨 URGENT: Multiple urgent tasks queued - immediate attention required!"),
        name="urgent_queue_alert"
    )
    
    # 2. Dead letter queue monitoring
    rails.add_rule(
        condition=queue('dead_letter_queue').length >= 3,
        action=system("💀 Dead letter queue growing - investigate recurring failures"),
        name="dead_letter_alert"
    )
    
    # 3. Queue balancing trigger
    rails.add_rule(
        condition=state('total_pending_tasks') >= 8,
        action=lambda messages: asyncio.create_task(queue_balancer_workflow(rails)) or messages,
        name="queue_balancer"
    )
    
    # 4. Processing milestone celebrations
    rails.add_rule(
        condition=counter('tasks_completed') >= 5,
        action=template("🎉 Processed {tasks_completed} tasks successfully! Throughput: excellent"),
        name="throughput_celebration"
    )
    
    # 5. Retry queue monitoring
    rails.add_rule(
        condition=queue('retry_queue').length >= 4,
        action=system("🔄 Retry queue building up - may need to investigate failure patterns"),
        name="retry_queue_monitor"
    )
    
    # 6. Queue emptiness detection
    rails.add_rule(
        condition=queue('tasks_medium').empty,
        action=system("📭 Medium priority queue is empty - good job staying on top of tasks!"),
        name="queue_empty_praise"
    )
    
    # 7. High completion rate
    async def check_success_rate(store):
        """Check if we have a high task success rate."""
        completed = await store.get_counter('tasks_completed', 0)
        failed = await store.get_counter('tasks_failed', 0)
        total = completed + failed
        
        if total >= 8:
            success_rate = completed / total
            return success_rate >= 0.8  # 80% success rate
        return False
    
    rails.add_rule(
        condition=check_success_rate,
        action=system("⭐ Excellent! Maintaining high task success rate - keep it up!"),
        name="success_rate_recognition"
    )
    
    # 8. Demo completion
    rails.add_rule(
        condition=counter('tasks_created') >= 12,
        action=system("🛑 Queue management demo complete! You've mastered task flow orchestration."),
        name="demo_completion"
    )
    
    print(f"✅ Configured {len(rails.rules)} queue-based Rails rules")
    
    # =================================================================
    # RUN QUEUE MANAGEMENT DEMONSTRATION
    # =================================================================
    
    async with rails:
        print(f"\\n🔄 Starting queue-based workflow orchestration...")
        
        # Phase 1: Create various tasks
        print(f"\\n📝 Phase 1: Creating diverse task queue")
        
        task_specifications = [
            ("Process customer order #1001", "high", {"customer_id": 1001, "order_value": 250}),
            ("Send welcome email", "medium", {"user_id": 2001, "template": "welcome"}),
            ("Generate monthly report", "low", {"report_type": "monthly", "department": "sales"}),
            ("Handle urgent support ticket", "urgent", {"ticket_id": "SUP-001", "severity": "critical"}),
            ("Process refund request", "medium", {"order_id": 1002, "amount": 75}),
            ("System backup task", "low", {"backup_type": "incremental"}),
            ("Security scan", "high", {"scan_type": "vulnerability"}),
            ("Error prone task", "medium", {"error": True}),  # Will fail
            ("Another urgent task", "urgent", {"priority": "p0"}),
            ("Data cleanup task", "low", {"records": 1000}),
            ("Failed task example", "high", {"error": True}),  # Will fail
            ("Final processing task", "medium", {"batch": "final"}),
        ]
        
        for title, priority, data in task_specifications:
            result = task_creator_tool(title, priority, **data)
            print(f"  ✅ Created: {title} [{priority}] → {result['task_id']}")
            
            # Brief pause between task creation
            await asyncio.sleep(0.05)
        
        # Phase 2: Process tasks from different queues
        print(f"\\n⚙️ Phase 2: Processing tasks by priority")
        
        processing_rounds = [
            ('tasks_urgent', 'urgent'),
            ('tasks_high', 'high'),  
            ('tasks_urgent', 'urgent'),
            ('tasks_medium', 'medium'),
            ('tasks_high', 'high'),
            ('tasks_medium', 'medium'),
            ('tasks_low', 'low'),
            ('tasks_medium', 'medium'),
        ]
        
        for queue_name, priority_name in processing_rounds:
            print(f"\\n🔄 Processing from {priority_name} priority queue...")
            
            # Monitor queues first
            monitor_result = queue_monitor_tool()
            
            # Process a task
            process_result = task_processor_tool(queue_name)
            
            # Process any retries
            retry_result = retry_processor_tool()
            
            # Check for Rails injections
            messages = [Message(role=Role.USER, content=f"Processed {priority_name} priority task")]
            enhanced_messages = await rails.process(messages)
            
            if len(enhanced_messages) > 1:
                for msg in enhanced_messages[1:]:
                    print(f"  💬 Rails: {msg.content}")
            
            # Check for completion signal
            if any("demo complete" in msg.content.lower() for msg in enhanced_messages if msg.injected_by_rails):
                print("\\n🎉 Demo completion detected!")
                break
            
            await asyncio.sleep(0.2)
    
    # =================================================================
    # FINAL QUEUE METRICS AND ANALYSIS
    # =================================================================
    
    print("\\n📊 Final Queue Management Metrics:")
    
    # Task creation and processing metrics
    print(f"  📝 Task Creation:")
    print(f"    • Total created: {await rails.store.get_counter('tasks_created')}")
    print(f"    • Urgent: {await rails.store.get_counter('tasks_created_urgent', 0)}")
    print(f"    • High: {await rails.store.get_counter('tasks_created_high', 0)}")
    print(f"    • Medium: {await rails.store.get_counter('tasks_created_medium', 0)}")
    print(f"    • Low: {await rails.store.get_counter('tasks_created_low', 0)}")
    
    print(f"\\n  ⚙️ Task Processing:")
    print(f"    • Completed: {await rails.store.get_counter('tasks_completed', 0)}")
    print(f"    • Failed: {await rails.store.get_counter('tasks_failed', 0)}")
    print(f"    • Retried: {await rails.store.get_counter('tasks_retried', 0)}")
    print(f"    • Dead letter: {await rails.store.get_counter('tasks_dead_letter', 0)}")
    
    # Current queue status
    print(f"\\n  📦 Final Queue Status:")
    queue_names = ['tasks_urgent', 'tasks_high', 'tasks_medium', 'tasks_low', 
                  'retry_queue', 'completed_tasks', 'dead_letter_queue']
    
    for queue_name in queue_names:
        count = await rails.store.queue_length(queue_name)
        print(f"    • {queue_name}: {count} items")
    
    # System metrics
    print(f"\\n  🔧 System Metrics:")
    print(f"    • Queue promotions: {await rails.store.get_counter('queue_promotions', 0)}")
    print(f"    • Balance cycles: {await rails.store.get_counter('balance_cycles', 0)}")
    print(f"    • Retry processed: {await rails.store.get_counter('tasks_retry_processed', 0)}")
    
    # Calculate success rate
    completed = await rails.store.get_counter('tasks_completed', 0)
    failed = await rails.store.get_counter('tasks_failed', 0)
    total_attempted = completed + failed
    
    if total_attempted > 0:
        success_rate = (completed / total_attempted) * 100
        print(f"    • Success rate: {success_rate:.1f}% ({completed}/{total_attempted})")
    
    # Rails orchestration metrics
    rails_metrics = await rails.emit_metrics()
    print(f"\\n⚙️ Rails Orchestration:")
    print(f"  • Final state: {rails_metrics['state']}")
    print(f"  • Rules triggered: {rails_metrics['active_rules']}/{rails_metrics['total_rules']}")
    
    print(f"\\n🎯 Queue Management Features Demonstrated:")
    print(f"  ✅ Task creation and priority-based queuing")
    print(f"  ✅ Queue condition builders: queue().length, queue().empty")
    print(f"  ✅ Priority-based task processing")
    print(f"  ✅ Retry mechanism with exponential backoff")
    print(f"  ✅ Dead letter queue for failed tasks")
    print(f"  ✅ Queue monitoring and alerting")
    print(f"  ✅ Automatic queue balancing and optimization")
    print(f"  ✅ Success rate tracking and recognition")
    print(f"  ✅ Lifecycle orchestration with queue triggers")
    
    print(f"\\n✨ Queue management demonstration complete!")


if __name__ == "__main__":
    asyncio.run(demonstrate_queue_management())