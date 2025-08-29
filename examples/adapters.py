import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum

import anyio

from rails import Rails, CounterCondition, StateCondition, LambdaCondition
from rails.adapters import BaseRailsAdapter, create_adapter


class WorkflowSignal(Enum):
    """Signals for workflow control."""
    CONTINUE = "continue"
    PAUSE = "pause" 
    STOP = "stop"
    ESCALATE = "escalate"
    QUEUE_FULL = "queue_full"


@dataclass
class WorkflowContext:
    """Context for workflow execution."""
    agent_id: str
    session_id: str
    current_phase: str
    metadata: Dict[str, Any]
    signal: WorkflowSignal = WorkflowSignal.CONTINUE


@dataclass 
class LLMObject:
    """Represents an object from an LLM response."""
    content: str
    confidence: float
    metadata: Dict[str, Any]
    source: str
    validated: bool = False
    sanitized: bool = False


class LLMObjectProcessor:
    """Tool for processing, sanitizing and validating LLM objects."""
    
    def __init__(self, rails: Rails):
        self.rails = rails
        self.logger = logging.getLogger(__name__)
    
    def sanitize_object(self, obj: LLMObject) -> LLMObject:
        """Sanitize LLM object content."""
        # Remove potentially harmful content
        sanitized_content = obj.content.strip()
        
        # Remove script tags, SQL injection patterns, etc.
        dangerous_patterns = ['<script', 'DROP TABLE', 'DELETE FROM', 'INSERT INTO']
        for pattern in dangerous_patterns:
            if pattern.lower() in sanitized_content.lower():
                sanitized_content = sanitized_content.replace(pattern, '[SANITIZED]')
        
        # Limit length
        if len(sanitized_content) > 1000:
            sanitized_content = sanitized_content[:1000] + "..."
        
        obj.content = sanitized_content
        obj.sanitized = True
        return obj
    
    def validate_object(self, obj: LLMObject) -> bool:
        """Validate LLM object meets quality criteria."""
        # Check confidence threshold
        if obj.confidence < 0.7:
            return False
        
        # Check content length
        if len(obj.content.strip()) < 10:
            return False
        
        # Check for required metadata
        required_fields = ['timestamp', 'model']
        if not all(field in obj.metadata for field in required_fields):
            return False
        
        return True
    
    async def process_llm_object(self, obj: LLMObject) -> Dict[str, Any]:
        """
        Main processing function: sanitize, validate, increment counter.
        
        Returns processing result and triggers Rails conditions.
        """
        self.logger.info(f"Processing LLM object from {obj.source}")
        
        # 1. Sanitize the object
        sanitized_obj = self.sanitize_object(obj)
        
        # 2. Validate the object
        is_valid = self.validate_object(sanitized_obj)
        sanitized_obj.validated = is_valid
        
        # 3. Increment processing counter
        processing_count = self.rails.store.increment_sync('llm_objects_processed')
        
        # 4. Update validation counters
        if is_valid:
            valid_count = self.rails.store.increment_sync('valid_objects')
        else:
            invalid_count = self.rails.store.increment_sync('invalid_objects')
        
        # 5. Set current object state
        self.rails.store.set_sync('last_object_source', obj.source)
        self.rails.store.set_sync('last_object_valid', is_valid)
        
        result = {
            'object_id': f"obj_{processing_count}",
            'processed': True,
            'valid': is_valid,
            'content_length': len(sanitized_obj.content),
            'total_processed': processing_count,
            'source': obj.source
        }
        
        self.logger.info(f"Processed object {result['object_id']}: valid={is_valid}, total={processing_count}")
        
        return result


@asynccontextmanager
async def workflow_lifecycle(
    context: WorkflowContext,
    rails: Rails,
    with_queue_monitoring: bool = True,
    with_stop_detection: bool = True
):
    """
    Lifecycle context manager for workflow orchestration.
    
    Based on the pattern from reference/workflows/lifecycle.py
    """
    logger = logging.getLogger(__name__)
    
    # Setup phase
    logger.info(f"Starting workflow lifecycle for agent {context.agent_id}")
    
    # Initialize monitoring tasks
    monitoring_tasks = []
    
    if with_queue_monitoring:
        # Start queue monitoring task
        queue_monitor = await anyio.to_thread.run_sync(
            _start_queue_monitoring, rails, context
        )
        monitoring_tasks.append(queue_monitor)
        logger.info("Queue monitoring started")
    
    if with_stop_detection:
        # Start stop signal detection
        stop_monitor = await anyio.to_thread.run_sync(
            _start_stop_detection, rails, context
        )
        monitoring_tasks.append(stop_monitor)
        logger.info("Stop signal detection started")
    
    try:
        # Set workflow state
        rails.store.set_sync('workflow_active', True)
        rails.store.set_sync('agent_id', context.agent_id)
        rails.store.set_sync('session_id', context.session_id)
        
        # Yield control to workflow execution
        yield context
        
    finally:
        # Cleanup phase
        logger.info("Cleaning up workflow lifecycle")
        
        # Cancel monitoring tasks
        for task in monitoring_tasks:
            if hasattr(task, 'cancel'):
                task.cancel()
        
        # Set cleanup state
        rails.store.set_sync('workflow_active', False)
        rails.store.set_sync('cleanup_completed', True)
        
        logger.info("Workflow lifecycle cleanup completed")


def _start_queue_monitoring(rails: Rails, context: WorkflowContext):
    """Start background queue monitoring."""
    # This would typically return a task handle
    # For this example, we'll simulate it
    class MockTask:
        def cancel(self): pass
    
    # Set up queue monitoring state
    rails.store.set_sync('queue_monitor_active', True)
    return MockTask()


def _start_stop_detection(rails: Rails, context: WorkflowContext):  
    """Start background stop signal detection."""
    class MockTask:
        def cancel(self): pass
    
    rails.store.set_sync('stop_monitor_active', True)
    return MockTask()


async def handle_queue_contingency(
    contingency_type: str, 
    rails: Rails, 
    context: WorkflowContext
) -> WorkflowSignal:
    """
    Handle queue contingencies that trigger other workflows.
    
    Based on queue patterns from reference implementation.
    """
    logger = logging.getLogger(__name__)
    
    if contingency_type == "queue_full":
        logger.warning("Queue is full, initiating overflow workflow")
        
        # Set overflow state
        rails.store.set_sync('queue_overflow', True)
        rails.store.increment_sync('overflow_events')
        
        # Trigger overflow agent workflow
        await trigger_overflow_workflow(rails, context)
        return WorkflowSignal.PAUSE
    
    elif contingency_type == "processing_backlog":
        logger.info("Processing backlog detected, scaling up")
        
        # Increment processing workers
        workers = rails.store.get_sync('active_workers', 1)
        rails.store.set_sync('active_workers', workers + 1)
        
        # Trigger scaling workflow  
        await trigger_scaling_workflow(rails, context)
        return WorkflowSignal.CONTINUE
        
    elif contingency_type == "error_threshold":
        logger.error("Error threshold exceeded, initiating remediation")
        
        # Set error state
        rails.store.set_sync('error_mode', True)
        rails.store.increment_sync('remediation_events')
        
        # Trigger remediation agent
        await trigger_remediation_workflow(rails, context)
        return WorkflowSignal.ESCALATE
    
    return WorkflowSignal.CONTINUE


async def trigger_overflow_workflow(rails: Rails, context: WorkflowContext):
    """Trigger overflow handling agent workflow."""
    logger = logging.getLogger(__name__)
    logger.info("Triggering overflow workflow agent")
    
    # This would typically start another agent/workflow
    # For demo purposes, we'll simulate it
    rails.store.set_sync('overflow_workflow_triggered', True)
    rails.store.set_sync('overflow_trigger_time', asyncio.get_event_loop().time())


async def trigger_scaling_workflow(rails: Rails, context: WorkflowContext):
    """Trigger scaling workflow."""
    logger = logging.getLogger(__name__)
    logger.info("Triggering scaling workflow")
    
    rails.store.set_sync('scaling_workflow_triggered', True)


async def trigger_remediation_workflow(rails: Rails, context: WorkflowContext):
    """Trigger remediation agent workflow."""
    logger = logging.getLogger(__name__)
    logger.info("Triggering remediation workflow agent")
    
    rails.store.set_sync('remediation_workflow_triggered', True)


class ComprehensiveRailsAdapter(BaseRailsAdapter):
    """
    Comprehensive Rails adapter demonstrating all functionality.
    """
    
    def __init__(self, rails: Rails, processor: LLMObjectProcessor):
        super().__init__(rails)
        self.processor = processor
        self.context: Optional[WorkflowContext] = None
        self.logger = logging.getLogger(__name__)
    
    async def process_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Process messages with Rails lifecycle management."""
        # Simulate LLM processing with various objects
        results = []
        
        for i, message in enumerate(messages):
            # Create LLM object from message
            llm_obj = LLMObject(
                content=message.get('content', ''),
                confidence=0.8 + (i * 0.05),  # Varying confidence
                metadata={
                    'timestamp': asyncio.get_event_loop().time(),
                    'model': 'gpt-4',
                    'message_index': i
                },
                source=f"message_{i}"
            )
            
            # Process the LLM object
            result = await self.processor.process_llm_object(llm_obj)
            results.append(result)
        
        return {'processed_messages': results, 'total_count': len(results)}
    
    async def update_rails_state(self, messages, modified_messages, result):
        """Update Rails state after processing."""
        # Increment message batch counter
        self.rails.store.increment_sync('message_batches_processed')
        
        # Check for queue contingencies
        total_processed = self.rails.store.get_counter_sync('llm_objects_processed', 0)
        
        if total_processed > 0 and total_processed % 5 == 0:
            # Every 5 objects, check for queue contingency
            signal = await handle_queue_contingency(
                "processing_backlog", self.rails, self.context
            )
            if signal != WorkflowSignal.CONTINUE:
                self.rails.store.set_sync('workflow_signal', signal.value)


async def demonstrate_rails_adapter():
    """
    Comprehensive demonstration of all Rails functionality.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("🚀 Rails Comprehensive Adapter Example")
    print("=" * 50)
    
    # Initialize Rails with comprehensive conditions
    rails = Rails()
    
    # =================================================================
    # SETUP CONDITIONS AND INJECTIONS
    # =================================================================
    
    # 1. Counter-based stop condition (main requirement)
    rails.when(
        CounterCondition('llm_objects_processed', 10, '>=')
    ).inject({
        "role": "system", 
        "content": "🛑 STOP SIGNAL: Processed 10 objects, initiating workflow termination."
    })
    
    # 2. Quality threshold monitoring  
    rails.when(
        lambda s: s.get_counter_sync('invalid_objects', 0) >= 3
    ).inject({
        "role": "system",
        "content": "⚠️ QUALITY ALERT: Multiple invalid objects detected, review processing pipeline."
    })
    
    # 3. Queue overflow detection
    rails.when(
        lambda s: s.get_sync('queue_overflow', False)
    ).inject({
        "role": "assistant", 
        "content": "📊 Queue overflow detected, scaling workflow activated."
    }, strategy='prepend')
    
    # 4. Workflow state changes
    rails.when(
        StateCondition('workflow_signal', 'escalate')
    ).inject({
        "role": "system",
        "content": "🚨 ESCALATION: Critical errors detected, human intervention required."
    })
    
    # 5. Error threshold using complex condition
    rails.when(
        LambdaCondition(lambda s: (
            s.get_counter_sync('invalid_objects', 0) > 0 and
            s.get_counter_sync('llm_objects_processed', 0) > 5 and
            (s.get_counter_sync('invalid_objects', 0) / s.get_counter_sync('llm_objects_processed', 1)) > 0.3
        ))
    ).inject({
        "role": "system",
        "content": "📈 STATISTICS: Invalid object ratio exceeds 30%, investigating..."
    })
    
    print(f"✅ Setup {rails.rule_count()} Rails injection rules")
    
    # =================================================================
    # INITIALIZE COMPONENTS
    # =================================================================
    
    processor = LLMObjectProcessor(rails)
    adapter = ComprehensiveRailsAdapter(rails, processor)
    
    # Create workflow context
    context = WorkflowContext(
        agent_id="demo_agent_001",
        session_id="session_123",
        current_phase="processing",
        metadata={"demo": True, "version": "1.0"}
    )
    adapter.context = context
    
    # =================================================================
    # RUN LIFECYCLE DEMONSTRATION  
    # =================================================================
    
    async with workflow_lifecycle(context, rails) as ctx:
        print(f"🔄 Started workflow lifecycle for {ctx.agent_id}")
        
        # Simulate processing multiple message batches (enough to hit the 10 object limit)
        for batch_num in range(5):  # Process 5 batches to exceed 10 objects
            print(f"\n📦 Processing batch {batch_num + 1}/5")
            
            # Create sample messages with varying quality
            messages = []
            for i in range(2):  # 2 messages per batch (5 batches × 2 = 10 objects total)
                # Vary quality to trigger different conditions
                if batch_num == 2 and i >= 1:  # Make some messages low quality
                    content = "bad"  # This will be flagged as invalid
                    confidence = 0.3
                else:
                    content = f"High quality message {batch_num}-{i} with sufficient content for validation."
                    confidence = 0.85
                
                messages.append({
                    'role': 'user',
                    'content': content,
                    'confidence': confidence
                })
            
            # Process with adapter - this will handle Rails checks internally
            result = await adapter.run(messages)
            
            # Get Rails-checked messages AFTER processing to see injections
            post_process_messages = await rails.check([])  # Check for any additional injections
            
            # Show results
            processed_count = rails.store.get_counter_sync('llm_objects_processed')
            valid_count = rails.store.get_counter_sync('valid_objects', 0)
            invalid_count = rails.store.get_counter_sync('invalid_objects', 0)
            
            print(f"  📊 Processed: {processed_count}, Valid: {valid_count}, Invalid: {invalid_count}")
            
            # Check for Rails injections triggered by the processing
            if post_process_messages:
                print(f"  💉 Rails injected {len(post_process_messages)} system message(s)")
                for msg in post_process_messages:
                    print(f"    -> {msg.get('content', '')}")
                    
                # Check for stop signal
                stop_signal_detected = any(
                    'STOP SIGNAL' in msg.get('content', '') 
                    for msg in post_process_messages
                )
                
                if stop_signal_detected:
                    print(f"\n🛑 STOP SIGNAL DETECTED at batch {batch_num + 1}")
                    break
            
            # Brief pause between batches
            await asyncio.sleep(0.1)
    
    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    
    print(f"\n📈 Final Statistics:")
    print(f"  • Total objects processed: {rails.store.get_counter_sync('llm_objects_processed')}")
    print(f"  • Valid objects: {rails.store.get_counter_sync('valid_objects', 0)}")
    print(f"  • Invalid objects: {rails.store.get_counter_sync('invalid_objects', 0)}")
    print(f"  • Message batches: {rails.store.get_counter_sync('message_batches_processed', 0)}")
    print(f"  • Overflow events: {rails.store.get_counter_sync('overflow_events', 0)}")
    print(f"  • Rails rules triggered: Multiple conditions demonstrated")
    
    # Show triggered workflows
    workflows_triggered = []
    if rails.store.get_sync('overflow_workflow_triggered', False):
        workflows_triggered.append('Overflow Handler')
    if rails.store.get_sync('scaling_workflow_triggered', False):
        workflows_triggered.append('Auto Scaler')
    if rails.store.get_sync('remediation_workflow_triggered', False):
        workflows_triggered.append('Remediation Agent')
    
    if workflows_triggered:
        print(f"  • Workflows triggered: {', '.join(workflows_triggered)}")
    
    print(f"\n✅ Rails Comprehensive Demo Complete!")
    
    # Show all Rails functionality used:
    print(f"\n🎯 Rails Functionality Demonstrated:")
    print(f"  ✓ Counter-based conditions (stop at 10 objects)")
    print(f"  ✓ State-based conditions (workflow signals)")
    print(f"  ✓ Lambda conditions (complex logic)")
    print(f"  ✓ Multiple injection strategies (append, prepend)")
    print(f"  ✓ LLM object processing and sanitization")
    print(f"  ✓ Modular lifecycle functions (AnyIO patterns)")
    print(f"  ✓ Queue contingency handling")
    print(f"  ✓ Workflow orchestration and signaling") 
    print(f"  ✓ Framework adapter integration")
    print(f"  ✓ Context manager lifecycle")


async def demonstrate_framework_adapters():
    """Show Rails integration with different framework patterns."""
    print(f"\n🔧 Framework Adapter Examples")
    print(f"=" * 40)
    
    rails = Rails()
    
    # Counter-based help injection
    rails.when(lambda s: s.get_counter_sync('turns') >= 3).inject({
        "role": "assistant",
        "content": "I notice we've been chatting for a while. Is there anything specific I can help you with?"
    })
    
    # 1. Simple Function Adapter
    def simple_chat_processor(messages):
        """Simple chat processing function."""
        return {
            "role": "assistant", 
            "content": f"Processed {len(messages)} messages in simple mode"
        }
    
    simple_adapter = create_adapter(rails, simple_chat_processor)
    
    print(f"🔧 Simple Adapter Test:")
    async with simple_adapter as adapter:
        rails.store.increment_sync('turns', 3)  # Trigger injection
        messages = [{"role": "user", "content": "Hello"}]
        result = await adapter.run(messages)
        final_messages = await rails.check(messages)
        
        print(f"  Input: {len(messages)} messages")
        print(f"  Output: {len(final_messages)} messages (Rails injected: {len(final_messages) - len(messages)})")
    
    # 2. Mock LangChain Adapter  
    try:
        from rails.adapters import LangChainAdapter
        
        class MockLangChainLLM:
            async def ainvoke(self, messages):
                return {"content": f"LangChain processed {len(messages)} messages"}
        
        mock_llm = MockLangChainLLM()
        langchain_adapter = LangChainAdapter(rails, mock_llm)
        
        print(f"\n🦜 LangChain Adapter Test:")
        async with langchain_adapter as adapter:
            result = await adapter.run([{"role": "user", "content": "Test"}])
            print(f"  LangChain integration successful")
    
    except ImportError:
        print(f"\n🦜 LangChain Adapter: Not available (graceful degradation)")
    
    # 3. Mock SmolAgents Adapter
    try:
        from rails.adapters import SmolAgentsAdapter
        
        class MockSmolAgent:
            async def run(self, task, **kwargs):
                return f"SmolAgent completed task: {task}"
        
        mock_agent = MockSmolAgent()
        smola_adapter = SmolAgentsAdapter(rails, mock_agent)
        
        print(f"\n🤖 SmolAgents Adapter Test:")
        async with smola_adapter as adapter:
            result = await adapter.run("Process these messages")
            print(f"  SmolAgents integration successful")
    
    except ImportError:
        print(f"\n🤖 SmolAgents Adapter: Not available (graceful degradation)")
    
    print(f"\n✅ Framework adapters demonstrate Rails' framework-agnostic design")


if __name__ == "__main__":
    async def main():
        await demonstrate_rails_adapter()
        await demonstrate_framework_adapters()
    
    asyncio.run(main())
