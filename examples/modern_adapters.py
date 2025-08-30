"""
Modern Rails adapter example using new lifecycle orchestration patterns.

This example demonstrates the new Rails architecture:
1. Fluent condition builders: counter(), state(), queue()
2. Rails.add_rule() for lifecycle orchestration
3. current_rails() for tool access
4. Pydantic v2 Message/Role models
5. Async store operations
6. Event streaming patterns
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from rails import (
    Rails, current_rails, Message, Role, 
    counter, state, queue, system, template
)
from rails.adapters import BaseAdapter, create_adapter


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
    """Tool for processing LLM objects - demonstrates current_rails() usage."""
    
    def __init__(self):
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
        Main processing function using current_rails() pattern.
        
        This demonstrates how tools access Rails store for lifecycle orchestration.
        """
        self.logger.info(f"Processing LLM object from {obj.source}")
        
        # Access Rails instance from context - key new pattern!
        rails = current_rails()
        
        # 1. Sanitize the object
        sanitized_obj = self.sanitize_object(obj)
        
        # 2. Validate the object
        is_valid = self.validate_object(sanitized_obj)
        sanitized_obj.validated = is_valid
        
        # 3. Increment processing counter using async methods
        processing_count = await rails.store.increment('llm_objects_processed')
        
        # 4. Update validation counters based on result
        if is_valid:
            await rails.store.increment('valid_objects')
        else:
            await rails.store.increment('invalid_objects')
        
        # 5. Set current object state for Rails conditions to use
        await rails.store.set('last_object_source', obj.source)
        await rails.store.set('last_object_valid', is_valid)
        await rails.store.set('last_processed_time', datetime.now().isoformat())
        
        # 6. Queue management - demonstrate queue operations
        await rails.store.push_queue('processed_objects', {
            'id': f"obj_{processing_count}",
            'source': obj.source,
            'valid': is_valid,
            'timestamp': datetime.now().isoformat()
        })
        
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


class ModernRailsAdapter(BaseAdapter):
    """Modern Rails adapter using new architecture patterns."""
    
    def __init__(self, processor: LLMObjectProcessor):
        super().__init__()
        self.processor = processor
        self.logger = logging.getLogger(__name__)
    
    async def process_messages(self, messages: List[Message], **kwargs) -> List[Message]:
        """Process messages with Rails lifecycle orchestration."""
        results = []
        
        # Process each message as an LLM object
        for i, message in enumerate(messages):
            # Create LLM object from message
            llm_obj = LLMObject(
                content=message.content,
                confidence=0.8 + (i * 0.05),  # Varying confidence
                metadata={
                    'timestamp': asyncio.get_event_loop().time(),
                    'model': 'gpt-4',
                    'message_index': i,
                    'original_role': message.role.value
                },
                source=f"message_{i}"
            )
            
            # Process through our tool
            result = await self.processor.process_llm_object(llm_obj)
            results.append(result)
        
        # Create assistant response with processing summary
        valid_count = sum(1 for r in results if r['valid'])
        invalid_count = len(results) - valid_count
        
        response_content = (
            f"Processed {len(results)} messages. "
            f"✅ {valid_count} valid, ❌ {invalid_count} invalid."
        )
        
        response_msg = Message(
            role=Role.ASSISTANT,
            content=response_content,
            metadata={'processing_results': results}
        )
        
        # Return original messages plus response
        result_messages = messages.copy()
        result_messages.append(response_msg)
        return result_messages
    
    async def post_process(self, original_messages: List[Message], result_messages: List[Message]) -> None:
        """Post-processing hook for additional state updates."""
        rails = current_rails()
        
        # Increment batch counter
        await rails.store.increment('message_batches_processed')
        
        # Check for any contingency conditions
        total_processed = await rails.store.get_counter('llm_objects_processed', 0)
        queue_length = await rails.store.queue_length('processed_objects')
        
        # Set state that other rules can use
        if total_processed > 0 and total_processed % 5 == 0:
            await rails.store.set('milestone_reached', total_processed)
        
        if queue_length > 10:
            await rails.store.set('queue_backlog', True)


async def demonstrate_modern_rails():
    """
    Comprehensive demonstration of modern Rails patterns.
    """
    print("🚀 Modern Rails Lifecycle Orchestration Demo")
    print("=" * 55)
    
    # Initialize Rails
    rails = Rails()
    
    # =================================================================
    # SETUP MODERN RULES USING FLUENT API
    # =================================================================
    
    print("🔧 Setting up Rails lifecycle rules...")
    
    # 1. Stop condition using fluent counter builder
    rails.add_rule(
        condition=counter('llm_objects_processed') >= 10,
        action=system("🛑 STOP: Processed 10 objects, workflow complete!"),
        name="completion_trigger"
    )
    
    # 2. Quality monitoring using counter conditions
    rails.add_rule(
        condition=counter('invalid_objects') >= 3,
        action=system("⚠️ QUALITY ALERT: Multiple validation failures detected!"),
        name="quality_monitor"
    )
    
    # 3. State-based guidance using fluent state builder
    rails.add_rule(
        condition=state('queue_backlog') == True,
        action=system("📊 Queue backlog detected - consider scaling processing"),
        name="backlog_alert"
    )
    
    # 4. Milestone celebration using template with store values
    rails.add_rule(
        condition=state('milestone_reached').exists,
        action=template("🎉 Milestone reached: {milestone_reached} objects processed!"),
        name="milestone_celebration"
    )
    
    # 5. Queue-based condition using fluent queue builder
    rails.add_rule(
        condition=queue('processed_objects').length >= 5,
        action=system("📦 Queue contains 5+ items - good throughput!"),
        name="throughput_tracker"
    )
    
    # 6. Complex condition using lambda for error rate analysis
    async def check_error_rate(store):
        """Custom condition for error rate monitoring."""
        invalid = await store.get_counter('invalid_objects', 0)
        total = await store.get_counter('llm_objects_processed', 0)
        if total > 3 and invalid > 0:
            rate = invalid / total
            return rate > 0.4  # 40% error rate threshold
        return False
    
    rails.add_rule(
        condition=check_error_rate,
        action=system("🔥 HIGH ERROR RATE: Review data quality and processing logic!"),
        name="error_rate_monitor"
    )
    
    print(f"✅ Configured {len(rails.rules)} Rails orchestration rules")
    
    # =================================================================
    # RUN DEMONSTRATION WITH LIFECYCLE CONTEXT
    # =================================================================
    
    # Initialize processor and adapter
    processor = LLMObjectProcessor()
    adapter = ModernRailsAdapter(processor)
    
    async with rails:  # Context manager sets current_rails()
        print(f"\\n🔄 Starting lifecycle orchestration...")
        
        # Create test messages with varying quality
        test_messages = [
            Message(role=Role.USER, content="High quality message with good content length", 
                   metadata={'test_case': 'valid'}),
            Message(role=Role.USER, content="Another good message for processing validation",
                   metadata={'test_case': 'valid'}),
            Message(role=Role.USER, content="bad",  # Low quality - will fail validation
                   metadata={'test_case': 'invalid'}),
            Message(role=Role.USER, content="Quality message number four with sufficient content",
                   metadata={'test_case': 'valid'}),
            Message(role=Role.USER, content="x",  # Another invalid one
                   metadata={'test_case': 'invalid'}),
            Message(role=Role.USER, content="This is message six with proper validation content",
                   metadata={'test_case': 'valid'}),
            Message(role=Role.USER, content="Seventh message demonstrates Rails lifecycle orchestration",
                   metadata={'test_case': 'valid'}),
            Message(role=Role.USER, content="bad content",  # Another invalid
                   metadata={'test_case': 'invalid'}),
            Message(role=Role.USER, content="Ninth message shows modern Rails patterns in action",
                   metadata={'test_case': 'valid'}),
            Message(role=Role.USER, content="Final message completes our comprehensive demonstration",
                   metadata={'test_case': 'valid'}),
        ]
        
        # Process in batches to see Rails rules trigger
        batch_size = 3
        all_messages = []
        
        for batch_num in range(0, len(test_messages), batch_size):
            batch = test_messages[batch_num:batch_num + batch_size]
            print(f"\\n📦 Processing batch {batch_num//batch_size + 1}: {len(batch)} messages")
            
            # First pass through Rails lifecycle orchestration
            enhanced_messages = await rails.process(batch)
            
            # Then through our adapter
            processed_messages = await adapter.process_messages(enhanced_messages)
            
            # Run post-processing
            await adapter.post_process(enhanced_messages, processed_messages)
            
            # Check for new Rails injections after processing
            final_messages = await rails.process(processed_messages)
            
            # Show any Rails injections
            if len(final_messages) > len(processed_messages):
                injected = final_messages[len(processed_messages):]
                for msg in injected:
                    print(f"  💬 Rails: {msg.content}")
            
            all_messages.extend(final_messages)
            
            # Check for stop condition
            if any("STOP:" in msg.content for msg in final_messages if msg.injected_by_rails):
                print("\\n🛑 Stop condition triggered!")
                break
            
            # Brief pause between batches
            await asyncio.sleep(0.1)
    
    # =================================================================
    # FINAL METRICS AND SUMMARY
    # =================================================================
    
    print("\\n📊 Final Rails Orchestration Metrics:")
    print(f"  • Total objects processed: {await rails.store.get_counter('llm_objects_processed')}")
    print(f"  • Valid objects: {await rails.store.get_counter('valid_objects', 0)}")
    print(f"  • Invalid objects: {await rails.store.get_counter('invalid_objects', 0)}")
    print(f"  • Batches processed: {await rails.store.get_counter('message_batches_processed', 0)}")
    print(f"  • Queue length: {await rails.store.queue_length('processed_objects')}")
    
    # Show Rails metrics
    metrics = await rails.emit_metrics()
    print(f"  • Rails state: {metrics['state']}")
    print(f"  • Active rules: {metrics['active_rules']}/{metrics['total_rules']}")
    
    print(f"\\n🎯 Modern Rails Features Demonstrated:")
    print(f"  ✅ Fluent condition builders: counter(), state(), queue()")
    print(f"  ✅ Rails.add_rule() lifecycle orchestration")
    print(f"  ✅ current_rails() tool context access")
    print(f"  ✅ Pydantic v2 Message/Role models")
    print(f"  ✅ Async store operations with events")
    print(f"  ✅ Queue-based task management")
    print(f"  ✅ Template injection with store values")
    print(f"  ✅ Complex custom condition functions")
    
    print(f"\\n✨ Modern Rails demonstration complete!")


async def demonstrate_simple_adapters():
    """Show simple adapter patterns with modern Rails."""
    print(f"\\n🔧 Simple Modern Adapter Examples")
    print(f"=" * 40)
    
    rails = Rails()
    
    # Simple help injection rule
    rails.add_rule(
        condition=counter('turns') >= 3,
        action=system("Need help? I'm here to assist you!"),
        name="help_prompt"
    )
    
    # Simple function adapter
    def simple_processor(messages):
        """Simple processing function."""
        return f"Processed {len(messages)} messages in simple mode"
    
    simple_adapter = create_adapter(rails, simple_processor)
    
    print(f"🔧 Simple Adapter Test:")
    async with rails:
        # Trigger help condition
        await rails.store.increment('turns', 3)
        
        test_messages = [Message(role=Role.USER, content="Hello Rails!")]
        
        # Process through Rails first
        enhanced_messages = await rails.process(test_messages)
        
        # Then through adapter
        final_messages = await simple_adapter.process_messages(enhanced_messages)
        
        print(f"  Input: {len(test_messages)} messages")
        print(f"  Enhanced: {len(enhanced_messages)} messages")
        print(f"  Final: {len(final_messages)} messages")
        
        # Show any Rails injections
        if len(enhanced_messages) > len(test_messages):
            injected = enhanced_messages[len(test_messages):]
            for msg in injected:
                print(f"  💬 Rails injected: {msg.content}")
    
    print(f"\\n✅ Simple adapter patterns demonstrated!")


if __name__ == "__main__":
    async def main():
        await demonstrate_modern_rails()
        await demonstrate_simple_adapters()
    
    asyncio.run(main())