"""
Basic Rails Usage Example - Core Patterns

This example demonstrates the fundamental Rails patterns:
1. Basic setup and lifecycle management
2. Fluent condition builders
3. Simple tools using current_rails()
4. Message injection patterns
5. Essential store operations
"""

import asyncio
from typing import Dict, Any

from rails import (
    Rails, current_rails, Message, Role,
    counter, state, system, template
)


def simple_counter_tool(operation: str = "increment") -> Dict[str, Any]:
    """
    Basic tool demonstrating current_rails() usage.
    
    This is the simplest pattern for tools to access Rails store.
    """
    # Get Rails instance from context
    rails = current_rails()
    
    # Perform operations based on input
    if operation == "increment":
        # Schedule increment operation (async)
        asyncio.create_task(rails.store.increment('tool_calls'))
        return {"action": "incremented", "counter": "tool_calls"}
    
    elif operation == "set_status":
        # Schedule state update
        asyncio.create_task(rails.store.set('last_tool_used', 'simple_counter_tool'))
        return {"action": "status_set", "tool": "simple_counter_tool"}
    
    else:
        return {"action": "unknown", "operation": operation}


def data_processor_tool(data: str) -> Dict[str, Any]:
    """Tool that processes data and updates Rails state accordingly."""
    rails = current_rails()
    
    # Count the processing attempt
    asyncio.create_task(rails.store.increment('data_processed'))
    
    # Simple processing logic
    word_count = len(data.split())
    char_count = len(data)
    
    # Set processing results in Rails store
    asyncio.create_task(rails.store.set('last_word_count', word_count))
    asyncio.create_task(rails.store.set('last_char_count', char_count))
    
    # Determine data quality
    if word_count >= 5 and char_count >= 20:
        asyncio.create_task(rails.store.increment('high_quality_data'))
        asyncio.create_task(rails.store.set('last_quality', 'high'))
        quality = "high"
    elif word_count >= 2:
        asyncio.create_task(rails.store.increment('medium_quality_data'))
        asyncio.create_task(rails.store.set('last_quality', 'medium'))
        quality = "medium"
    else:
        asyncio.create_task(rails.store.increment('low_quality_data'))
        asyncio.create_task(rails.store.set('last_quality', 'low'))
        quality = "low"
    
    return {
        "processed": True,
        "word_count": word_count,
        "char_count": char_count,
        "quality": quality
    }


async def basic_rails_demo():
    """Demonstrate basic Rails usage patterns."""
    print("🚀 Basic Rails Usage Demo")
    print("=" * 30)
    
    # 1. Initialize Rails
    rails = Rails()
    
    # =================================================================
    # 2. SETUP BASIC RULES USING FLUENT BUILDERS
    # =================================================================
    
    print("🔧 Setting up basic Rails rules...")
    
    # Simple counter-based rule
    rails.add_rule(
        condition=counter('tool_calls') >= 3,
        action=system("I notice you've used tools 3 times. Great job exploring!"),
        name="tool_usage_milestone"
    )
    
    # State-based rule using fluent builder
    rails.add_rule(
        condition=state('last_quality') == 'high',
        action=system("Excellent! You provided high-quality data."),
        name="quality_praise"
    )
    
    # Another counter rule for data processing
    rails.add_rule(
        condition=counter('data_processed') >= 2,
        action=template("You've processed {data_processed} pieces of data so far."),
        name="processing_update"
    )
    
    # Low quality data guidance
    rails.add_rule(
        condition=state('last_quality') == 'low',
        action=system("Try providing more detailed information for better results."),
        name="quality_guidance"
    )
    
    # Stop condition
    rails.add_rule(
        condition=counter('data_processed') >= 4,
        action=system("Demo complete! You've mastered basic Rails patterns."),
        name="completion"
    )
    
    print(f"✅ Configured {len(rails.rules)} Rails rules")
    
    # =================================================================
    # 3. RUN BASIC DEMONSTRATION
    # =================================================================
    
    async with rails:  # This sets the context for current_rails()
        print("\\n🔄 Starting basic Rails demonstration...")
        
        # Test data of varying quality
        test_inputs = [
            "Hello",  # Low quality (1 word, short)
            "This is better data",  # Medium quality  
            "This is high quality data with sufficient detail for processing",  # High quality
            "Final test"  # Medium quality
        ]
        
        for i, data in enumerate(test_inputs):
            print(f"\\n📝 Step {i + 1}: Processing '{data[:30]}{'...' if len(data) > 30 else ''}'")
            
            # Use our simple tools
            counter_result = simple_counter_tool("increment")
            status_result = simple_counter_tool("set_status")
            processing_result = data_processor_tool(data)
            
            print(f"  🔧 Tool results: {processing_result['quality']} quality, {processing_result['word_count']} words")
            
            # Create initial message
            initial_messages = [Message(
                role=Role.USER,
                content=f"Please process: {data}",
                metadata={"step": i + 1}
            )]
            
            # Process through Rails lifecycle
            enhanced_messages = await rails.process(initial_messages)
            
            # Show any Rails injections
            if len(enhanced_messages) > len(initial_messages):
                injected_messages = enhanced_messages[len(initial_messages):]
                for msg in injected_messages:
                    print(f"  💬 Rails: {msg.content}")
            
            # Check for completion
            if any("Demo complete" in msg.content for msg in enhanced_messages if msg.injected_by_rails):
                print("\\n🎉 Basic demo completed successfully!")
                break
            
            # Brief pause
            await asyncio.sleep(0.1)
    
    # =================================================================
    # 4. SHOW FINAL RESULTS
    # =================================================================
    
    print("\\n📊 Final Results:")
    print(f"  • Tool calls made: {await rails.store.get_counter('tool_calls')}")
    print(f"  • Data pieces processed: {await rails.store.get_counter('data_processed')}")
    print(f"  • High quality data: {await rails.store.get_counter('high_quality_data', 0)}")
    print(f"  • Medium quality data: {await rails.store.get_counter('medium_quality_data', 0)}")
    print(f"  • Low quality data: {await rails.store.get_counter('low_quality_data', 0)}")
    
    # Show final state
    print(f"\\n🔧 Final State:")
    print(f"  • Last tool used: {await rails.store.get('last_tool_used', 'None')}")
    print(f"  • Last data quality: {await rails.store.get('last_quality', 'None')}")
    print(f"  • Last word count: {await rails.store.get('last_word_count', 0)}")
    
    print(f"\\n🎯 Basic Patterns Demonstrated:")
    print(f"  ✅ Rails() initialization and context management")
    print(f"  ✅ Fluent condition builders: counter(), state()")
    print(f"  ✅ Tools using current_rails() for store access")
    print(f"  ✅ Rails.add_rule() for lifecycle orchestration")
    print(f"  ✅ Basic message injection with system() and template()")
    print(f"  ✅ Store operations: increment, set, get")
    print(f"  ✅ Message processing through Rails lifecycle")


async def simple_conversation_demo():
    """Demonstrate Rails in a simple conversation context."""
    print("\\n💬 Simple Conversation Demo")
    print("=" * 35)
    
    rails = Rails()
    
    # Conversation-focused rules
    rails.add_rule(
        condition=counter('messages') >= 3,
        action=system("We're having a nice conversation! How can I help you further?"),
        name="conversation_engagement"
    )
    
    rails.add_rule(
        condition=state('topic') == 'rails',
        action=system("Great! Rails is a powerful lifecycle orchestration library."),
        name="rails_topic_response"
    )
    
    async with rails:
        print("🔄 Starting simple conversation...")
        
        conversation_turns = [
            ("Hello there!", None),
            ("Tell me about Rails", "rails"),
            ("That's interesting", None),
            ("Thanks for the info!", None)
        ]
        
        for i, (user_message, topic) in enumerate(conversation_turns):
            print(f"\\n👤 User: {user_message}")
            
            # Update conversation state
            await rails.store.increment('messages')
            if topic:
                await rails.store.set('topic', topic)
            
            # Process through Rails
            messages = [Message(role=Role.USER, content=user_message)]
            enhanced_messages = await rails.process(messages)
            
            # Show Rails responses
            for msg in enhanced_messages:
                if msg.injected_by_rails:
                    print(f"🤖 Rails: {msg.content}")
            
            await asyncio.sleep(0.1)
    
    print("\\n✅ Simple conversation demo complete!")


if __name__ == "__main__":
    async def main():
        await basic_rails_demo()
        await simple_conversation_demo()
    
    asyncio.run(main())