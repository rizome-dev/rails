"""
Example usage of the refactored Rails library for message injection.

This demonstrates the core PRD requirement:
"inject specific messages into the last position of the message chain 
of the agent when a specific condition has been met"
"""

import asyncio
from rails import Rails, CounterCondition, StateCondition

async def main():
    # Initialize Rails
    rails = Rails()
    
    print("🚀 Rails Message Injection Demo")
    print("=" * 40)
    
    # Example 1: Basic counter-based injection
    print("\n📊 Example 1: Counter-based injection")
    help_message = {
        "role": "system", 
        "content": "I notice you've asked several questions. How can I better assist you?"
    }
    
    # Inject message when user has made 3+ interactions
    rails.when(lambda s: s.get_counter_sync('user_turns') >= 3).inject(help_message)
    
    # Simulate conversation turns
    messages = [{"role": "user", "content": "What's the weather?"}]
    
    for turn in range(1, 5):
        rails.store.increment_sync('user_turns', 1)
        result = await rails.check(messages)
        
        print(f"Turn {turn}: {len(result)} messages")
        if len(result) > 1:
            print(f"  💡 Injected: {result[-1]['content']}")
    
    # Example 2: State-based injection with different strategies
    print("\n🎯 Example 2: Multiple conditions and strategies")
    
    # Clear previous rules for clean demo
    rails.clear_rules()
    
    debug_msg = {"role": "system", "content": "[DEBUG MODE ACTIVE]"}
    error_msg = {"role": "assistant", "content": "I'll help you resolve this error."}
    
    # Prepend debug message when in debug mode
    rails.when(lambda s: s.get_sync('mode') == 'debug').inject(debug_msg, strategy='prepend')
    
    # Append help when error state is set
    rails.on_state('status', 'error', error_msg)
    
    # Test different scenarios
    scenarios = [
        {"mode": "normal", "status": "ok", "description": "Normal operation"},
        {"mode": "debug", "status": "ok", "description": "Debug mode only"},
        {"mode": "normal", "status": "error", "description": "Error state only"},
        {"mode": "debug", "status": "error", "description": "Debug + Error"},
    ]
    
    original_msg = [{"role": "user", "content": "I'm having trouble with my code"}]
    
    for scenario in scenarios:
        # Set state
        rails.store.set_sync('mode', scenario['mode'])
        rails.store.set_sync('status', scenario['status'])
        
        result = await rails.check(original_msg)
        
        print(f"\n{scenario['description']}: {len(result)} messages")
        for i, msg in enumerate(result):
            print(f"  [{i}] {msg['role']}: {msg['content']}")
    
    # Example 3: Complex condition logic
    print("\n🧠 Example 3: Complex conditions")
    rails.clear_rules()
    
    # Create complex condition using lambda
    complex_condition = lambda s: (
        s.get_counter_sync('errors') >= 2 and 
        s.get_sync('user_level') == 'beginner'
    )
    
    beginner_help = {
        "role": "assistant",
        "content": "I see you're encountering multiple errors. Would you like me to provide more detailed explanations?"
    }
    
    rails.when(complex_condition).inject(beginner_help)
    
    # Test complex condition
    rails.store.set_sync('user_level', 'beginner')
    rails.store.increment_sync('errors', 1)
    
    result = await rails.check([{"role": "user", "content": "Another error..."}])
    print(f"1 error, beginner: {len(result)} messages")
    
    rails.store.increment_sync('errors', 1)  # Now 2 errors
    result = await rails.check([{"role": "user", "content": "Another error..."}])
    print(f"2 errors, beginner: {len(result)} messages")
    if len(result) > 1:
        print(f"  💡 Injected: {result[-1]['content']}")
    
    # Example 4: Context manager usage
    print("\n🔄 Example 4: Context manager")
    
    async with Rails() as session_rails:
        session_msg = {"role": "system", "content": "Session started"}
        session_rails.when(lambda s: True).inject(session_msg)  # Always inject
        
        result = await session_rails.check([])
        print(f"Session context: {len(result)} messages")
        print(f"  Message: {result[0]['content']}")
    
    print("\n✅ Demo completed! Rails successfully implements message injection based on conditions.")

if __name__ == "__main__":
    asyncio.run(main())