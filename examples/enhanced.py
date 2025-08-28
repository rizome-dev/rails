import asyncio
import json
import time
from typing import Dict, Any, List

from rails import (
    Rails, current_rails, lifecycle_function, CounterCondition, StateCondition,
    execute_background_workflow, WorkflowOrchestrator
)


def api_client_tool(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example tool that uses current_rails() to track API usage and conditions.
    
    This demonstrates how tools can easily access and manipulate the Rails 
    instance they're running within.
    """
    rails = current_rails()
    
    # Track API calls
    rails.store.increment_sync('api_calls_total')
    rails.store.increment_sync(f'api_calls_{endpoint}')
    
    # Simulate API processing
    processing_time = 0.1 + (len(str(data)) * 0.001)
    time.sleep(processing_time)
    
    # Check for API rate limits
    total_calls = rails.store.get_counter_sync('api_calls_total')
    
    # Tool can add conditional rules based on its state
    if total_calls >= 8:  # Close to limit
        rails.when(
            lambda s: s.get_counter_sync('api_calls_total') >= 10
        ).inject({
            "role": "system",
            "content": "⚠️ API rate limit approaching. Consider throttling requests."
        })
    
    # Simulate different responses based on endpoint
    if endpoint == 'process_data':
        # Simulate potential errors
        if 'invalid' in str(data).lower():
            rails.store.increment_sync('api_errors')
            return {"success": False, "error": "Invalid data format"}
        
        return {"success": True, "processed": len(str(data)), "endpoint": endpoint}
    
    elif endpoint == 'validate':
        is_valid = 'name' in data and 'id' in data
        if not is_valid:
            rails.store.increment_sync('validation_errors')
        
        return {"success": is_valid, "validated": is_valid}
    
    else:
        return {"success": True, "endpoint": endpoint, "data_size": len(str(data))}


def data_sanitizer_tool(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Tool that sanitizes data and tracks sanitization stats."""
    rails = current_rails()
    
    # Track sanitization attempts
    rails.store.increment_sync('sanitization_attempts')
    
    # Simulate sanitization
    sanitized = {}
    for key, value in raw_data.items():
        # Remove potentially dangerous keys
        if key.lower() not in ['password', 'secret', 'token', 'key']:
            if isinstance(value, str):
                # Basic sanitization
                sanitized[key] = value.replace('<', '&lt;').replace('>', '&gt;')
            else:
                sanitized[key] = value
    
    # Track what was removed
    removed_keys = set(raw_data.keys()) - set(sanitized.keys())
    if removed_keys:
        rails.store.increment_sync('sensitive_keys_removed', len(removed_keys))
    
    # Set state based on sanitization result
    if len(sanitized) < len(raw_data):
        rails.store.set_sync('last_sanitization_removed_data', True)
    else:
        rails.store.set_sync('last_sanitization_removed_data', False)
    
    return {"sanitized_data": sanitized, "removed_keys": list(removed_keys)}


def quality_checker_tool(data: Dict[str, Any]) -> Dict[str, Any]:
    """Tool that checks data quality and sets Rails state accordingly."""
    rails = current_rails()
    
    # Track quality checks
    rails.store.increment_sync('quality_checks')
    
    # Check data quality
    quality_score = 0
    issues = []
    
    if 'name' in data:
        quality_score += 30
    else:
        issues.append("Missing name field")
    
    if 'id' in data:
        quality_score += 20
    else:
        issues.append("Missing id field")
        
    if 'sanitized_data' in data and data['sanitized_data']:
        quality_score += 25
    else:
        issues.append("Data not sanitized")
    
    if not issues:
        quality_score += 25  # Bonus for no issues
    
    # Set quality state in Rails
    rails.store.set_sync('last_quality_score', quality_score)
    
    # Track quality trends
    if quality_score >= 80:
        rails.store.increment_sync('high_quality_items')
    elif quality_score >= 60:
        rails.store.increment_sync('medium_quality_items')
    else:
        rails.store.increment_sync('low_quality_items')
    
    return {
        "quality_score": quality_score,
        "issues": issues,
        "passed": len(issues) == 0
    }


# ============================================================================
# LIFECYCLE FUNCTIONS FOR MODULAR SETUP/CLEANUP
# ============================================================================

@lifecycle_function(name="data_processing_setup", priority=10)
async def data_processing_lifecycle(rails):
    """
    Lifecycle function that sets up data processing environment.
    
    This demonstrates modular lifecycle management that can be composed
    with other lifecycle functions.
    """
    # Setup phase
    rails.store.set_sync('processing_session_id', f"session_{int(time.time())}")
    rails.store.set_sync('processing_start_time', time.time())
    rails.store.set_sync('processing_mode', 'production')
    
    print(f"🔧 Data processing environment initialized")
    print(f"   Session ID: {rails.store.get_sync('processing_session_id')}")
    
    yield  # Main execution happens here
    
    # Cleanup phase
    end_time = time.time()
    start_time = rails.store.get_sync('processing_start_time', end_time)
    duration = end_time - start_time
    
    rails.store.set_sync('processing_duration', duration)
    print(f"🔧 Data processing session completed in {duration:.2f}s")
    
    # Log final stats
    total_calls = rails.store.get_counter_sync('api_calls_total', 0)
    errors = rails.store.get_counter_sync('api_errors', 0)
    print(f"   Total API calls: {total_calls}, Errors: {errors}")


@lifecycle_function(name="monitoring_setup", priority=5)
async def monitoring_lifecycle(rails):
    """Lifecycle function for monitoring and alerting setup."""
    # Setup monitoring
    rails.store.set_sync('monitoring_active', True)
    rails.store.set_sync('alert_threshold_errors', 3)
    
    print("📊 Monitoring system activated")
    
    # Add monitoring rules during setup
    rails.when(
        lambda s: s.get_counter_sync('api_errors') >= s.get_sync('alert_threshold_errors', 3)
    ).inject({
        "role": "system",
        "content": "🚨 ALERT: High error rate detected! Please review recent operations."
    })
    
    yield  # Main execution
    
    # Cleanup monitoring
    rails.store.set_sync('monitoring_active', False)
    print("📊 Monitoring system deactivated")


async def error_recovery_workflow(rails):
    """Workflow that handles error recovery when triggered by conditions."""
    print("🔄 Error recovery workflow triggered")
    
    # Get error information
    error_count = rails.store.get_counter_sync('api_errors', 0)
    
    # Reset some counters to allow recovery
    if error_count >= 2:
        rails.store.set_counter_sync('api_errors', 0)
        rails.store.set_sync('recovery_mode', True)
        print(f"   Reset error count from {error_count} to 0")
        
        # Inject recovery message
        rails.when(lambda s: True).inject({
            "role": "assistant",
            "content": f"I've reset the error counter and enabled recovery mode. Previous errors: {error_count}"
        })


async def quality_improvement_workflow(rails):
    """Workflow that improves data quality when low quality is detected."""
    print("✨ Quality improvement workflow triggered")
    
    low_quality_count = rails.store.get_counter_sync('low_quality_items', 0)
    
    if low_quality_count >= 2:
        rails.store.set_sync('quality_improvement_active', True)
        
        # Inject guidance message
        rails.when(lambda s: True).inject({
            "role": "system",
            "content": f"Quality improvement mode activated. {low_quality_count} low-quality items detected."
        })
    
    # Background task to improve quality (simulated)
    await asyncio.sleep(0.5)  # Simulate improvement work
    rails.store.set_sync('quality_improvement_complete', True)
    print("   Quality improvement process completed")


async def performance_optimization_workflow(rails):
    """Background workflow for performance optimization."""
    print("⚡ Performance optimization running in background...")
    
    # Simulate optimization work
    await asyncio.sleep(1.0)
    
    # Update performance metrics
    current_score = rails.store.get_sync('performance_score', 75)
    optimized_score = min(100, current_score + 10)
    rails.store.set_sync('performance_score', optimized_score)
    
    print(f"   Performance optimization complete: {current_score} -> {optimized_score}")
    
    return {"original_score": current_score, "optimized_score": optimized_score}


async def demonstrate_enhanced_rails():
    """
    Main demonstration of enhanced Rails capabilities.
    
    This shows the composable patterns from the reference implementation
    applied to Rails for production-grade agent lifecycle management.
    """
    
    print("🚀 Enhanced Rails Demonstration - Composable Lifecycle Management")
    print("=" * 70)
    
    # Create Rails instance with composable lifecycle functions
    rails = Rails()
    
    # Demonstrate .with_lifecycle() composition
    rails.with_lifecycle("data_processing_setup", "monitoring_setup")
    
    # Setup conditional message injection (existing functionality)
    rails.when(
        CounterCondition('api_calls_total', 5, '>=')
    ).inject({
        "role": "system",
        "content": "📈 Processed 5+ API calls. System performing well."
    })
    
    # NEW: Setup conditional workflow execution with .then()
    rails.when(
        CounterCondition('api_errors', 2, '>=')
    ).then(error_recovery_workflow)
    
    rails.when(
        CounterCondition('low_quality_items', 2, '>=') 
    ).then(quality_improvement_workflow)
    
    # Background optimization workflow
    rails.when(
        CounterCondition('api_calls_total', 7, '>=')
    ).then(performance_optimization_workflow, background=True)
    
    # Stop condition when we've processed enough
    rails.when(
        CounterCondition('api_calls_total', 10, '>=')
    ).inject({
        "role": "system",
        "content": "🛑 STOP SIGNAL: Processed 10 API calls, demonstration complete."
    })
    
    # Use Rails with lifecycle management
    async with rails:
        print("\n📝 Starting data processing simulation...")
        
        # Simulate processing various data items using tools
        test_data = [
            {"name": "item1", "id": 1, "value": "good data"},
            {"name": "item2", "id": 2, "value": "also good", "password": "secret123"},
            {"invalid": "missing required fields"},
            {"name": "item4", "id": 4, "token": "abc123", "value": "mixed data"},
            {"name": "item5", "id": 5, "value": "clean data"},
            {"name": "item6", "value": "missing id"},
            {"name": "item7", "id": 7, "key": "private", "value": "more data"},
            {"name": "item8", "id": 8, "value": "good quality"},
            {"name": "item9", "id": 9, "secret": "hidden", "value": "needs cleaning"},
            {"name": "item10", "id": 10, "value": "final item"},
        ]
        
        messages = []
        
        for i, data_item in enumerate(test_data):
            print(f"\n--- Processing Item {i+1} ---")
            
            # Step 1: Sanitize data using tool (tool accesses Rails via current_rails())
            sanitized = data_sanitizer_tool(data_item)
            print(f"Sanitized: removed {len(sanitized['removed_keys'])} sensitive keys")
            
            # Step 2: Quality check
            quality_result = quality_checker_tool(sanitized)
            print(f"Quality score: {quality_result['quality_score']}/100")
            
            # Step 3: Process via API (may cause errors)
            if quality_result['quality_score'] >= 60:
                api_result = api_client_tool('process_data', sanitized['sanitized_data'])
                print(f"API result: {api_result}")
            else:
                # Simulate validation for low quality items
                api_result = api_client_tool('validate', data_item)
                print(f"Validation result: {api_result}")
            
            # Check Rails conditions and apply rules
            enhanced_messages = await rails.check(messages)
            
            # Display any new messages from Rails
            if len(enhanced_messages) > len(messages):
                for new_msg in enhanced_messages[len(messages):]:
                    print(f"💬 Rails: {new_msg['content']}")
            
            messages = enhanced_messages
            
            # Short delay to see the progression
            await asyncio.sleep(0.2)
            
            # Stop if we hit the stop signal
            if any("STOP SIGNAL" in msg.get('content', '') for msg in messages):
                print("\n🛑 Stop signal detected, ending processing...")
                break
    
    print("\n📊 Final Statistics:")
    print(f"   API Calls: {rails.store.get_counter_sync('api_calls_total', 0)}")
    print(f"   API Errors: {rails.store.get_counter_sync('api_errors', 0)}")
    print(f"   High Quality Items: {rails.store.get_counter_sync('high_quality_items', 0)}")
    print(f"   Medium Quality Items: {rails.store.get_counter_sync('medium_quality_items', 0)}")
    print(f"   Low Quality Items: {rails.store.get_counter_sync('low_quality_items', 0)}")
    print(f"   Sensitive Keys Removed: {rails.store.get_counter_sync('sensitive_keys_removed', 0)}")
    print(f"   Performance Score: {rails.store.get_sync('performance_score', 'N/A')}")
    print(f"   Recovery Mode: {rails.store.get_sync('recovery_mode', False)}")
    
    print(f"\n🎯 Rails Configuration:")
    print(f"   Total Rules: {rails.rule_count()}")
    print(f"   Injection Rules: {rails.injection_rule_count()}")
    print(f"   Execution Rules: {rails.execution_rule_count()}")
    
    print("\n✅ Enhanced Rails demonstration complete!")
    print("   - Tools successfully accessed Rails via current_rails()")
    print("   - Lifecycle functions provided modular setup/cleanup")
    print("   - .then() method executed workflows on conditions")
    print("   - Background execution handled performance optimization")
    print("   - Composable patterns enabled flexible workflow management")


async def demonstrate_advanced_orchestration():
    """
    Demonstrate advanced workflow orchestration capabilities.
    
    This shows more complex patterns using the new execution capabilities.
    """
    print("\n🎼 Advanced Workflow Orchestration Demo")
    print("=" * 50)
    
    rails = Rails()
    
    async with rails:
        # Create orchestrator for complex workflows
        orchestrator = WorkflowOrchestrator(rails, max_concurrent=3)
        
        async with orchestrator.orchestration_context():
            print("🔄 Running conditional pipeline...")
            
            # Define conditional pipeline
            pipeline_steps = [
                (lambda s: True, lambda: print("Step 1: Initialize")),
                (lambda s: s.get_counter_sync('test_counter', 0) >= 0, lambda: rails.store.increment_sync('test_counter')),
                (lambda s: s.get_counter_sync('test_counter', 0) >= 1, lambda: print("Step 3: Process data")),
                (lambda s: True, lambda: print("Step 4: Finalize")),
            ]
            
            pipeline_results = await orchestrator.execute_conditional_pipeline(pipeline_steps)
            print(f"Pipeline results: {len(pipeline_results)} steps executed")
            
            print("\n⚡ Running parallel workflows...")
            
            # Define parallel workflows
            parallel_workflows = [
                lambda: time.sleep(0.1) or "Task A complete",
                lambda: time.sleep(0.2) or "Task B complete", 
                lambda: time.sleep(0.05) or "Task C complete",
            ]
            
            parallel_results = await orchestrator.execute_parallel_workflows(parallel_workflows)
            print(f"Parallel results: {len(parallel_results)} workflows completed")
            
            for task_id, result in parallel_results.items():
                if result['success']:
                    print(f"   {task_id}: {result['result']}")
    
    print("✅ Advanced orchestration demo complete!")


if __name__ == "__main__":
    async def main():
        await demonstrate_enhanced_rails()
        await demonstrate_advanced_orchestration()
    
    asyncio.run(main())
