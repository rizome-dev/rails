"""
Modern Enhanced Rails Example - Lifecycle Orchestration Patterns

This example demonstrates advanced Rails patterns with the new architecture:
1. Tools using current_rails() for store access
2. State-based guidance and error recovery
3. Queue-based task management
4. Event streaming and observability
5. Complex workflow orchestration
"""

import asyncio
import time
from typing import Dict, Any, List
from datetime import datetime

from rails import (
    Rails, current_rails, Message, Role, 
    counter, state, queue, system, template
)


def data_processing_tool(data_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Advanced tool demonstrating current_rails() usage patterns.
    
    This tool performs data processing while updating Rails store
    for sophisticated lifecycle orchestration.
    """
    rails = current_rails()
    
    # Track processing attempts
    asyncio.create_task(rails.store.increment('processing_attempts'))
    
    # Simulate processing with different outcomes
    item_id = data_item.get('id', 'unknown')
    processing_time = 0.1 + (len(str(data_item)) * 0.001)
    time.sleep(processing_time)
    
    # Determine processing outcome
    result = {'item_id': item_id, 'processed': True}
    
    # Simulate different processing scenarios
    content = str(data_item.get('content', ''))
    
    if 'error' in content.lower():
        # Simulate processing error
        result['success'] = False
        result['error'] = 'Processing failed'
        asyncio.create_task(rails.store.increment('processing_errors'))
        asyncio.create_task(rails.store.set('last_error_item', item_id))
    elif len(content) < 10:
        # Low quality data
        result['success'] = False
        result['error'] = 'Insufficient data quality'
        asyncio.create_task(rails.store.increment('quality_failures'))
        asyncio.create_task(rails.store.set('last_quality_issue', item_id))
    else:
        # Successful processing
        result['success'] = True
        result['quality_score'] = min(100, len(content) * 2)
        asyncio.create_task(rails.store.increment('successful_processing'))
        
        # Update quality metrics
        if result['quality_score'] >= 80:
            asyncio.create_task(rails.store.increment('high_quality_items'))
        else:
            asyncio.create_task(rails.store.increment('medium_quality_items'))
    
    # Update processing state for Rails conditions
    asyncio.create_task(rails.store.set('last_processed_item', item_id))
    asyncio.create_task(rails.store.set('last_processing_time', datetime.now().isoformat()))
    
    # Queue management - add to appropriate queue based on outcome
    if result['success']:
        asyncio.create_task(rails.store.push_queue('completed_items', {
            'item_id': item_id,
            'timestamp': datetime.now().isoformat(),
            'quality_score': result.get('quality_score', 0)
        }))
    else:
        asyncio.create_task(rails.store.push_queue('failed_items', {
            'item_id': item_id,
            'error': result.get('error', 'Unknown error'),
            'timestamp': datetime.now().isoformat()
        }))
    
    return result


def quality_analyzer_tool(processing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tool for analyzing processing quality and setting Rails state."""
    rails = current_rails()
    
    # Analyze batch quality
    total_items = len(processing_results)
    successful_items = sum(1 for r in processing_results if r.get('success', False))
    
    success_rate = successful_items / total_items if total_items > 0 else 0
    
    # Update Rails state with analysis results
    asyncio.create_task(rails.store.set('batch_success_rate', success_rate))
    asyncio.create_task(rails.store.set('batch_size', total_items))
    asyncio.create_task(rails.store.increment('batches_analyzed'))
    
    # Set alert conditions
    if success_rate < 0.5:
        asyncio.create_task(rails.store.set('quality_crisis', True))
        asyncio.create_task(rails.store.increment('quality_crises'))
    elif success_rate < 0.8:
        asyncio.create_task(rails.store.set('quality_concern', True))
    else:
        asyncio.create_task(rails.store.set('quality_excellent', True))
    
    return {
        'total_items': total_items,
        'successful_items': successful_items,
        'success_rate': success_rate,
        'analysis_complete': True
    }


def performance_monitor_tool() -> Dict[str, Any]:
    """Tool for monitoring system performance and updating Rails state."""
    rails = current_rails()
    
    # Get current metrics from Rails store
    try:
        # These will be sync operations wrapped in asyncio.create_task
        total_processed = asyncio.create_task(rails.store.get_counter('processing_attempts', 0))
        errors = asyncio.create_task(rails.store.get_counter('processing_errors', 0))
        
        # Calculate performance metrics
        error_rate = 0  # We'll update this with actual values in real async context
        
        # Set performance state
        asyncio.create_task(rails.store.set('system_error_rate', error_rate))
        asyncio.create_task(rails.store.set('monitoring_timestamp', datetime.now().isoformat()))
        
        # Performance thresholds
        if error_rate > 0.2:  # 20% error rate
            asyncio.create_task(rails.store.set('performance_degraded', True))
        
        return {
            'monitoring_active': True,
            'error_rate': error_rate,
            'status': 'healthy' if error_rate < 0.1 else 'degraded'
        }
    
    except Exception as e:
        # Handle monitoring errors gracefully
        asyncio.create_task(rails.store.increment('monitoring_errors'))
        return {'monitoring_active': False, 'error': str(e)}


async def error_recovery_workflow(rails: Rails) -> None:
    """Workflow for handling error recovery when triggered by conditions."""
    print("🔄 Error recovery workflow activated")
    
    # Get current error state
    error_count = await rails.store.get_counter('processing_errors', 0)
    quality_failures = await rails.store.get_counter('quality_failures', 0)
    
    # Implement recovery strategies
    if error_count >= 3:
        print(f"  📊 High error count detected: {error_count}")
        
        # Reset error counters to allow recovery
        await rails.store.reset_counter('processing_errors')
        await rails.store.set('recovery_mode', True)
        await rails.store.set('recovery_timestamp', datetime.now().isoformat())
        
        print("  ✅ Error counters reset, recovery mode enabled")
    
    if quality_failures >= 2:
        print(f"  🔍 Quality issues detected: {quality_failures}")
        
        # Enable enhanced quality checking
        await rails.store.set('enhanced_quality_mode', True)
        print("  ✅ Enhanced quality mode enabled")
    
    # Log recovery action
    await rails.store.increment('recovery_actions')


async def performance_optimization_workflow(rails: Rails) -> Dict[str, Any]:
    """Background workflow for performance optimization."""
    print("⚡ Performance optimization workflow running...")
    
    # Simulate optimization work
    await asyncio.sleep(0.5)
    
    # Get current performance metrics
    total_processed = await rails.store.get_counter('processing_attempts', 0)
    successful = await rails.store.get_counter('successful_processing', 0)
    
    # Calculate and update performance score
    if total_processed > 0:
        performance_score = (successful / total_processed) * 100
    else:
        performance_score = 100
    
    # Apply optimizations
    optimized_score = min(100, performance_score + 5)
    await rails.store.set('performance_score', optimized_score)
    await rails.store.set('optimization_timestamp', datetime.now().isoformat())
    
    print(f"  📈 Performance optimized: {performance_score:.1f}% → {optimized_score:.1f}%")
    
    return {
        'original_score': performance_score,
        'optimized_score': optimized_score,
        'optimization_applied': True
    }


async def queue_management_workflow(rails: Rails) -> None:
    """Workflow for managing queues and task distribution."""
    print("📦 Queue management workflow activated")
    
    # Check queue states
    completed_count = await rails.store.queue_length('completed_items')
    failed_count = await rails.store.queue_length('failed_items')
    
    print(f"  📊 Queue status: {completed_count} completed, {failed_count} failed")
    
    # Process failed items for retry if needed
    if failed_count > 3:
        # Move some failed items to retry queue
        for _ in range(min(3, failed_count)):
            failed_item = await rails.store.pop_queue('failed_items')
            if failed_item:
                # Add retry metadata
                failed_item['retry_timestamp'] = datetime.now().isoformat()
                failed_item['retry_count'] = failed_item.get('retry_count', 0) + 1
                
                await rails.store.push_queue('retry_queue', failed_item)
        
        await rails.store.set('retry_processing_active', True)
        print(f"  🔄 Moved items to retry queue")
    
    # Archive completed items if queue is full
    if completed_count > 10:
        archived_count = 0
        while await rails.store.queue_length('completed_items') > 5:
            item = await rails.store.pop_queue('completed_items')
            if item:
                await rails.store.push_queue('archived_items', item)
                archived_count += 1
        
        print(f"  📚 Archived {archived_count} completed items")
    
    await rails.store.increment('queue_management_cycles')


async def demonstrate_modern_enhanced_rails():
    """
    Comprehensive demonstration of enhanced Rails capabilities.
    """
    print("🚀 Modern Enhanced Rails - Advanced Lifecycle Orchestration")
    print("=" * 65)
    
    # Initialize Rails
    rails = Rails()
    
    # =================================================================
    # SETUP ADVANCED LIFECYCLE RULES
    # =================================================================
    
    print("🔧 Configuring advanced Rails lifecycle orchestration...")
    
    # 1. Error recovery trigger
    rails.add_rule(
        condition=counter('processing_errors') >= 3,
        action=lambda messages: asyncio.create_task(error_recovery_workflow(rails)) or messages,
        name="error_recovery_trigger"
    )
    
    # 2. Quality crisis management
    rails.add_rule(
        condition=state('quality_crisis') == True,
        action=system("🚨 QUALITY CRISIS: Success rate below 50% - immediate action required!"),
        name="quality_crisis_alert"
    )
    
    # 3. Performance optimization trigger
    rails.add_rule(
        condition=counter('processing_attempts') >= 8,
        action=lambda messages: asyncio.create_task(performance_optimization_workflow(rails)) or messages,
        name="performance_optimization"
    )
    
    # 4. Queue management trigger
    rails.add_rule(
        condition=queue('failed_items').length >= 3,
        action=lambda messages: asyncio.create_task(queue_management_workflow(rails)) or messages,
        name="queue_management"
    )
    
    # 5. Recovery mode guidance
    rails.add_rule(
        condition=state('recovery_mode') == True,
        action=template("🔄 Recovery mode active since {recovery_timestamp} - monitoring closely"),
        name="recovery_status"
    )
    
    # 6. Success celebration with templates
    rails.add_rule(
        condition=state('quality_excellent') == True,
        action=template("🎉 Excellent quality achieved! Success rate: {batch_success_rate:.1%}"),
        name="quality_celebration"
    )
    
    # 7. Complex condition for performance monitoring
    async def check_performance_degradation(store):
        """Custom condition for performance monitoring."""
        error_rate = await store.get('system_error_rate', 0)
        performance_score = await store.get('performance_score', 100)
        
        return error_rate > 0.15 or performance_score < 70
    
    rails.add_rule(
        condition=check_performance_degradation,
        action=system("⚡ Performance degradation detected - optimization recommended"),
        name="performance_monitor"
    )
    
    # 8. Stop condition
    rails.add_rule(
        condition=counter('processing_attempts') >= 15,
        action=system("🛑 Processing limit reached - demonstration complete!"),
        name="completion_trigger"
    )
    
    print(f"✅ Configured {len(rails.rules)} advanced Rails rules")
    
    # =================================================================
    # RUN ENHANCED DEMONSTRATION
    # =================================================================
    
    async with rails:
        print(f"\\n🔄 Starting enhanced lifecycle orchestration...")
        
        # Simulate complex data processing scenarios
        test_datasets = [
            # Batch 1: Mixed quality
            [
                {'id': 'item_1', 'content': 'High quality data item with sufficient content for processing'},
                {'id': 'item_2', 'content': 'Another good quality item'},
                {'id': 'item_3', 'content': 'bad'},  # Low quality
            ],
            
            # Batch 2: Error scenarios
            [
                {'id': 'item_4', 'content': 'Good data for processing validation'},
                {'id': 'item_5', 'content': 'error in processing'},  # Error trigger
                {'id': 'item_6', 'content': 'Quality content here'},
            ],
            
            # Batch 3: More errors to trigger recovery
            [
                {'id': 'item_7', 'content': 'processing error occurred'},  # Error
                {'id': 'item_8', 'content': 'system error detected'},     # Error
                {'id': 'item_9', 'content': 'Excellent quality data item'},
            ],
            
            # Batch 4: Recovery testing
            [
                {'id': 'item_10', 'content': 'Post-recovery high quality data processing'},
                {'id': 'item_11', 'content': 'System performance validation content'},
                {'id': 'item_12', 'content': 'Final quality assurance data'},
            ],
            
            # Batch 5: Performance testing
            [
                {'id': 'item_13', 'content': 'Performance optimization test data'},
                {'id': 'item_14', 'content': 'Load testing with quality content'},
                {'id': 'item_15', 'content': 'Final batch processing validation'},
            ]
        ]
        
        all_messages = []
        
        for batch_num, dataset in enumerate(test_datasets):
            print(f"\\n📊 Processing Batch {batch_num + 1}: {len(dataset)} items")
            
            # Process each item using our tools
            batch_results = []
            for item in dataset:
                result = data_processing_tool(item)
                batch_results.append(result)
                
                # Brief pause between items
                await asyncio.sleep(0.05)
            
            # Analyze batch quality
            analysis = quality_analyzer_tool(batch_results)
            print(f"  📈 Batch analysis: {analysis['successful_items']}/{analysis['total_items']} success ({analysis['success_rate']:.1%})")
            
            # Monitor system performance
            performance = performance_monitor_tool()
            
            # Process any messages through Rails lifecycle
            batch_messages = [Message(role=Role.USER, content=f"Processed batch {batch_num + 1}")]
            enhanced_messages = await rails.process(batch_messages)
            
            # Show any Rails injections
            if len(enhanced_messages) > len(batch_messages):
                injected = enhanced_messages[len(batch_messages):]
                for msg in injected:
                    print(f"  💬 Rails: {msg.content}")
            
            all_messages.extend(enhanced_messages)
            
            # Check for stop condition
            if any("complete!" in msg.content for msg in enhanced_messages if msg.injected_by_rails):
                print("\\n🛑 Completion trigger activated!")
                break
            
            await asyncio.sleep(0.2)
    
    # =================================================================
    # FINAL COMPREHENSIVE METRICS
    # =================================================================
    
    print("\\n📊 Final Enhanced Rails Metrics:")
    print(f"  • Processing attempts: {await rails.store.get_counter('processing_attempts')}")
    print(f"  • Successful processing: {await rails.store.get_counter('successful_processing', 0)}")
    print(f"  • Processing errors: {await rails.store.get_counter('processing_errors', 0)}")
    print(f"  • Quality failures: {await rails.store.get_counter('quality_failures', 0)}")
    print(f"  • High quality items: {await rails.store.get_counter('high_quality_items', 0)}")
    print(f"  • Recovery actions: {await rails.store.get_counter('recovery_actions', 0)}")
    print(f"  • Queue management cycles: {await rails.store.get_counter('queue_management_cycles', 0)}")
    
    # Queue status
    print(f"\\n📦 Queue Status:")
    print(f"  • Completed items: {await rails.store.queue_length('completed_items')}")
    print(f"  • Failed items: {await rails.store.queue_length('failed_items')}")
    print(f"  • Retry queue: {await rails.store.queue_length('retry_queue')}")
    print(f"  • Archived items: {await rails.store.queue_length('archived_items')}")
    
    # System state
    print(f"\\n🔧 System State:")
    print(f"  • Recovery mode: {await rails.store.get('recovery_mode', False)}")
    print(f"  • Enhanced quality mode: {await rails.store.get('enhanced_quality_mode', False)}")
    print(f"  • Performance score: {await rails.store.get('performance_score', 'N/A')}")
    print(f"  • Last success rate: {await rails.store.get('batch_success_rate', 'N/A')}")
    
    # Rails orchestration metrics
    metrics = await rails.emit_metrics()
    print(f"\\n⚙️ Rails Orchestration:")
    print(f"  • Final state: {metrics['state']}")
    print(f"  • Rules configured: {metrics['total_rules']}")
    print(f"  • Rules active: {metrics['active_rules']}")
    
    print(f"\\n🎯 Enhanced Rails Features Demonstrated:")
    print(f"  ✅ Tools accessing Rails via current_rails()")
    print(f"  ✅ State-based guidance and error recovery")
    print(f"  ✅ Queue-based task management")
    print(f"  ✅ Event streaming and observability")
    print(f"  ✅ Complex workflow orchestration")
    print(f"  ✅ Performance monitoring and optimization")
    print(f"  ✅ Template injection with live data")
    print(f"  ✅ Custom condition functions")
    print(f"  ✅ Automatic lifecycle management")
    
    print(f"\\n✨ Enhanced Rails demonstration complete!")


if __name__ == "__main__":
    asyncio.run(demonstrate_modern_enhanced_rails())