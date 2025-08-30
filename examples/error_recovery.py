"""
Error Recovery Example - Robust Failure Handling Patterns

This example demonstrates Rails error recovery and resilience patterns:
1. Multi-level error detection and classification
2. Automatic recovery strategies and circuit breakers
3. Degraded mode operations
4. Error pattern analysis and adaptive responses
5. State-based recovery workflows
"""

import asyncio
import random
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from rails import (
    Rails, current_rails, Message, Role,
    counter, state, system, template
)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SystemStatus(Enum):
    """System operational status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"


@dataclass
class ErrorEvent:
    """Represents an error event in the system."""
    id: str
    component: str
    severity: ErrorSeverity
    message: str
    timestamp: str
    context: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[str] = None


def system_health_checker_tool() -> Dict[str, Any]:
    """Tool for monitoring system health and detecting issues."""
    rails = current_rails()
    
    # Simulate health check with random failures
    components = ['database', 'api_service', 'cache', 'queue_processor', 'file_storage']
    health_status = {}
    
    for component in components:
        # Simulate component health (80% healthy)
        is_healthy = random.random() > 0.2
        health_status[component] = is_healthy
        
        if not is_healthy:
            # Log component failure
            asyncio.create_task(rails.store.increment(f'errors_{component}'))
            asyncio.create_task(rails.store.increment('total_errors'))
            asyncio.create_task(rails.store.set(f'{component}_status', 'unhealthy'))
            
            # Create error event
            error_id = f"err_{int(asyncio.get_event_loop().time() * 1000)}"
            error_event = ErrorEvent(
                id=error_id,
                component=component,
                severity=ErrorSeverity.HIGH if component in ['database', 'api_service'] else ErrorSeverity.MEDIUM,
                message=f"{component} health check failed",
                timestamp=datetime.now().isoformat(),
                context={"health_check": True}
            )
            
            # Store error for pattern analysis
            asyncio.create_task(rails.store.push_queue('error_events', error_event.__dict__))
        else:
            asyncio.create_task(rails.store.set(f'{component}_status', 'healthy'))
    
    # Calculate overall system health
    healthy_count = sum(health_status.values())
    health_percentage = (healthy_count / len(components)) * 100
    
    # Update system status
    asyncio.create_task(rails.store.set('system_health_percentage', health_percentage))
    asyncio.create_task(rails.store.set('last_health_check', datetime.now().isoformat()))
    
    if health_percentage >= 80:
        asyncio.create_task(rails.store.set('system_status', SystemStatus.HEALTHY.value))
    elif health_percentage >= 60:
        asyncio.create_task(rails.store.set('system_status', SystemStatus.DEGRADED.value))
    else:
        asyncio.create_task(rails.store.set('system_status', SystemStatus.CRITICAL.value))
    
    return {
        "health_percentage": health_percentage,
        "component_status": health_status,
        "healthy_components": healthy_count,
        "total_components": len(components)
    }


def error_simulator_tool(component: str, severity: str = "medium") -> Dict[str, Any]:
    """Tool for simulating different types of errors."""
    rails = current_rails()
    
    # Create simulated error
    error_id = f"sim_{int(asyncio.get_event_loop().time() * 1000)}"
    error_severity = ErrorSeverity(severity.lower())
    
    error_messages = {
        ErrorSeverity.LOW: "Minor performance degradation detected",
        ErrorSeverity.MEDIUM: f"{component} experiencing intermittent issues",
        ErrorSeverity.HIGH: f"{component} service disruption - functionality impacted", 
        ErrorSeverity.CRITICAL: f"CRITICAL: {component} complete failure - immediate attention required"
    }
    
    error_event = ErrorEvent(
        id=error_id,
        component=component,
        severity=error_severity,
        message=error_messages[error_severity],
        timestamp=datetime.now().isoformat(),
        context={"simulated": True, "component": component}
    )
    
    # Update error tracking
    asyncio.create_task(rails.store.increment(f'errors_{component}'))
    asyncio.create_task(rails.store.increment('total_errors'))
    asyncio.create_task(rails.store.increment(f'errors_{severity.lower()}'))
    
    # Store error event
    asyncio.create_task(rails.store.push_queue('error_events', error_event.__dict__))
    
    # Update component status
    if error_severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
        asyncio.create_task(rails.store.set(f'{component}_status', 'failed'))
    else:
        asyncio.create_task(rails.store.set(f'{component}_status', 'degraded'))
    
    # Trigger cascading effects for critical errors
    if error_severity == ErrorSeverity.CRITICAL:
        asyncio.create_task(rails.store.set('system_status', SystemStatus.CRITICAL.value))
        asyncio.create_task(rails.store.set('critical_error_active', True))
        asyncio.create_task(rails.store.set('last_critical_error', error_id))
    
    return {
        "error_id": error_id,
        "component": component,
        "severity": severity,
        "message": error_event.message,
        "simulated": True
    }


def recovery_action_tool(component: str, action_type: str = "restart") -> Dict[str, Any]:
    """Tool for performing recovery actions on failed components."""
    rails = current_rails()
    
    # Simulate recovery actions
    recovery_actions = {
        "restart": {"success_rate": 0.7, "time": 2},
        "reset": {"success_rate": 0.8, "time": 3},
        "rollback": {"success_rate": 0.9, "time": 5},
        "manual_intervention": {"success_rate": 0.95, "time": 10}
    }
    
    action_config = recovery_actions.get(action_type, recovery_actions["restart"])
    
    # Simulate recovery attempt
    success = random.random() < action_config["success_rate"]
    recovery_time = action_config["time"]
    
    # Update recovery metrics
    asyncio.create_task(rails.store.increment('recovery_attempts'))
    asyncio.create_task(rails.store.increment(f'recovery_attempts_{action_type}'))
    
    if success:
        # Recovery successful
        asyncio.create_task(rails.store.increment('recovery_successes'))
        asyncio.create_task(rails.store.set(f'{component}_status', 'healthy'))
        asyncio.create_task(rails.store.set(f'{component}_last_recovery', datetime.now().isoformat()))
        
        # Reset error counters for this component
        asyncio.create_task(rails.store.reset_counter(f'errors_{component}'))
        
        return {
            "component": component,
            "action": action_type,
            "status": "success",
            "recovery_time": recovery_time,
            "timestamp": datetime.now().isoformat()
        }
    else:
        # Recovery failed
        asyncio.create_task(rails.store.increment('recovery_failures'))
        asyncio.create_task(rails.store.set(f'{component}_recovery_failed', True))
        
        return {
            "component": component,
            "action": action_type,
            "status": "failed",
            "recovery_time": recovery_time,
            "needs_escalation": True
        }


def circuit_breaker_tool(component: str, operation: str) -> Dict[str, Any]:
    """Tool implementing circuit breaker pattern for fault tolerance."""
    rails = current_rails()
    
    async def check_circuit_state():
        # Get current error count for component
        error_count = await rails.store.get_counter(f'errors_{component}', 0)
        circuit_state = await rails.store.get(f'{component}_circuit_state', 'closed')
        last_attempt = await rails.store.get(f'{component}_last_attempt')
        
        # Circuit breaker thresholds
        failure_threshold = 5
        timeout_duration = 30  # seconds
        
        if circuit_state == 'closed':
            # Normal operation
            if error_count >= failure_threshold:
                # Trip circuit breaker
                await rails.store.set(f'{component}_circuit_state', 'open')
                await rails.store.set(f'{component}_circuit_opened', datetime.now().isoformat())
                await rails.store.increment('circuit_breakers_tripped')
                return {"status": "circuit_opened", "reason": "failure_threshold_exceeded"}
            else:
                # Allow operation
                return {"status": "operation_allowed", "circuit_state": "closed"}
        
        elif circuit_state == 'open':
            # Circuit is open, check if timeout has elapsed
            if last_attempt:
                last_time = datetime.fromisoformat(last_attempt)
                if (datetime.now() - last_time).seconds >= timeout_duration:
                    # Move to half-open state
                    await rails.store.set(f'{component}_circuit_state', 'half_open')
                    return {"status": "testing_allowed", "circuit_state": "half_open"}
            
            return {"status": "operation_blocked", "circuit_state": "open"}
        
        elif circuit_state == 'half_open':
            # Test operation - simulate success/failure
            test_success = random.random() > 0.3  # 70% success rate in recovery
            
            if test_success:
                # Close circuit
                await rails.store.set(f'{component}_circuit_state', 'closed')
                await rails.store.reset_counter(f'errors_{component}')
                await rails.store.increment('circuit_breakers_closed')
                return {"status": "circuit_closed", "recovery": True}
            else:
                # Back to open
                await rails.store.set(f'{component}_circuit_state', 'open')
                await rails.store.increment(f'errors_{component}')
                return {"status": "circuit_reopened", "test_failed": True}
        
        return {"status": "unknown_state"}
    
    # Update last attempt timestamp
    asyncio.create_task(rails.store.set(f'{component}_last_attempt', datetime.now().isoformat()))
    
    # Schedule circuit check
    asyncio.create_task(check_circuit_state())
    
    return {"circuit_check": "scheduled", "component": component, "operation": operation}


async def automatic_recovery_workflow(rails: Rails) -> None:
    """Workflow for automatic error recovery."""
    print("🔧 Automatic recovery workflow activated")
    
    # Get current error state
    total_errors = await rails.store.get_counter('total_errors', 0)
    critical_active = await rails.store.get('critical_error_active', False)
    system_status = await rails.store.get('system_status', SystemStatus.HEALTHY.value)
    
    print(f"  📊 Error analysis: {total_errors} total errors, status: {system_status}")
    
    # Implement recovery strategies based on error patterns
    if critical_active:
        print("  🚨 Critical error detected - initiating emergency recovery")
        
        # Emergency recovery sequence
        critical_components = ['database', 'api_service']
        for component in critical_components:
            component_status = await rails.store.get(f'{component}_status', 'unknown')
            if component_status in ['failed', 'unhealthy']:
                print(f"    🔄 Emergency recovery for {component}")
                # Simulate emergency recovery
                await rails.store.set(f'{component}_status', 'recovering')
                await asyncio.sleep(0.1)  # Simulate recovery time
                await rails.store.set(f'{component}_status', 'healthy')
        
        # Clear critical error state
        await rails.store.set('critical_error_active', False)
        await rails.store.set('system_status', SystemStatus.RECOVERING.value)
        await rails.store.increment('emergency_recoveries')
        
    elif total_errors >= 5:
        print("  ⚠️ Multiple errors detected - systematic recovery")
        
        # Get error events for analysis
        error_events = await rails.store.get_queue('error_events')
        
        # Group errors by component
        component_errors = {}
        for error_data in error_events:
            component = error_data.get('component', 'unknown')
            component_errors[component] = component_errors.get(component, 0) + 1
        
        # Prioritize recovery by error count
        for component, error_count in sorted(component_errors.items(), key=lambda x: x[1], reverse=True):
            if error_count >= 2:
                print(f"    🔧 Recovering {component} ({error_count} errors)")
                await rails.store.set(f'{component}_status', 'recovering')
                await asyncio.sleep(0.05)
                await rails.store.set(f'{component}_status', 'healthy')
                await rails.store.reset_counter(f'errors_{component}')
        
        await rails.store.increment('systematic_recoveries')
        await rails.store.set('system_status', SystemStatus.HEALTHY.value)
    
    # Update recovery timestamp
    await rails.store.set('last_recovery_time', datetime.now().isoformat())
    print("  ✅ Automatic recovery completed")


async def degraded_mode_workflow(rails: Rails) -> None:
    """Workflow for operating in degraded mode during persistent issues."""
    print("🔄 Degraded mode workflow activated")
    
    # Enable degraded mode operations
    await rails.store.set('degraded_mode_active', True)
    await rails.store.set('degraded_mode_start', datetime.now().isoformat())
    
    # Disable non-critical features
    non_critical_features = ['analytics', 'recommendations', 'background_jobs']
    for feature in non_critical_features:
        await rails.store.set(f'{feature}_enabled', False)
        print(f"    ⏸️ Disabled {feature}")
    
    # Enable fallback mechanisms
    fallback_configs = {
        'cache_ttl': 300,  # Longer cache TTL
        'rate_limit': 50,  # Lower rate limits
        'timeout': 5000    # Shorter timeouts
    }
    
    for config, value in fallback_configs.items():
        await rails.store.set(f'fallback_{config}', value)
        print(f"    🔧 Set {config} to {value}")
    
    await rails.store.increment('degraded_mode_activations')
    print("  ✅ Degraded mode configuration applied")


async def demonstrate_error_recovery():
    """Comprehensive demonstration of error recovery patterns."""
    print("🚀 Error Recovery & Resilience Patterns Demo")
    print("=" * 48)
    
    # Initialize Rails
    rails = Rails()
    
    # =================================================================
    # SETUP ERROR RECOVERY LIFECYCLE RULES
    # =================================================================
    
    print("🔧 Configuring error recovery Rails lifecycle rules...")
    
    # 1. Critical error immediate response
    rails.add_rule(
        condition=state('critical_error_active') == True,
        action=system("🚨 CRITICAL ERROR DETECTED - Initiating emergency protocols!"),
        name="critical_error_alert"
    )
    
    # 2. High error count triggers automatic recovery
    rails.add_rule(
        condition=counter('total_errors') >= 5,
        action=lambda messages: asyncio.create_task(automatic_recovery_workflow(rails)) or messages,
        name="auto_recovery_trigger"
    )
    
    # 3. System health degradation response
    async def check_system_degraded(store):
        health = await store.get('system_health_percentage', 100)
        return health < 70
    
    rails.add_rule(
        condition=check_system_degraded,
        action=lambda messages: asyncio.create_task(degraded_mode_workflow(rails)) or messages,
        name="degraded_mode_trigger"
    )
    
    # 4. Circuit breaker notifications
    rails.add_rule(
        condition=counter('circuit_breakers_tripped') >= 2,
        action=system("⚡ Multiple circuit breakers tripped - system protection active"),
        name="circuit_breaker_alert"
    )
    
    # 5. Recovery success celebration
    rails.add_rule(
        condition=counter('recovery_successes') >= 3,
        action=template("🎉 Recovery successful! {recovery_successes} components restored"),
        name="recovery_success"
    )
    
    # 6. Persistent failure escalation
    rails.add_rule(
        condition=counter('recovery_failures') >= 3,
        action=system("🔺 Multiple recovery failures - escalating to human intervention"),
        name="recovery_escalation"
    )
    
    # 7. System health restored
    async def check_system_healthy(store):
        health = await store.get('system_health_percentage', 100)
        degraded_active = await store.get('degraded_mode_active', False)
        return health >= 80 and degraded_active
    
    rails.add_rule(
        condition=check_system_healthy,
        action=system("✅ System health restored - exiting degraded mode"),
        name="health_restored"
    )
    
    # 8. Demo completion
    rails.add_rule(
        condition=counter('systematic_recoveries') >= 2,
        action=system("🛑 Error recovery demo complete! Resilience patterns mastered."),
        name="demo_completion"
    )
    
    print(f"✅ Configured {len(rails.rules)} error recovery Rails rules")
    
    # =================================================================
    # RUN ERROR RECOVERY DEMONSTRATION
    # =================================================================
    
    async with rails:
        print(f"\\n🔄 Starting error recovery demonstration...")
        
        # Phase 1: System health monitoring
        print(f"\\n📊 Phase 1: Baseline system health monitoring")
        
        for i in range(3):
            print(f"  🔍 Health check {i + 1}/3")
            health_result = system_health_checker_tool()
            
            messages = [Message(role=Role.USER, content=f"Health check {i + 1} completed")]
            enhanced_messages = await rails.process(messages)
            
            if len(enhanced_messages) > 1:
                for msg in enhanced_messages[1:]:
                    print(f"    💬 Rails: {msg.content}")
            
            await asyncio.sleep(0.2)
        
        # Phase 2: Introduce various errors
        print(f"\\n⚠️ Phase 2: Introducing system errors")
        
        error_scenarios = [
            ("database", "high"),
            ("api_service", "medium"),
            ("cache", "medium"),
            ("database", "critical"),  # Critical error
            ("queue_processor", "high"),
            ("file_storage", "low"),
        ]
        
        for component, severity in error_scenarios:
            print(f"  💥 Simulating {severity} error in {component}")
            error_result = error_simulator_tool(component, severity)
            
            # Test circuit breaker
            circuit_result = circuit_breaker_tool(component, "read_operation")
            
            messages = [Message(role=Role.USER, content=f"Error in {component} ({severity})")]
            enhanced_messages = await rails.process(messages)
            
            if len(enhanced_messages) > 1:
                for msg in enhanced_messages[1:]:
                    print(f"    💬 Rails: {msg.content}")
            
            await asyncio.sleep(0.3)
        
        # Phase 3: Recovery attempts
        print(f"\\n🔧 Phase 3: Recovery operations")
        
        recovery_operations = [
            ("database", "restart"),
            ("api_service", "reset"),
            ("queue_processor", "rollback"),
            ("database", "reset"),  # Second attempt
            ("cache", "restart"),
        ]
        
        for component, action in recovery_operations:
            print(f"  🔄 Attempting {action} recovery for {component}")
            recovery_result = recovery_action_tool(component, action)
            
            messages = [Message(role=Role.USER, content=f"Recovery {action} on {component}")]
            enhanced_messages = await rails.process(messages)
            
            if len(enhanced_messages) > 1:
                for msg in enhanced_messages[1:]:
                    print(f"    💬 Rails: {msg.content}")
            
            # Check for completion
            if any("demo complete" in msg.content.lower() for msg in enhanced_messages if msg.injected_by_rails):
                print("\\n🎉 Demo completion detected!")
                break
            
            await asyncio.sleep(0.2)
        
        # Final health check
        print(f"\\n📊 Final system health assessment")
        final_health = system_health_checker_tool()
        
        messages = [Message(role=Role.USER, content="Final health assessment")]
        final_messages = await rails.process(messages)
        
        if len(final_messages) > 1:
            for msg in final_messages[1:]:
                print(f"  💬 Rails: {msg.content}")
    
    # =================================================================
    # FINAL ERROR RECOVERY METRICS
    # =================================================================
    
    print("\\n📊 Final Error Recovery Metrics:")
    
    # Error statistics
    print(f"  💥 Error Statistics:")
    print(f"    • Total errors: {await rails.store.get_counter('total_errors')}")
    print(f"    • Low severity: {await rails.store.get_counter('errors_low', 0)}")
    print(f"    • Medium severity: {await rails.store.get_counter('errors_medium', 0)}")
    print(f"    • High severity: {await rails.store.get_counter('errors_high', 0)}")
    print(f"    • Critical severity: {await rails.store.get_counter('errors_critical', 0)}")
    
    # Recovery statistics
    print(f"\\n  🔧 Recovery Statistics:")
    print(f"    • Recovery attempts: {await rails.store.get_counter('recovery_attempts')}")
    print(f"    • Recovery successes: {await rails.store.get_counter('recovery_successes', 0)}")
    print(f"    • Recovery failures: {await rails.store.get_counter('recovery_failures', 0)}")
    print(f"    • Emergency recoveries: {await rails.store.get_counter('emergency_recoveries', 0)}")
    print(f"    • Systematic recoveries: {await rails.store.get_counter('systematic_recoveries', 0)}")
    
    # Calculate recovery success rate
    attempts = await rails.store.get_counter('recovery_attempts', 0)
    successes = await rails.store.get_counter('recovery_successes', 0)
    if attempts > 0:
        success_rate = (successes / attempts) * 100
        print(f"    • Recovery success rate: {success_rate:.1f}%")
    
    # Circuit breaker statistics
    print(f"\\n  ⚡ Circuit Breaker Statistics:")
    print(f"    • Circuit breakers tripped: {await rails.store.get_counter('circuit_breakers_tripped', 0)}")
    print(f"    • Circuit breakers closed: {await rails.store.get_counter('circuit_breakers_closed', 0)}")
    
    # System status
    print(f"\\n  🔧 Final System Status:")
    print(f"    • System status: {await rails.store.get('system_status', 'unknown')}")
    print(f"    • Health percentage: {await rails.store.get('system_health_percentage', 'N/A')}%")
    print(f"    • Degraded mode activations: {await rails.store.get_counter('degraded_mode_activations', 0)}")
    print(f"    • Degraded mode active: {await rails.store.get('degraded_mode_active', False)}")
    
    # Component status
    components = ['database', 'api_service', 'cache', 'queue_processor', 'file_storage']
    print(f"\\n  📦 Component Status:")
    for component in components:
        status = await rails.store.get(f'{component}_status', 'unknown')
        error_count = await rails.store.get_counter(f'errors_{component}', 0)
        print(f"    • {component}: {status} ({error_count} errors)")
    
    # Error events queue
    error_events_count = await rails.store.queue_length('error_events')
    print(f"\\n  📝 Error Events: {error_events_count} recorded")
    
    # Rails orchestration metrics
    rails_metrics = await rails.emit_metrics()
    print(f"\\n⚙️ Rails Orchestration:")
    print(f"  • Final state: {rails_metrics['state']}")
    print(f"  • Rules triggered: {rails_metrics['active_rules']}/{rails_metrics['total_rules']}")
    
    print(f"\\n🎯 Error Recovery Features Demonstrated:")
    print(f"  ✅ Multi-level error detection and classification")
    print(f"  ✅ Automatic recovery workflows and strategies")
    print(f"  ✅ Circuit breaker pattern for fault tolerance")
    print(f"  ✅ Degraded mode operations during persistent issues")
    print(f"  ✅ Error pattern analysis and adaptive responses")
    print(f"  ✅ Emergency protocols for critical failures")
    print(f"  ✅ System health monitoring and restoration")
    print(f"  ✅ Recovery success tracking and optimization")
    print(f"  ✅ State-based recovery lifecycle orchestration")
    
    print(f"\\n✨ Error recovery demonstration complete!")


if __name__ == "__main__":
    asyncio.run(demonstrate_error_recovery())