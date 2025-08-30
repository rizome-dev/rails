"""Tests for Rails lifecycle orchestration."""


import pytest

from rails import (
    AlwaysCondition,
    AndCondition,
    # Injectors
    AppendInjector,
    ConditionalInjector,
    InsertInjector,
    LambdaCondition,
    Message,
    NeverCondition,
    NotCondition,
    OrCondition,
    PrependInjector,
    QueueConfig,
    Rails,
    RailState,
    ReplaceInjector,
    Role,
    Store,
    StoreConfig,
    append_message,
    # Conditions
    counter,
    current_rails,
    queue,
    rails_context,
    state,
    system,
    template,
)


class TestRails:
    """Test Rails core functionality."""

    @pytest.mark.asyncio
    async def test_basic_message_injection(self):
        """Test that Rails can inject messages based on conditions."""
        rails = Rails()

        # Create an append injector
        injector = AppendInjector(
            message=Message(
                role=Role.SYSTEM,
                content="Conversation limit reached"
            )
        )

        rails.add_rule(
            condition=counter("turns") >= 3,
            action=injector.inject,
            name="limit_conversation"
        )

        # Set counter to trigger condition
        await rails.store.increment("turns", 3)

        messages = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!")
        ]

        result = await rails.process(messages)

        assert len(result) == 3
        assert result[-1].content == "Conversation limit reached"

    @pytest.mark.asyncio
    async def test_multiple_rules(self):
        """Test multiple rules with different priorities."""
        rails = Rails()

        # High priority rule
        rails.add_rule(
            condition=state("mode") == "debug",
            action=AppendInjector(
                message=Message(
                    role=Role.SYSTEM,
                    content="Debug mode active"
                )
            ).inject,
            priority=10
        )

        # Low priority rule
        rails.add_rule(
            condition=counter("errors") >= 1,
            action=AppendInjector(
                message=Message(
                    role=Role.SYSTEM,
                    content="Errors detected"
                )
            ).inject,
            priority=1
        )

        await rails.store.set("mode", "debug")
        await rails.store.increment("errors", 2)

        messages = [Message(role=Role.USER, content="test")]
        result = await rails.process(messages)

        # Both rules should trigger, high priority first
        assert len(result) == 3
        assert result[1].content == "Debug mode active"
        assert result[2].content == "Errors detected"

    @pytest.mark.asyncio
    async def test_queue_based_orchestration(self):
        """Test queue-based task orchestration."""
        rails = Rails()

        # Add rule for when task queue gets too long
        rails.add_rule(
            condition=queue("tasks").length > 5,
            action=AppendInjector(
                message=Message(
                    role=Role.SYSTEM,
                    content="Too many pending tasks. Focus on completion."
                )
            ).inject
        )

        # Push tasks to queue
        for i in range(6):
            await rails.store.push_queue("tasks", f"task_{i}")

        messages = [Message(role=Role.USER, content="What should I do?")]
        result = await rails.process(messages)

        assert len(result) == 2
        assert "Too many pending tasks" in result[-1].content

    @pytest.mark.asyncio
    async def test_workflow_rules(self):
        """Test rules that trigger workflows instead of message injection."""
        rails = Rails()

        workflow_executed = False

        async def cleanup_workflow(messages):
            nonlocal workflow_executed
            workflow_executed = True
            # Access Rails from context
            r = current_rails()
            # Pop all tasks from queue
            while await r.store.pop_queue("tasks"):
                pass
            return messages

        rails.add_rule(
            condition=counter("failures") > 3,
            action=cleanup_workflow,
            name="cleanup_on_failures"
        )

        # Set context for current_rails() to work
        token = rails_context.set(rails)
        try:
            await rails.store.increment("failures", 4)
            await rails.store.push_queue("tasks", "task1")
            await rails.store.push_queue("tasks", "task2")

            messages = [Message(role=Role.USER, content="test")]
            await rails.process(messages)

            assert workflow_executed
            assert await rails.store.queue_length("tasks") == 0
        finally:
            rails_context.reset(token)

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test Rails as context manager."""
        async with Rails() as rails:
            rails.add_rule(
                condition=counter("calls") >= 1,
                action=AppendInjector(
                    message=Message(
                        role=Role.SYSTEM,
                        content="First call"
                    )
                ).inject
            )

            await rails.store.increment("calls")

            messages = [Message(role=Role.USER, content="test")]
            result = await rails.process(messages)

            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_current_rails_context(self):
        """Test current_rails() context variable."""

        # Should raise when no Rails active
        with pytest.raises(RuntimeError):
            current_rails()

        async with Rails() as rails:
            # Should work within context
            r = current_rails()
            assert r is rails

            # Test from simulated tool
            async def my_tool():
                r = current_rails()
                await r.store.increment("tool_calls")
                return "Tool executed"

            result = await my_tool()
            assert result == "Tool executed"
            assert await rails.store.get_counter("tool_calls") == 1

    @pytest.mark.asyncio
    async def test_message_replacement(self):
        """Test message replacement injector."""
        rails = Rails()

        rails.add_rule(
            condition=state("reset") == True,
            action=ReplaceInjector(
                messages=[
                    Message(role=Role.SYSTEM, content="Conversation reset"),
                    Message(role=Role.ASSISTANT, content="How can I help you?")
                ]
            ).inject
        )

        await rails.store.set("reset", True)

        messages = [
            Message(role=Role.USER, content="Old message 1"),
            Message(role=Role.ASSISTANT, content="Old response")
        ]

        result = await rails.process(messages)

        assert len(result) == 2
        assert result[0].content == "Conversation reset"
        assert result[1].content == "How can I help you?"

    @pytest.mark.asyncio
    async def test_conditional_injector(self):
        """Test conditional message injection."""
        rails = Rails()

        injector = ConditionalInjector(
            condition=counter("errors") > 2,
            injector=AppendInjector(
                message=Message(
                    role=Role.SYSTEM,
                    content="Multiple errors detected"
                )
            )
        )

        rails.add_rule(
            condition=AlwaysCondition(),
            action=injector.inject
        )

        # Should not inject initially
        messages = [Message(role=Role.USER, content="test")]
        result = await rails.process(messages)
        assert len(result) == 1

        # Trigger condition
        await rails.store.increment("errors", 3)
        result = await rails.process(messages)
        assert len(result) == 2
        assert "Multiple errors" in result[-1].content

    @pytest.mark.asyncio
    async def test_system_injector_helper(self):
        """Test system message helper function."""
        rails = Rails()

        rails.add_rule(
            condition=counter("warnings") >= 3,
            action=system("Please be more careful with your inputs")
        )

        await rails.store.increment("warnings", 3)

        messages = [Message(role=Role.USER, content="test")]
        result = await rails.process(messages)

        assert len(result) == 2
        assert result[-1].role == Role.SYSTEM
        assert "be more careful" in result[-1].content

    @pytest.mark.asyncio
    async def test_template_injector(self):
        """Test template message injection."""
        rails = Rails()

        rails.add_rule(
            condition=state("user_name").exists,
            action=template("Hello {user_name}, how can I help you today?")
        )

        await rails.store.set("user_name", "Alice")

        messages = []
        result = await rails.process(messages)

        assert len(result) == 1
        assert "Hello Alice" in result[0].content


class TestConditions:
    """Test condition system."""

    @pytest.mark.asyncio
    async def test_fluent_counter_conditions(self):
        """Test fluent counter condition builders."""
        store = Store()

        # Test greater than or equal
        cond = counter("turns") >= 3
        assert not await cond.evaluate(store)

        await store.increment("turns", 3)
        assert await cond.evaluate(store)

        # Test greater than
        cond = counter("errors") > 2
        await store.increment("errors", 2)
        assert not await cond.evaluate(store)

        await store.increment("errors", 1)
        assert await cond.evaluate(store)

        # Test less than
        cond = counter("limit") < 10
        assert await cond.evaluate(store)

        await store.increment("limit", 10)
        assert not await cond.evaluate(store)

    @pytest.mark.asyncio
    async def test_fluent_state_conditions(self):
        """Test fluent state condition builders."""
        store = Store()

        # Test equality
        cond = state("mode") == "debug"
        assert not await cond.evaluate(store)

        await store.set("mode", "debug")
        assert await cond.evaluate(store)

        # Test inequality
        cond = state("status") != "error"
        assert await cond.evaluate(store)  # True when not set

        await store.set("status", "error")
        assert not await cond.evaluate(store)

        # Test exists
        cond = state("user_id").exists
        assert not await cond.evaluate(store)

        await store.set("user_id", "123")
        assert await cond.evaluate(store)

    @pytest.mark.asyncio
    async def test_fluent_queue_conditions(self):
        """Test fluent queue condition builders."""
        store = Store()

        # Test length comparison
        cond = queue("tasks").length > 3
        assert not await cond.evaluate(store)

        for i in range(4):
            await store.push_queue("tasks", f"task_{i}")
        assert await cond.evaluate(store)

        # Test is_empty
        cond = queue("errors").is_empty
        assert await cond.evaluate(store)

        await store.push_queue("errors", "error1")
        assert not await cond.evaluate(store)

        # Test contains
        cond = queue("tasks").contains("task_2")
        assert await cond.evaluate(store)

        cond = queue("tasks").contains("task_99")
        assert not await cond.evaluate(store)

    @pytest.mark.asyncio
    async def test_composite_conditions(self):
        """Test AND, OR, NOT composite conditions."""
        store = Store()

        await store.increment("count", 5)
        await store.set("mode", "test")

        # Test AND
        and_cond = AndCondition(
            counter("count") >= 5,
            state("mode") == "test"
        )
        assert await and_cond.evaluate(store)

        and_cond = AndCondition(
            counter("count") >= 10,
            state("mode") == "test"
        )
        assert not await and_cond.evaluate(store)

        # Test OR
        or_cond = OrCondition(
            counter("count") >= 10,
            state("mode") == "test"
        )
        assert await or_cond.evaluate(store)

        # Test NOT
        not_cond = NotCondition(state("mode") == "production")
        assert await not_cond.evaluate(store)

    @pytest.mark.asyncio
    async def test_lambda_condition(self):
        """Test lambda-based custom conditions."""
        store = Store()

        # Complex custom condition
        async def complex_check(s: Store) -> bool:
            c = await s.get_counter("count")
            m = await s.get("mode")
            return c > 5 and m == "active"

        cond = LambdaCondition(func=complex_check)

        assert not await cond.evaluate(store)

        await store.increment("count", 6)
        await store.set("mode", "active")
        assert await cond.evaluate(store)

    @pytest.mark.asyncio
    async def test_always_never_conditions(self):
        """Test AlwaysCondition and NeverCondition."""
        store = Store()

        always = AlwaysCondition()
        assert await always.evaluate(store)

        never = NeverCondition()
        assert not await never.evaluate(store)


class TestInjectors:
    """Test message injectors."""

    @pytest.mark.asyncio
    async def test_append_injector(self):
        """Test appending messages."""
        injector = AppendInjector(
            message=Message(role=Role.SYSTEM, content="Appended")
        )

        messages = [Message(role=Role.USER, content="Hello")]
        result = await injector.inject(messages)

        assert len(result) == 2
        assert result[-1].content == "Appended"

    @pytest.mark.asyncio
    async def test_prepend_injector(self):
        """Test prepending messages."""
        injector = PrependInjector(
            message=Message(role=Role.SYSTEM, content="Prepended")
        )

        messages = [Message(role=Role.USER, content="Hello")]
        result = await injector.inject(messages)

        assert len(result) == 2
        assert result[0].content == "Prepended"

    @pytest.mark.asyncio
    async def test_insert_injector(self):
        """Test inserting messages at specific index."""
        injector = InsertInjector(
            message=Message(role=Role.SYSTEM, content="Inserted"),
            index=1
        )

        messages = [
            Message(role=Role.USER, content="First"),
            Message(role=Role.ASSISTANT, content="Last")
        ]
        result = await injector.inject(messages)

        assert len(result) == 3
        assert result[1].content == "Inserted"

    @pytest.mark.asyncio
    async def test_replace_injector(self):
        """Test replacing all messages."""
        injector = ReplaceInjector(
            messages=[
                Message(role=Role.SYSTEM, content="New 1"),
                Message(role=Role.ASSISTANT, content="New 2")
            ]
        )

        messages = [
            Message(role=Role.USER, content="Old 1"),
            Message(role=Role.ASSISTANT, content="Old 2"),
            Message(role=Role.USER, content="Old 3")
        ]
        result = await injector.inject(messages)

        assert len(result) == 2
        assert result[0].content == "New 1"
        assert result[1].content == "New 2"


class TestStore:
    """Test Rails store functionality."""

    @pytest.mark.asyncio
    async def test_counter_operations(self):
        """Test counter operations."""
        store = Store()

        # Test increment
        result = await store.increment("test", 5)
        assert result == 5

        # Test get
        value = await store.get_counter("test")
        assert value == 5

        # Test reset
        await store.reset_counter("test")
        value = await store.get_counter("test")
        assert value == 0

    @pytest.mark.asyncio
    async def test_state_operations(self):
        """Test state operations."""
        store = Store()

        # Test set/get
        await store.set("key", "value")
        value = await store.get("key")
        assert value == "value"

        # Test delete
        deleted = await store.delete("key")
        assert deleted

        value = await store.get("key", "default")
        assert value == "default"

    @pytest.mark.asyncio
    async def test_queue_operations(self):
        """Test queue operations."""
        store = Store()

        # Test push/pop
        await store.push_queue("tasks", "task1")
        await store.push_queue("tasks", "task2")

        assert await store.queue_length("tasks") == 2

        item = await store.pop_queue("tasks")
        assert item == "task1"  # FIFO by default

        assert await store.queue_length("tasks") == 1

        # Test get_queue
        await store.push_queue("tasks", "task3")
        items = await store.get_queue("tasks")
        assert items == ["task2", "task3"]

        # Test clear
        await store.clear_queue("tasks")
        assert await store.queue_length("tasks") == 0

    @pytest.mark.asyncio
    async def test_queue_config(self):
        """Test queue configuration."""
        config = StoreConfig(
            default_queues={
                "limited": QueueConfig(max_size=2),
                "dedup": QueueConfig(auto_dedup=True),
                "lifo": QueueConfig(fifo=False)
            }
        )
        store = Store(config=config)

        # Test max_size
        await store.push_queue("limited", "item1")
        await store.push_queue("limited", "item2")
        await store.push_queue("limited", "item3")  # Should remove item1

        items = await store.get_queue("limited")
        assert len(items) == 2
        assert "item1" not in items

        # Test auto_dedup
        await store.push_queue("dedup", "duplicate")
        await store.push_queue("dedup", "duplicate")
        assert await store.queue_length("dedup") == 1

        # Test LIFO
        await store.push_queue("lifo", "first")
        await store.push_queue("lifo", "second")
        item = await store.pop_queue("lifo")
        assert item == "second"  # LIFO pops last item

    @pytest.mark.asyncio
    async def test_event_emission(self):
        """Test event emission and streaming."""
        store = Store()
        events = []

        async def collector(event):
            events.append(event)

        store.subscribe_events(collector)

        # Trigger some events
        await store.increment("counter", 1, triggered_by="test")
        await store.set("key", "value", triggered_by="test")
        await store.push_queue("queue", "item", triggered_by="test")

        # Events should be collected
        assert len(events) == 3
        assert events[0].event_type == "counter_increment"
        assert events[1].event_type == "state_set"
        assert events[2].event_type == "queue_push"

        store.unsubscribe_events(collector)

    @pytest.mark.asyncio
    async def test_snapshot_and_clear(self):
        """Test snapshot and clear operations."""
        store = Store()

        # Add some data
        await store.increment("counter", 5)
        await store.set("state", "active")
        await store.push_queue("tasks", "task1")

        # Get snapshot
        snapshot = await store.get_snapshot()
        assert snapshot["counters"]["counter"] == 5
        assert snapshot["state"]["state"] == "active"
        assert "task1" in snapshot["queues"]["tasks"]

        # Clear all
        await store.clear()

        # Verify cleared
        assert await store.get_counter("counter") == 0
        assert await store.get("state") is None
        assert await store.queue_length("tasks") == 0


class TestMiddleware:
    """Test middleware functionality."""

    @pytest.mark.asyncio
    async def test_middleware_stack(self):
        """Test processing through middleware stack."""
        rails = Rails()

        # Add logging middleware
        async def logging_middleware(messages, store):
            await store.increment("middleware_calls")
            return messages

        # Add injection middleware
        async def injection_middleware(messages, store):
            if await store.get_counter("inject_header") > 0:
                messages.insert(0, Message(
                    role=Role.SYSTEM,
                    content="Middleware header"
                ))
            return messages

        rails.add_middleware(logging_middleware)
        rails.add_middleware(injection_middleware)

        await rails.store.increment("inject_header")

        messages = [Message(role=Role.USER, content="test")]
        result = await rails.process_with_middleware(messages)

        assert await rails.store.get_counter("middleware_calls") == 1
        assert len(result) == 2
        assert result[0].content == "Middleware header"


class TestRuleManagement:
    """Test rule management features."""

    @pytest.mark.asyncio
    async def test_rule_enable_disable(self):
        """Test enabling and disabling rules."""
        rails = Rails()

        rails.add_rule(
            condition=AlwaysCondition(),
            action=append_message(Message(role=Role.SYSTEM, content="Always")),
            name="always_rule"
        )

        # Rule should be active
        messages = []
        result = await rails.process(messages)
        assert len(result) == 1

        # Disable rule
        rails.disable_rule("always_rule")
        result = await rails.process(messages)
        assert len(result) == 0

        # Re-enable rule
        rails.enable_rule("always_rule")
        result = await rails.process(messages)
        assert len(result) == 1

    def test_clear_rules(self):
        """Test clearing all rules."""
        rails = Rails()

        rails.add_rule(
            condition=AlwaysCondition(),
            action=lambda m: m,
            name="rule1"
        )
        rails.add_rule(
            condition=NeverCondition(),
            action=lambda m: m,
            name="rule2"
        )

        assert len(rails.rules) == 2

        rails.clear_rules()
        assert len(rails.rules) == 0

    def test_get_active_rules(self):
        """Test getting active rules."""
        rails = Rails()

        rails.add_rule(
            condition=AlwaysCondition(),
            action=lambda m: m,
            name="rule1"
        )
        rails.add_rule(
            condition=NeverCondition(),
            action=lambda m: m,
            name="rule2"
        )

        assert len(rails.get_active_rules()) == 2

        rails.disable_rule("rule1")
        assert len(rails.get_active_rules()) == 1


class TestMetrics:
    """Test metrics and observability."""

    @pytest.mark.asyncio
    async def test_emit_metrics(self):
        """Test metrics emission."""
        rails = Rails()

        rails.add_rule(
            condition=AlwaysCondition(),
            action=lambda m: m,
            name="rule1"
        )
        rails.add_middleware(lambda m, s: m)

        await rails.store.increment("counter", 5)
        await rails.store.set("key", "value")

        metrics = await rails.emit_metrics()

        assert metrics["state"] == RailState.INITIALIZED.value
        assert metrics["total_rules"] == 1
        assert metrics["active_rules"] == 1
        assert metrics["middleware_count"] == 1
        assert metrics["store_snapshot"]["counters"]["counter"] == 5
        assert metrics["store_snapshot"]["state"]["key"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
