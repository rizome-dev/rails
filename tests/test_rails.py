"""Test suite for Rails message injection functionality."""

import pytest
import asyncio

from rails import (
    Rails, Store, Message, 
    CounterCondition, StateCondition, LambdaCondition,
    AppendInjector, PrependInjector, ReplaceInjector,
    counter_at_least, state_equals
)
from rails.conditions import AndCondition, OrCondition, NotCondition


class TestRails:
    """Test cases for Rails core class."""
    
    def test_rails_initialization(self):
        """Test Rails can be initialized."""
        rails = Rails()
        assert rails.store is not None
        assert isinstance(rails.store, Store)
        assert rails.rule_count() == 0
        
    def test_when_inject_pattern(self):
        """Test basic when().inject() pattern."""
        rails = Rails()
        message = {"role": "system", "content": "Help the user"}
        
        # Should return self for chaining
        result = rails.when(lambda s: s.get_counter_sync('turns') >= 3).inject(message)
        assert result is rails
        assert rails.rule_count() == 1
        
    def test_invalid_inject_without_when(self):
        """Test that inject() fails without when()."""
        rails = Rails()
        message = {"role": "system", "content": "Help"}
        
        with pytest.raises(ValueError, match="inject\\(\\) must be called after when\\(\\)"):
            rails.inject(message)
            
    @pytest.mark.asyncio
    async def test_basic_message_injection(self):
        """Test basic message injection when condition is met."""
        rails = Rails()
        message = {"role": "system", "content": "Injected message"}
        
        # Set up condition and injection
        rails.when(lambda s: s.get_counter_sync('turns') >= 2).inject(message)
        
        # Set counter to meet condition
        rails.store.increment_sync('turns', 2)
        
        # Test injection
        original_messages = [{"role": "user", "content": "Hello"}]
        result = await rails.check(original_messages)
        
        assert len(result) == 2
        assert result[0] == original_messages[0]
        assert result[1] == message
        
    @pytest.mark.asyncio
    async def test_no_injection_when_condition_not_met(self):
        """Test no injection when condition is not met."""
        rails = Rails()
        message = {"role": "system", "content": "Should not appear"}
        
        rails.when(lambda s: s.get_counter_sync('turns') >= 5).inject(message)
        
        # Counter is 0, condition not met
        original_messages = [{"role": "user", "content": "Hello"}]
        result = await rails.check(original_messages)
        
        assert result == original_messages
        
    @pytest.mark.asyncio
    async def test_multiple_rules(self):
        """Test multiple injection rules."""
        rails = Rails()
        
        msg1 = {"role": "system", "content": "First injection"}
        msg2 = {"role": "assistant", "content": "Second injection"}
        
        rails.when(lambda s: s.get_counter_sync('turns') >= 2).inject(msg1)
        rails.when(lambda s: s.get_sync('mode') == 'help').inject(msg2, strategy='prepend')
        
        # Meet both conditions
        rails.store.increment_sync('turns', 3)
        rails.store.set_sync('mode', 'help')
        
        original_messages = [{"role": "user", "content": "Hello"}]
        result = await rails.check(original_messages)
        
        # Should have: msg2 (prepended), original message, msg1 (appended)
        assert len(result) == 3
        assert result[0] == msg2
        assert result[1] == original_messages[0]
        assert result[2] == msg1
        
    def test_clear_rules(self):
        """Test clearing injection rules."""
        rails = Rails()
        message = {"role": "system", "content": "Test"}
        
        rails.when(lambda s: True).inject(message)
        assert rails.rule_count() == 1
        
        rails.clear_rules()
        assert rails.rule_count() == 0
        
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test Rails as context manager."""
        async with Rails() as rails:
            message = {"role": "system", "content": "Test"}
            rails.when(lambda s: True).inject(message)
            
            # Should work inside context
            result = await rails.check([])
            assert len(result) == 1
            assert result[0] == message


class TestConditions:
    """Test cases for condition system."""
    
    def test_counter_condition(self):
        """Test counter-based conditions."""
        store = Store()
        condition = CounterCondition("test_counter", 5)
        
        # Counter is 0, condition should fail
        assert not condition(store)
        
        # Increment counter to meet condition
        store.increment_sync("test_counter", 5)
        assert condition(store)
        
        # Test different comparison operators
        gt_condition = CounterCondition("test_counter", 3, ">")
        assert gt_condition(store)
        
        eq_condition = CounterCondition("test_counter", 5, "==")
        assert eq_condition(store)
        
    def test_state_condition(self):
        """Test state-based conditions."""
        store = Store()
        condition = StateCondition("status", "active")
        
        # No state set, condition should fail
        assert not condition(store)
        
        # Set matching state
        store.set_sync("status", "active")
        assert condition(store)
        
        # Set different state
        store.set_sync("status", "inactive")
        assert not condition(store)
        
    def test_lambda_condition(self):
        """Test lambda-based conditions."""
        store = Store()
        
        # Simple lambda condition
        condition = LambdaCondition(lambda s: s.get_counter_sync('count') > 3)
        
        assert not condition(store)
        
        store.increment_sync('count', 5)
        assert condition(store)
        
    def test_convenience_condition_functions(self):
        """Test convenience functions for common conditions."""
        store = Store()
        
        # Test counter_at_least
        condition = counter_at_least("turns", 3)
        assert not condition(store)
        
        store.increment_sync("turns", 3)
        assert condition(store)
        
        # Test state_equals
        condition = state_equals("mode", "debug")
        assert not condition(store)
        
        store.set_sync("mode", "debug")
        assert condition(store)


class TestInjectors:
    """Test cases for injection strategies."""
    
    def test_append_injector(self):
        """Test append injection strategy."""
        injector = AppendInjector()
        messages = [{"role": "user", "content": "Hello"}]
        new_message = {"role": "assistant", "content": "Hi there"}
        
        result = injector.inject(messages, new_message)
        
        assert len(result) == 2
        assert result[0] == messages[0]
        assert result[1] == new_message
        
    def test_prepend_injector(self):
        """Test prepend injection strategy."""
        injector = PrependInjector()
        messages = [{"role": "user", "content": "Hello"}]
        new_message = {"role": "system", "content": "System message"}
        
        result = injector.inject(messages, new_message)
        
        assert len(result) == 2
        assert result[0] == new_message
        assert result[1] == messages[0]
        
    def test_replace_injector(self):
        """Test replace injection strategies."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        new_message = {"role": "system", "content": "Replaced"}
        
        # Test replace_last
        injector = ReplaceInjector(replace_last=True)
        result = injector.inject(messages, new_message)
        assert len(result) == 2
        assert result[0] == messages[0]
        assert result[1] == new_message
        
        # Test replace_all
        injector = ReplaceInjector(replace_all=True)
        result = injector.inject(messages, new_message)
        assert len(result) == 1
        assert result[0] == new_message


class TestStore:
    """Test cases for Rails store."""
    
    @pytest.mark.asyncio
    async def test_async_counter_operations(self):
        """Test async counter operations."""
        store = Store()
        
        # Test async increment
        result = await store.increment("test_counter", 3)
        assert result == 3
        
        # Test async get
        value = await store.get_counter("test_counter")
        assert value == 3
        
        # Test async set
        await store.set_counter("test_counter", 10)
        value = await store.get_counter("test_counter")
        assert value == 10
        
    @pytest.mark.asyncio
    async def test_async_state_operations(self):
        """Test async state operations."""
        store = Store()
        
        # Test async set/get
        await store.set("key1", "value1")
        value = await store.get("key1")
        assert value == "value1"
        
        # Test default value
        value = await store.get("nonexistent", "default")
        assert value == "default"
        
        # Test exists
        exists = await store.exists("key1")
        assert exists is True
        
        # Test delete
        deleted = await store.delete("key1")
        assert deleted is True
        
        exists = await store.exists("key1")
        assert exists is False
        
    def test_sync_methods(self):
        """Test synchronous store methods."""
        store = Store()
        
        # Counter operations
        store.increment_sync("counter", 5)
        assert store.get_counter_sync("counter") == 5
        
        # State operations  
        store.set_sync("key", "value")
        assert store.get_sync("key") == "value"
        assert store.get_sync("missing", "default") == "default"


class TestConvenienceMethods:
    """Test convenience methods on Rails."""
    
    @pytest.mark.asyncio
    async def test_on_counter_convenience(self):
        """Test on_counter convenience method."""
        rails = Rails()
        message = {"role": "system", "content": "Counter reached threshold"}
        
        rails.on_counter("turns", 3, message)
        assert rails.rule_count() == 1
        
        # Test injection
        rails.store.increment_sync("turns", 3)
        result = await rails.check([])
        
        assert len(result) == 1
        assert result[0] == message
        
    @pytest.mark.asyncio
    async def test_on_state_convenience(self):
        """Test on_state convenience method."""
        rails = Rails()
        message = {"role": "assistant", "content": "State changed to debug"}
        
        rails.on_state("mode", "debug", message)
        assert rails.rule_count() == 1
        
        # Test injection
        rails.store.set_sync("mode", "debug")
        result = await rails.check([])
        
        assert len(result) == 1
        assert result[0] == message


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_empty_message_list(self):
        """Test Rails with empty message list."""
        rails = Rails()
        message = {"role": "system", "content": "Empty list injection"}
        
        rails.when(lambda s: True).inject(message)
        
        result = await rails.check([])
        assert len(result) == 1
        assert result[0] == message
        
    @pytest.mark.asyncio
    async def test_none_message_handling(self):
        """Test Rails handling None messages gracefully."""
        rails = Rails()
        
        # Should not crash with None in message list
        messages = [{"role": "user", "content": "Hello"}, None, {"role": "user", "content": "World"}]
        result = await rails.check(messages)
        
        # Should preserve original list structure
        assert len(result) == 3
        assert result[1] is None
        
    @pytest.mark.asyncio
    async def test_malformed_messages(self):
        """Test Rails with various malformed message formats."""
        rails = Rails()
        message = {"role": "system", "content": "Recovery message"}
        
        rails.when(lambda s: True).inject(message)
        
        # Test various malformed formats
        malformed_messages = [
            "string_message",
            123,
            {"role": "user"},  # Missing content
            {"content": "Hello"},  # Missing role
            {},  # Empty dict
            {"role": "user", "content": None},  # None content
        ]
        
        result = await rails.check(malformed_messages)
        
        # Should still inject message
        assert len(result) == len(malformed_messages) + 1
        assert result[-1] == message
        
            
    def test_invalid_injection_strategies(self):
        """Test invalid injection strategy names."""
        rails = Rails()
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            rails.when(lambda s: True).inject(
                {"role": "system", "content": "Test"}, 
                strategy="invalid_strategy"
            )
            
    @pytest.mark.asyncio
    async def test_condition_evaluation_errors(self):
        """Test Rails handles condition evaluation errors gracefully."""
        rails = Rails()
        
        def error_condition(store):
            raise ValueError("Condition failed")
            
        def working_condition(store):
            return store.get_counter_sync("test") >= 1
            
        # Add both failing and working conditions
        rails.when(error_condition).inject({"role": "system", "content": "Should not appear"})
        rails.when(working_condition).inject({"role": "system", "content": "This works"})
        
        rails.store.increment_sync("test", 1)
        
        messages = [{"role": "user", "content": "Test"}]
        result = await rails.check(messages)
        
        # Should only see successful injection
        assert len(result) == 2
        assert result[1]["content"] == "This works"
        
        
        
        
        
    def test_rule_count_consistency(self):
        """Test rule count remains consistent."""
        rails = Rails()
        
        assert rails.rule_count() == 0
        
        # Add rules
        for i in range(5):
            rails.when(lambda s, i=i: s.get_counter_sync(f"test_{i}") >= 1).inject(
                {"role": "system", "content": f"Message {i}"}
            )
            assert rails.rule_count() == i + 1
            
        # Clear rules
        rails.clear_rules()
        assert rails.rule_count() == 0
        
        # Should be able to add more rules after clearing
        rails.when(lambda s: True).inject({"role": "system", "content": "After clear"})
        assert rails.rule_count() == 1
        
    def test_condition_helper_edge_cases(self):
        """Test condition helper functions with edge cases."""
        store = Store()
        
        # Test counter_at_least with zero and negative values
        condition = counter_at_least("test_counter", 0)
        assert condition(store) is True  # 0 >= 0
        
        condition = counter_at_least("test_counter", -5)
        assert condition(store) is True  # 0 >= -5
        
        # Test state_equals with unusual values
        store.set_sync("test_state", None)
        condition = state_equals("test_state", None)
        assert condition(store) is True
        
        store.set_sync("test_state", "")
        condition = state_equals("test_state", "")
        assert condition(store) is True
        
        # Test with non-string values
        store.set_sync("numeric_state", 42)
        condition = state_equals("numeric_state", 42)
        assert condition(store) is True
        
    @pytest.mark.asyncio
    async def test_injector_edge_cases(self):
        """Test injector edge cases."""
        # Test replace with empty message list
        replace_injector = ReplaceInjector(replace_last=True)
        result = replace_injector.inject([], {"role": "system", "content": "Replace in empty"})
        assert len(result) == 1
        assert result[0]["content"] == "Replace in empty"
        
        # Test replace all with empty list
        replace_all_injector = ReplaceInjector(replace_all=True)
        result = replace_all_injector.inject([], {"role": "system", "content": "Replace all empty"})
        assert len(result) == 1
        
        # Test append with None message
        append_injector = AppendInjector()
        messages = [{"role": "user", "content": "Hello"}]
        result = append_injector.inject(messages, None)
        assert len(result) == 2
        assert result[1] is None
        
        


class TestComplexConditionCombinations:
    """Test basic combinations of condition types."""
    
    def test_nested_conditions(self):
        """Test basic AND/OR/NOT combinations."""
        store = Store()
        
        # Set up test data
        store.increment_sync("counter1", 5)
        store.set_sync("state1", "active")
        
        # Simple nested condition: counter1 >= 3 AND state1 == 'active'
        and_condition = AndCondition(
            CounterCondition("counter1", 3, ">="),
            StateCondition("state1", "active")
        )
        
        # Should be True: 5 >= 3 AND 'active' == 'active'
        assert and_condition(store) is True
        
        # Test OR condition
        or_condition = OrCondition(
            CounterCondition("counter1", 10, ">="),  # False: 5 >= 10
            StateCondition("state1", "active")       # True: 'active' == 'active'
        )
        assert or_condition(store) is True
        
        # Test NOT condition
        not_condition = NotCondition(StateCondition("state1", "inactive"))
        assert not_condition(store) is True  # NOT ('active' == 'inactive')
        

class TestStoreOperations:
    """Test Store operations and data types."""
    
    def test_store_data_types(self):
        """Test Store with basic data types."""
        store = Store()
        
        test_values = [
            ("string", "test_string"),
            ("integer", 42),
            ("boolean", True),
            ("none", None),
        ]
        
        # Test sync operations
        for key, value in test_values:
            store.set_sync(key, value)
            retrieved = store.get_sync(key)
            assert retrieved == value
            
    @pytest.mark.asyncio
    async def test_store_cleanup(self):
        """Test Store cleanup operations."""
        store = Store()
        
        # Add some data
        await store.set("test_key", "test_value")
        await store.increment("test_counter", 5)
            
        # Verify data exists
        assert await store.get("test_key") == "test_value"
        assert await store.get_counter("test_counter") == 5
        
        # Clear all data
        await store.clear()
        
        # Verify data is gone
        assert await store.get("test_key", "default") == "default"
        assert await store.get_counter("test_counter", 0) == 0
        assert await store.exists("test_key") is False


if __name__ == "__main__":
    pytest.main([__file__])