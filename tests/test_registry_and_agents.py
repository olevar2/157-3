"""
Tests for registry validation and AI agents registry
"""

import pytest
from engines.ai_enhancement.registry import (
    INDICATOR_REGISTRY, 
    validate_registry,
    AI_AGENTS_REGISTRY as AGENT_REGISTRY, 
    validate_ai_agents as validate_agents
)

def test_indicator_registry_size_and_callable():
    """Test that indicator registry contains exactly 157 callable indicators"""
    # validate_registry should return the number of real indicators
    count = validate_registry()
    assert isinstance(count, int)
    assert count == len(INDICATOR_REGISTRY) == 157
    # All entries must be callable
    for name, cls in INDICATOR_REGISTRY.items():
        assert callable(cls), f"Indicator {name} is not callable"

def test_ai_agents_registry_loaded_and_valid():
    """Test that AI agents registry contains exactly 9 properly configured agents"""
    # validate_agents should return number of loaded agents
    agent_count = validate_agents()
    assert isinstance(agent_count, int)
    assert agent_count == len(AGENT_REGISTRY) == 9
    # Each agent entry must have required keys
    for agent_name, config in AGENT_REGISTRY.items():
        assert "model" in config and "max_tokens" in config, \
            f"Agent {agent_name} missing configuration keys"

def test_indicator_registry_no_dummies():
    """Test that no dummy indicators exist in the registry"""
    for name, cls in INDICATOR_REGISTRY.items():
        # Check that it's not a dummy by name
        if hasattr(cls, '__name__'):
            assert 'dummy' not in cls.__name__.lower(), \
                f"Found dummy indicator '{name}' in registry"

def test_ai_agents_have_required_fields():
    """Test that all AI agents have the required configuration fields"""
    required_fields = ['type', 'class', 'description', 'specialization', 'indicators_used', 'status']
    
    for agent_name, config in AGENT_REGISTRY.items():
        for field in required_fields:
            assert field in config, \
                f"Agent {agent_name} missing required field '{field}'"
        
        # Check that status is active
        assert config['status'] == 'active', \
            f"Agent {agent_name} is not active"
        
        # Check that indicators_used is a positive number
        assert isinstance(config['indicators_used'], int), \
            f"Agent {agent_name} indicators_used should be an integer"
        assert config['indicators_used'] > 0, \
            f"Agent {agent_name} should use at least 1 indicator"

def test_registry_functions_return_correct_types():
    """Test that registry functions return the expected types"""
    # validate_registry should return an integer
    indicator_count = validate_registry()
    assert isinstance(indicator_count, int)
    assert indicator_count > 0
    
    # validate_agents should return an integer count
    agent_validation_result = validate_agents()
    assert isinstance(agent_validation_result, int)
    assert agent_validation_result > 0

def test_agent_specializations_are_unique():
    """Test that each agent has a unique specialization"""
    specializations = [config['specialization'] for config in AGENT_REGISTRY.values()]
    assert len(specializations) == len(set(specializations)), \
        "Agent specializations should be unique"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
