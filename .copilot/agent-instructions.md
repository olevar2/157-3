# Platform3 Copilot Agent Instructions

## Platform Overview
Platform3 is a sophisticated trading and analysis platform with the following key characteristics:

### Core Components
- **157+ Technical Indicators** - Comprehensive trading analysis tools
- **MCP Coordination System** - Model Context Protocol for agent communication
- **Agent Bridge Architecture** - Coordinated multi-agent system
- **Python-Based Core** - Main logic in Python with TypeScript components
- **Real-time Analysis** - Live trading data processing and validation

### Key Patterns & Conventions

#### Error Handling Pattern
```python
try:
    # Operation logic
    result = perform_operation()
    logger.info(f"Operation completed successfully")
    return {'status': 'success', 'data': result}
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    return {'status': 'error', 'message': str(e)}
except Exception as e:
    logger.critical(f"Unexpected error: {e}")
    raise
```

#### Logging Convention
```python
import logging
logger = logging.getLogger(__name__)
# Use: logger.info(), logger.error(), logger.warning(), logger.debug()
```

#### MCP Handler Pattern
```python
async def handle_mcp_operation(self, params):
    """Handle MCP operation with proper validation."""
    try:
        # Validate inputs
        if not self.validate_params(params):
            return {'error': 'Invalid parameters'}
        
        # Process operation
        result = await self.process(params)
        
        # Return standardized response
        return {'status': 'success', 'data': result}
    except Exception as e:
        logger.error(f'MCP operation failed: {e}')
        return {'status': 'error', 'message': str(e)}
```

#### Indicator Validation Pattern
```python
def validate_indicator(self):
    """Validate indicator configuration and data."""
    checks = [
        self.initialized,
        len(self.data) > 0,
        self.config is not None,
        hasattr(self, 'calculate')
    ]
    return all(checks)
```

### Critical Files & Directories
- `comprehensive_validation_test.py` - Main validation system
- `copilot_mcp_initializer.py` - MCP coordination setup
- `analyze_indicators.py` - Indicator analysis tools
- `analyze_constructor_errors.py` - Error detection system
- Agent bridge files (pattern: `*_bridge.py`)
- Indicator files (pattern: `*_indicator.py`)

### Agent Instructions

#### When Fixing Errors:
1. **Always check MCP coordination impact** - Changes might affect agent communication
2. **Validate indicator calculations** - Ensure trading logic remains accurate
3. **Follow Platform3 error handling patterns** - Use established logging and error patterns
4. **Consider performance implications** - Platform3 processes real-time data
5. **Maintain backward compatibility** - Don't break existing indicator integrations

#### When Adding Features:
1. **Use existing architectural patterns** - Follow bridge and indicator patterns
2. **Add proper validation** - Include comprehensive error checking
3. **Document with docstrings** - Maintain code documentation standards
4. **Add logging** - Include appropriate log levels for debugging
5. **Consider MCP integration** - How does this interact with other agents?

#### When Analyzing Code:
1. **Focus on data flow** - Understand how indicators feed into analysis
2. **Check agent coordination** - Verify MCP communication patterns
3. **Validate error handling** - Ensure robust error management
4. **Review performance** - Look for optimization opportunities
5. **Assess maintainability** - Consider long-term code health

### Common Issues & Solutions

#### Constructor Errors
- Usually related to missing parameters or incorrect initialization
- Check `analyze_constructor_errors.py` for patterns
- Ensure all required attributes are initialized

#### Indicator Validation Failures
- Verify data integrity and calculation logic
- Check for proper error handling in calculation methods
- Ensure indicator follows Platform3 validation patterns

#### MCP Coordination Issues
- Verify agent bridge configurations
- Check async/await patterns in MCP handlers
- Ensure proper error propagation between agents

### Testing Approach
- Run comprehensive validation before major changes
- Use existing test files as patterns for new tests
- Focus on integration testing for agent coordination
- Validate indicator calculations with known data sets

### Performance Considerations
- Platform3 processes real-time trading data
- Optimize for low latency in critical paths
- Use async/await for I/O operations
- Cache expensive calculations when appropriate

---

**Remember**: Platform3 is a production trading system. All changes should be thoroughly tested and validated before deployment. When in doubt, prioritize stability and data integrity over new features.
