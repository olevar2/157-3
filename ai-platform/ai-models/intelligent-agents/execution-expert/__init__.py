"""
Execution Expert - Optimal Trade Execution and Timing

Professional-grade trade execution optimization that ensures optimal
entry and exit timing, order management, and slippage minimization
for maximum profit generation to support humanitarian causes.

Ultra-Fast Implementation: Achieves <0.1ms execution analysis using
JIT-compiled algorithms for real-time trading optimization.
"""

# Import original model for compatibility
from .model import ExecutionExpert, ExecutionStrategy, ExecutionSignal, OrderType

# Import ultra-fast model as default for production
from .ultra_fast_model import (
    UltraFastExecutionExpert,
    ultra_fast_execution_expert,
    optimize_execution_ultra_fast,
    calculate_slippage_fast,
    get_optimal_order_type_fast,
    calculate_market_impact_fast,
    get_execution_timing_fast,
    optimize_execution_with_67_indicators  # Enhanced function
)

# Use ultra-fast model as default while maintaining backward compatibility
ExecutionExpertDefault = UltraFastExecutionExpert

__all__ = [
    'ExecutionExpert', 'ExecutionStrategy', 'ExecutionSignal', 'OrderType',
    'UltraFastExecutionExpert', 'ExecutionExpertDefault',
    'ultra_fast_execution_expert', 'optimize_execution_ultra_fast',
    'calculate_slippage_fast', 'get_optimal_order_type_fast',
    'calculate_market_impact_fast', 'get_execution_timing_fast',
    'optimize_execution_with_67_indicators'  # Enhanced function
]
