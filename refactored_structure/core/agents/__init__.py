"""
Base Agent Interface for Platform3
Provides the foundational structure for all agent implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime


class BaseAgentInterface(ABC):
    """Base class for all Platform3 agents"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"agent.{agent_name}")
        self.indicators = {}
        self.status = "initialized"
        self.created_at = datetime.now()
        
    @abstractmethod
    def execute_analysis(self, market_data: Dict, agent_signals: Dict) -> Dict[str, Any]:
        """
        Execute the agent's analysis
        
        Args:
            market_data: Market data for analysis
            agent_signals: Signals from other agents
            
        Returns:
            Analysis results dictionary
        """
        return {
            "agent": self.agent_name,
            "status": "not_implemented",
            "analysis_timestamp": datetime.now().isoformat(),
            "error": "Analysis not implemented"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "agent_name": self.agent_name,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "indicators_loaded": len(self.indicators)
        }
    
    def _prepare_market_data(self, market_data: Any):
        """
        Prepare market data for analysis
        
        Args:
            market_data: Raw market data
            
        Returns:
            Processed DataFrame
        """
        import pandas as pd
        import numpy as np
        
        if isinstance(market_data, list) and len(market_data) > 0:
            df = pd.DataFrame(market_data)
        elif isinstance(market_data, dict):
            if 'ohlcv' in market_data:
                df = pd.DataFrame(market_data['ohlcv'])
            else:
                df = pd.DataFrame([market_data])
        else:
            # Generate sample data for testing
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            df = pd.DataFrame({
                'open': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 105,
                'low': np.random.randn(100).cumsum() + 95,
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
            
        return df
    
    def _load_assigned_indicators(self):
        """Load indicators assigned to this agent"""
        try:
            from ..indicators.registry import INDICATOR_REGISTRY
            if hasattr(self, 'assigned_indicators'):
                for category, indicator_names in self.assigned_indicators.items():
                    self.indicators[category] = {}
                    for indicator_name in indicator_names:
                        if indicator_name in INDICATOR_REGISTRY:
                            self.indicators[category][indicator_name] = INDICATOR_REGISTRY[indicator_name]
                            self.logger.info(f"Loaded {indicator_name} for {self.agent_name}")
        except Exception as e:
            self.logger.error(f"Failed to load indicators: {e}")