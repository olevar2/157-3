"""
Platform3 Base Agent Interface
Provides a base interface for all genius agents with indicator loading capabilities
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

class BaseAgentInterface:
    """
    Base interface for all Platform3 genius agents
    Handles common functionality like indicator loading
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize the base agent interface
        
        Args:
            agent_name: The name of the agent (snake_case)
        """
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"Agent:{agent_name}")
        
        # Initialize indicator dictionaries
        self.assigned_indicators = {}
        self.indicators = {}
        
        # Load indicator assignments
        self._load_indicator_assignments()
    
    def _load_indicator_assignments(self):
        """Load indicator assignments from mapping file"""
        try:
            # Import loader
            from .indicator_loader import get_assigned_indicators_for_agent, load_indicators_for_agent
            
            # Get assigned indicators
            self.assigned_indicators = get_assigned_indicators_for_agent(self.agent_name)
            self.logger.info(f"Loaded indicator assignments: {sum(len(inds) for inds in self.assigned_indicators.values())} indicators")
            
            # Load indicators from registry
            self.indicators = load_indicators_for_agent(self.agent_name)
            self.logger.info(f"Loaded indicators from registry: {sum(len(inds) for inds in self.indicators.values())} indicators")
            
        except Exception as e:
            self.logger.error(f"Failed to load indicator assignments: {e}")
    
    def execute_analysis(self, market_data: Dict, agent_signals: Dict) -> Dict[str, Any]:
        """
        Execute analysis using assigned indicators
        
        Args:
            market_data: Dictionary or list containing market data
            agent_signals: Dictionary containing signals from other agents
            
        Returns:
            Dictionary containing analysis results
        """
        # Default implementation returns basic info
        return {
            "agent": self.agent_name,
            "status": "active",
            "analysis_timestamp": datetime.now().isoformat(),
            "message": "Base analysis - override in subclass"
        }
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """
        Get information about assigned indicators
        
        Returns:
            Dictionary containing indicator information
        """
        total_assigned = sum(len(inds) for inds in self.assigned_indicators.values())
        total_loaded = sum(len(inds) for inds in self.indicators.values())
        
        return {
            "agent": self.agent_name,
            "total_assigned_indicators": total_assigned,
            "total_loaded_indicators": total_loaded,
            "categories": list(self.assigned_indicators.keys()),
            "loading_status": "complete" if total_loaded == total_assigned else "partial"
        }
