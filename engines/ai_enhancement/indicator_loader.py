"""
Platform3 Indicator Loader Utility
Provides functionality to load indicators for agents based on GENIUS_AGENT_INDICATOR_MAPPING.md
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add Platform3 to path
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

class IndicatorLoader:
    """
    Utility class for loading indicators for genius agents
    Uses mappings from GENIUS_AGENT_INDICATOR_MAPPING.md
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize the indicator loader for a specific agent
        
        Args:
            agent_name: The name of the agent (snake_case)
        """
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"IndicatorLoader:{agent_name}")
        
        # Load indicator mappings
        self._load_mappings()
    
    def _load_mappings(self):
        """Load indicator mappings from file or generate them"""
        try:
            # First attempt to import from pre-generated file
            from .indicator_mappings import AGENT_INDICATOR_MAPPINGS
            self.mappings = AGENT_INDICATOR_MAPPINGS
            self.logger.info(f"Loaded indicator mappings from pre-generated file")
        except ImportError:
            # If file doesn't exist, generate mappings from markdown
            self.logger.info(f"Pre-generated mappings not found, parsing markdown file...")
            
            try:
                # Import parser function
                from indicator_mapping_integrator import parse_indicator_mapping
                
                # Get path to mapping file
                mapping_file = os.path.join(current_dir, "GENIUS_AGENT_INDICATOR_MAPPING.md")
                
                # Parse mapping file
                self.mappings = parse_indicator_mapping(mapping_file)
                self.logger.info(f"Generated mappings from {mapping_file}")
            except Exception as e:
                self.logger.error(f"Failed to parse indicator mappings: {e}")
                self.mappings = {}
    
    def get_assigned_indicators(self) -> Dict[str, List[str]]:
        """
        Get the indicators assigned to this agent
        
        Returns:
            Dictionary of indicator categories and their indicators
        """
        if self.agent_name in self.mappings:
            return self.mappings[self.agent_name]
        else:
            self.logger.warning(f"No mappings found for agent {self.agent_name}")
            return {}
    
    def _find_case_insensitive_indicator(self, indicator_name: str, registry: Dict[str, Any]) -> Optional[Any]:
        """
        Find an indicator in the registry, ignoring case differences
        
        Args:
            indicator_name: The name of the indicator to find
            registry: The indicator registry
            
        Returns:
            The indicator instance if found, None otherwise
        """
        # Try exact match first
        if indicator_name in registry:
            return registry[indicator_name]
        
        # Try lowercase match
        lowercase_name = indicator_name.lower()
        if lowercase_name in registry:
            self.logger.info(f"Found indicator {indicator_name} as lowercase {lowercase_name}")
            return registry[lowercase_name]
        
        # Try removing 'Indicator' suffix for better matching
        if lowercase_name.endswith('indicator'):
            base_name = lowercase_name[:-9]  # Remove 'indicator' from end
            if base_name in registry:
                self.logger.info(f"Found indicator {indicator_name} as base form {base_name}")
                return registry[base_name]
        
        # Try all registry keys with case insensitive comparison
        for key in registry:
            if key.lower() == lowercase_name or key.lower().replace('indicator', '') == lowercase_name.replace('indicator', ''):
                self.logger.info(f"Found case-insensitive match for {indicator_name}: {key}")
                return registry[key]
        
        # Try with spaces and underscores replaced
        normalized_name = lowercase_name.replace(' ', '').replace('_', '')
        for key in registry:
            if key.lower().replace(' ', '').replace('_', '') == normalized_name:
                self.logger.info(f"Found normalized match for {indicator_name}: {key}")
                return registry[key]
        
        return None
    
    def load_indicators(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all indicators assigned to this agent from the registry
        
        Returns:
            Dictionary of loaded indicators by category
        """
        # Get assigned indicators
        assigned_indicators = self.get_assigned_indicators()
        
        if not assigned_indicators:
            return {}
        
        # Dictionary to store loaded indicators
        loaded_indicators = {}
        
        try:
            # Import indicator registry
            try:
                from registry import INDICATOR_REGISTRY
            except ImportError:
                try:
                    from ...registry import INDICATOR_REGISTRY
                except ImportError:
                    from engines.ai_enhancement.registry import INDICATOR_REGISTRY
            
            # Load indicators by category
            for category, indicator_names in assigned_indicators.items():
                loaded_indicators[category] = {}
                
                for indicator_name in indicator_names:
                    try:
                        # Try to find indicator with case-insensitive matching
                        indicator = self._find_case_insensitive_indicator(indicator_name, INDICATOR_REGISTRY)
                        
                        if indicator:
                            loaded_indicators[category][indicator_name] = indicator
                            self.logger.info(f"Loaded {indicator_name} for {self.agent_name}")
                        else:
                            self.logger.warning(f"Indicator {indicator_name} not found in registry (not even with case-insensitive search)")
                    except Exception as e:
                        self.logger.error(f"Error loading indicator {indicator_name}: {e}")
            
            self.logger.info(f"Loaded {sum(len(inds) for inds in loaded_indicators.values())} indicators for {self.agent_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load indicators: {e}")
        
        return loaded_indicators

# Utility function for easy access
def load_indicators_for_agent(agent_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Load indicators for a specific agent
    
    Args:
        agent_name: The name of the agent (snake_case)
        
    Returns:
        Dictionary of loaded indicators by category
    """
    loader = IndicatorLoader(agent_name)
    return loader.load_indicators()

def get_assigned_indicators_for_agent(agent_name: str) -> Dict[str, List[str]]:
    """
    Get assigned indicators for a specific agent
    
    Args:
        agent_name: The name of the agent (snake_case)
        
    Returns:
        Dictionary of indicator categories and their indicators
    """
    loader = IndicatorLoader(agent_name)
    return loader.get_assigned_indicators()
