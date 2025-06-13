"""
Base Indicator Interface for Platform3
Provides the foundational structure for all indicator implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime


class BaseIndicator(ABC):
    """Base class for all Platform3 indicators"""
    
    def __init__(self, name: str, category: str, parameters: Optional[Dict] = None):
        self.name = name
        self.category = category
        self.parameters = parameters or {}
        self.logger = logging.getLogger(f"indicator.{name}")
        self.created_at = datetime.now()
        self.calculation_count = 0
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame, float]:
        """
        Calculate the indicator value(s)
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Calculated indicator values
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data
        
        Args:
            data: Market data DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            self.logger.error(f"Missing required columns: {required_columns}")
            return False
            
        if data.empty:
            self.logger.error("Empty dataset provided")
            return False
            
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get indicator information"""
        return {
            "name": self.name,
            "category": self.category,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat(),
            "calculation_count": self.calculation_count
        }
    
    def safe_calculate(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame, float]:
        """
        Safely calculate indicator with error handling
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Calculated values or default value on error
        """
        try:
            if not self.validate_data(data):
                return pd.Series([0.0] * len(data), index=data.index)
                
            result = self.calculate(data)
            self.calculation_count += 1
            return result
            
        except Exception as e:
            self.logger.error(f"Calculation failed for {self.name}: {e}")
            return pd.Series([0.0] * len(data), index=data.index)


class TechnicalIndicator(BaseIndicator):
    """Base class for technical indicators"""
    
    def __init__(self, name: str, parameters: Optional[Dict] = None):
        super().__init__(name, "technical", parameters)


class PhysicsIndicator(BaseIndicator):
    """Base class for physics-based indicators"""
    
    def __init__(self, name: str, parameters: Optional[Dict] = None):
        super().__init__(name, "physics", parameters)