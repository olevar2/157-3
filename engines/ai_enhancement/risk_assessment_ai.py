"""
Risk Assessment AI Module for Platform3
Provides comprehensive risk analysis capabilities for the adaptive layer
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RiskAssessmentAI:
    """
    Advanced AI-based risk assessment for trading decisions
    Integrates with the adaptive indicator system
    """
    
    def __init__(self):
        """Initialize the Risk Assessment AI"""
        self.initialized = True
        self.risk_models = {
            'volatility': self._volatility_risk_model,
            'correlation': self._correlation_risk_model,
            'liquidity': self._liquidity_risk_model,
            'market_impact': self._market_impact_model,
            'tail_risk': self._tail_risk_model
        }
        logger.info("RiskAssessmentAI initialized successfully")
    
    def assess_position_risk(self, market_data: Dict[str, Any], 
                           position_size: float = 1.0,
                           timeframe: str = '1H') -> Dict[str, Any]:
        """
        Assess risk for a potential position
        
        Args:
            market_data: Current market data including OHLCV
            position_size: Proposed position size (1.0 = standard lot)
            timeframe: Trading timeframe
            
        Returns:
            Dictionary containing comprehensive risk assessment
        """
        try:
            risk_assessment = {
                'overall_risk_score': 0.0,
                'risk_level': 'LOW',
                'risk_components': {},
                'recommendations': [],
                'max_position_size': 0.0,
                'stop_loss_suggestion': 0.0,
                'take_profit_suggestion': 0.0,
                'confidence': 0.0
            }
            
            # Calculate individual risk components
            total_risk = 0.0
            component_count = 0
            
            for risk_type, risk_model in self.risk_models.items():
                component_risk = risk_model(market_data, position_size)
                risk_assessment['risk_components'][risk_type] = component_risk
                total_risk += component_risk['score']
                component_count += 1
            
            # Calculate overall risk score (0-100, lower is better)
            risk_assessment['overall_risk_score'] = total_risk / component_count if component_count > 0 else 50.0
            
            # Determine risk level
            if risk_assessment['overall_risk_score'] < 30:
                risk_assessment['risk_level'] = 'LOW'
                risk_assessment['max_position_size'] = position_size * 1.5
            elif risk_assessment['overall_risk_score'] < 70:
                risk_assessment['risk_level'] = 'MEDIUM'
                risk_assessment['max_position_size'] = position_size
            else:
                risk_assessment['risk_level'] = 'HIGH'
                risk_assessment['max_position_size'] = position_size * 0.5
            
            # Generate recommendations
            risk_assessment['recommendations'] = self._generate_risk_recommendations(risk_assessment)
            
            # Set confidence based on data quality
            risk_assessment['confidence'] = self._calculate_confidence(market_data)
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error in position risk assessment: {e}")
            return self._default_risk_assessment()
    
    def assess_portfolio_risk(self, positions: List[Dict[str, Any]], 
                            market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess overall portfolio risk across multiple positions
        
        Args:
            positions: List of current positions with their details
            market_data: Current market data
            
        Returns:
            Portfolio risk assessment
        """
        try:
            portfolio_risk = {
                'total_exposure': 0.0,
                'correlation_risk': 0.0,
                'concentration_risk': 0.0,
                'diversification_score': 0.0,
                'var_estimate': 0.0,  # Value at Risk
                'expected_shortfall': 0.0,
                'recommendations': []
            }
            
            if not positions:
                return portfolio_risk
            
            # Calculate total exposure
            portfolio_risk['total_exposure'] = sum(pos.get('size', 0) * pos.get('price', 0) for pos in positions)
            
            # Assess concentration risk
            portfolio_risk['concentration_risk'] = self._calculate_concentration_risk(positions)
            
            # Assess correlation risk
            portfolio_risk['correlation_risk'] = self._calculate_correlation_risk(positions, market_data)
            
            # Calculate diversification score
            portfolio_risk['diversification_score'] = self._calculate_diversification_score(positions)
            
            # Estimate VaR and Expected Shortfall
            portfolio_risk['var_estimate'] = self._estimate_var(positions, market_data)
            portfolio_risk['expected_shortfall'] = portfolio_risk['var_estimate'] * 1.3  # Conservative estimate
            
            # Generate portfolio recommendations
            portfolio_risk['recommendations'] = self._generate_portfolio_recommendations(portfolio_risk)
            
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"Error in portfolio risk assessment: {e}")
            return {'error': str(e)}
    
    def _volatility_risk_model(self, market_data: Dict[str, Any], position_size: float) -> Dict[str, Any]:
        """Model volatility-based risk"""
        try:
            # Use price series if available, otherwise estimate from OHLC
            if 'price_series' in market_data and len(market_data['price_series']) > 1:
                prices = market_data['price_series']
                returns = np.diff(np.log(prices))
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
            else:
                # Estimate from OHLC
                high = market_data.get('high', market_data.get('close', 1.0))
                low = market_data.get('low', market_data.get('close', 1.0))
                close = market_data.get('close', 1.0)
                volatility = (high - low) / close if close > 0 else 0.1
            
            # Scale risk score based on volatility (0-100)
            risk_score = min(volatility * 1000, 100)  # Normalize for typical FX volatility
            
            return {
                'score': risk_score,
                'volatility': volatility,
                'description': f"Volatility-based risk: {volatility:.4f}"
            }
        except:
            return {'score': 50.0, 'volatility': 0.01, 'description': "Default volatility risk"}
    
    def _correlation_risk_model(self, market_data: Dict[str, Any], position_size: float) -> Dict[str, Any]:
        """Model correlation-based risk"""
        # Simplified correlation risk model
        symbol = market_data.get('symbol', 'UNKNOWN')
        
        # Major pairs typically have lower correlation risk
        major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        
        if symbol in major_pairs:
            risk_score = 25.0  # Lower correlation risk for majors
        else:
            risk_score = 45.0  # Higher correlation risk for minors/exotics
        
        return {
            'score': risk_score,
            'correlation_estimate': risk_score / 100,
            'description': f"Correlation risk for {symbol}"
        }
    
    def _liquidity_risk_model(self, market_data: Dict[str, Any], position_size: float) -> Dict[str, Any]:
        """Model liquidity-based risk"""
        volume = market_data.get('volume', 1000)
        
        # Risk increases with position size relative to volume
        liquidity_ratio = position_size / max(volume, 100)
        risk_score = min(liquidity_ratio * 5000, 100)  # Scale appropriately
        
        return {
            'score': risk_score,
            'liquidity_ratio': liquidity_ratio,
            'description': f"Liquidity risk based on volume {volume}"
        }
    
    def _market_impact_model(self, market_data: Dict[str, Any], position_size: float) -> Dict[str, Any]:
        """Model market impact risk"""
        # Simplified market impact model
        base_impact = position_size * 0.1  # Basic linear impact model
        risk_score = min(base_impact * 100, 100)
        
        return {
            'score': risk_score,
            'estimated_impact': base_impact,
            'description': f"Market impact risk for position size {position_size}"
        }
    
    def _tail_risk_model(self, market_data: Dict[str, Any], position_size: float) -> Dict[str, Any]:
        """Model tail risk (extreme events)"""
        # Simple tail risk estimate based on recent volatility
        high = market_data.get('high', market_data.get('close', 1.0))
        low = market_data.get('low', market_data.get('close', 1.0))
        close = market_data.get('close', 1.0)
        
        if close > 0:
            daily_range = (high - low) / close
            tail_risk_score = min(daily_range * 2000, 100)  # Scale for typical FX ranges
        else:
            tail_risk_score = 30.0
        
        return {
            'score': tail_risk_score,
            'daily_range': daily_range if close > 0 else 0.01,
            'description': f"Tail risk based on daily range"
        }
    
    def _generate_risk_recommendations(self, risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate actionable risk recommendations"""
        recommendations = []
        
        risk_score = risk_assessment['overall_risk_score']
        risk_level = risk_assessment['risk_level']
        
        if risk_level == 'HIGH':
            recommendations.append("🚨 High risk detected - consider reducing position size")
            recommendations.append("[DOWN] Implement tight stop losses")
            recommendations.append("⏰ Monitor position closely")
        elif risk_level == 'MEDIUM':
            recommendations.append("[WARN] Moderate risk - standard risk management applies")
            recommendations.append("[CHART] Monitor key risk metrics")
        else:
            recommendations.append("[OK] Low risk environment - normal position sizing acceptable")
            recommendations.append("[UP] Consider scaling up if opportunity warrants")
        
        # Component-specific recommendations
        components = risk_assessment.get('risk_components', {})
        
        if components.get('volatility', {}).get('score', 0) > 60:
            recommendations.append("🌪️ High volatility - widen stops and reduce leverage")
        
        if components.get('liquidity', {}).get('score', 0) > 70:
            recommendations.append("💧 Low liquidity - avoid large positions")
        
        return recommendations
    
    def _calculate_confidence(self, market_data: Dict[str, Any]) -> float:
        """Calculate confidence in the risk assessment"""
        confidence_factors = []
        
        # Data completeness
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        available_fields = sum(1 for field in required_fields if field in market_data)
        data_completeness = available_fields / len(required_fields)
        confidence_factors.append(data_completeness)
        
        # Price series availability
        if 'price_series' in market_data and len(market_data['price_series']) > 10:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Volume validity
        volume = market_data.get('volume', 0)
        if volume > 100:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _default_risk_assessment(self) -> Dict[str, Any]:
        """Return default risk assessment when calculation fails"""
        return {
            'overall_risk_score': 50.0,
            'risk_level': 'MEDIUM',
            'risk_components': {},
            'recommendations': ['[WARN] Default risk assessment - limited data available'],
            'max_position_size': 1.0,
            'stop_loss_suggestion': 0.02,
            'take_profit_suggestion': 0.03,
            'confidence': 0.3
        }
    
    def _calculate_concentration_risk(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate position concentration risk"""
        if not positions:
            return 0.0
        
        total_value = sum(pos.get('value', 0) for pos in positions)
        if total_value == 0:
            return 0.0
        
        # Calculate Herfindahl index for concentration
        weights = [pos.get('value', 0) / total_value for pos in positions]
        herfindahl = sum(w**2 for w in weights)
        
        # Convert to risk score (0-100)
        return min(herfindahl * 100, 100)
    
    def _calculate_correlation_risk(self, positions: List[Dict[str, Any]], market_data: Dict[str, Any]) -> float:
        """Calculate correlation risk across positions"""
        # Simplified correlation risk - would need actual correlation matrix in production
        if len(positions) <= 1:
            return 0.0
        
        # Estimate based on currency exposure
        currencies = set()
        for pos in positions:
            symbol = pos.get('symbol', '')
            if len(symbol) >= 6:
                currencies.add(symbol[:3])  # Base currency
                currencies.add(symbol[3:6])  # Quote currency
        
        # More currency diversity = lower correlation risk
        diversity_score = len(currencies) / (len(positions) * 2)
        correlation_risk = max(0, (1 - diversity_score) * 100)
        
        return min(correlation_risk, 100)
    
    def _calculate_diversification_score(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate portfolio diversification score"""
        if not positions:
            return 0.0
        
        # Simple diversification based on number of positions and value distribution
        num_positions = len(positions)
        
        if num_positions == 1:
            return 20.0
        elif num_positions <= 3:
            return 40.0
        elif num_positions <= 5:
            return 70.0
        else:
            return 90.0
    
    def _estimate_var(self, positions: List[Dict[str, Any]], market_data: Dict[str, Any]) -> float:
        """Estimate Value at Risk for the portfolio"""
        if not positions:
            return 0.0
        
        # Simplified VaR calculation
        total_value = sum(pos.get('value', 0) for pos in positions)
        
        # Assume 2% daily VaR for typical FX portfolio
        estimated_var = total_value * 0.02
        
        return estimated_var
    
    def _generate_portfolio_recommendations(self, portfolio_risk: Dict[str, Any]) -> List[str]:
        """Generate portfolio-level recommendations"""
        recommendations = []
        
        concentration_risk = portfolio_risk.get('concentration_risk', 0)
        correlation_risk = portfolio_risk.get('correlation_risk', 0)
        diversification_score = portfolio_risk.get('diversification_score', 0)
        
        if concentration_risk > 70:
            recommendations.append("[TARGET] High concentration risk - diversify positions")
        
        if correlation_risk > 60:
            recommendations.append("🔗 High correlation risk - reduce correlated positions")
        
        if diversification_score < 40:
            recommendations.append("[CHART] Low diversification - add uncorrelated positions")
        
        if not recommendations:
            recommendations.append("[OK] Portfolio risk levels appear balanced")
        
        return recommendations

# Global instance for easy access
_risk_assessment_ai = None

def get_risk_assessment_ai() -> RiskAssessmentAI:
    """Get the global RiskAssessmentAI instance"""
    global _risk_assessment_ai
    if _risk_assessment_ai is None:
        _risk_assessment_ai = RiskAssessmentAI()
    return _risk_assessment_ai
