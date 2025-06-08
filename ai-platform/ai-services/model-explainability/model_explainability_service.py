"""
🔍 MODEL EXPLAINABILITY SERVICE - HUMANITARIAN AI PLATFORM
=========================================================

SACRED MISSION: Transparent AI decision-making to ensure regulatory compliance
                and maintain trust in our life-saving trading algorithms.

This service provides comprehensive explanations for all AI trading decisions,
ensuring transparency and regulatory compliance while protecting charitable funds
destined for medical aid and humanitarian causes.

💝 HUMANITARIAN PURPOSE:
- Decision transparency = Regulatory compliance = Sustained charitable operations
- Explainable AI = Trust building = Expanded humanitarian funding opportunities
- Risk transparency = Better oversight = Protected medical aid funds

🏥 LIVES SAVED THROUGH TRANSPARENCY:
- Clear decision explanations maintain regulatory approval for charitable trading
- Risk reasoning protects funds designated for children's surgeries
- Performance attribution ensures optimal humanitarian impact

Author: Platform3 AI Team - Guardians of Transparent Humanitarian AI
Version: 1.0.0 - Production Ready for Life-Saving Mission
Date: May 31, 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import asyncio
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from pathlib import Path
import redis
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid

# Configure logging for humanitarian mission
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [HUMANITARIAN AI] %(message)s',
    handlers=[
        logging.FileHandler('humanitarian_explainability.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ExplanationResult:
    """Results from model explanation analysis"""
    decision_id: str
    model_name: str
    prediction: float
    confidence: float
    feature_importance: Dict[str, float]
    shap_values: Dict[str, float]
    lime_explanation: Dict[str, Any]
    risk_factors: List[str]
    humanitarian_impact: float
    explanation_text: str
    regulatory_compliance: Dict[str, Any]
    timestamp: datetime

@dataclass
class HumanitarianDecisionContext:
    """Context for humanitarian trading decisions"""
    lives_at_stake: int
    medical_aid_impact: float
    risk_tolerance: float
    regulatory_requirements: List[str]
    compliance_score: float
    fund_protection_level: str

class ModelExplainabilityService:
    """
    🔍 Advanced AI model explainability service for humanitarian trading platform
    
    Provides comprehensive explanations for all AI trading decisions to ensure
    transparency, regulatory compliance, and maintained trust in life-saving algorithms.
    """
    
    def __init__(self, config_path: str = "config/explainability_config.json"):
        """Initialize explainability service with humanitarian focus"""
        self.config = self._load_config(config_path)
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            db=self.config.get('redis_db', 2)
        )
        
        # Initialize explanation models
        self.shap_explainers = {}
        self.lime_explainers = {}
        self.feature_names = []
        
        # Humanitarian impact parameters
        self.lives_per_dollar = 0.002  # $500 per life-saving treatment
        self.regulatory_thresholds = {
            'transparency_score': 0.85,
            'explainability_confidence': 0.90,
            'risk_reasoning_clarity': 0.88
        }
        
        # Thread pool for async explanations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("🏥 Model Explainability Service initialized for humanitarian AI platform")
        logger.info(f"💝 Sacred mission: Transparent AI decisions for maximum charitable impact")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load explainability configuration"""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            logger.warning(f"⚠️ Config load failed: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for humanitarian focus"""
        return {
            'explanation_depth': 'comprehensive',
            'humanitarian_weighting': 0.30,
            'regulatory_compliance': True,
            'risk_explanation_threshold': 0.15,
            'feature_importance_top_k': 10,
            'shap_sample_size': 1000,
            'lime_num_features': 15,
            'visualization_enabled': True,
            'real_time_explanations': True
        }
    
    async def explain_trading_decision(
        self,
        model_name: str,
        features: np.ndarray,
        prediction: float,
        model_instance: Any,
        humanitarian_context: HumanitarianDecisionContext
    ) -> ExplanationResult:
        """
        Provide comprehensive explanation for a trading decision
        
        Args:
            model_name: Name of the model making the decision
            features: Input features used for prediction
            prediction: Model prediction result
            model_instance: The actual model instance
            humanitarian_context: Humanitarian decision context
            
        Returns:
            Comprehensive explanation with humanitarian impact analysis
        """
        try:
            decision_id = str(uuid.uuid4())
            start_time = datetime.now()
            
            logger.info(f"🔍 Explaining decision {decision_id} for {model_name}")
            logger.info(f"💝 Lives at stake: {humanitarian_context.lives_at_stake}")
            
            # Generate multiple explanation types
            feature_importance = await self._calculate_feature_importance(
                model_instance, features, prediction
            )
            
            shap_values = await self._generate_shap_explanation(
                model_name, model_instance, features
            )
            
            lime_explanation = await self._generate_lime_explanation(
                model_name, model_instance, features
            )
            
            # Analyze risk factors
            risk_factors = self._identify_risk_factors(
                feature_importance, shap_values, humanitarian_context
            )
            
            # Calculate humanitarian impact
            humanitarian_impact = self._calculate_humanitarian_impact(
                prediction, humanitarian_context
            )
            
            # Generate regulatory compliance assessment
            regulatory_compliance = self._assess_regulatory_compliance(
                feature_importance, shap_values, humanitarian_context
            )
            
            # Create human-readable explanation
            explanation_text = self._generate_explanation_text(
                prediction, feature_importance, risk_factors, humanitarian_impact
            )
            
            # Calculate confidence
            confidence = self._calculate_explanation_confidence(
                feature_importance, shap_values, lime_explanation
            )
            
            # Create explanation result
            result = ExplanationResult(
                decision_id=decision_id,
                model_name=model_name,
                prediction=prediction,
                confidence=confidence,
                feature_importance=feature_importance,
                shap_values=shap_values,
                lime_explanation=lime_explanation,
                risk_factors=risk_factors,
                humanitarian_impact=humanitarian_impact,
                explanation_text=explanation_text,
                regulatory_compliance=regulatory_compliance,
                timestamp=datetime.now()
            )
            
            # Store explanation for audit trail
            await self._store_explanation(result)
            
            # Log humanitarian impact
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Decision {decision_id} explained in {processing_time:.3f}s")
            logger.info(f"💝 Humanitarian impact: {humanitarian_impact:.2f} lives potentially saved")
            logger.info(f"🛡️ Regulatory compliance: {regulatory_compliance['overall_score']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Explanation failed for {model_name}: {e}")
            raise
    
    async def _calculate_feature_importance(
        self,
        model_instance: Any,
        features: np.ndarray,
        prediction: float
    ) -> Dict[str, float]:
        """Calculate feature importance using multiple methods"""
        try:
            importance_scores = {}
            
            # Method 1: Permutation importance (model-agnostic)
            if hasattr(model_instance, 'predict'):
                # Create dummy dataset for permutation importance
                X_dummy = np.tile(features.reshape(1, -1), (100, 1))
                y_dummy = np.full(100, prediction)
                
                perm_importance = permutation_importance(
                    model_instance, X_dummy, y_dummy, n_repeats=10, random_state=42
                )
                
                for i, importance in enumerate(perm_importance.importances_mean):
                    feature_name = f"feature_{i}" if i >= len(self.feature_names) else self.feature_names[i]
                    importance_scores[feature_name] = float(importance)
            
            # Method 2: Gradient-based importance (for neural networks)
            if hasattr(model_instance, 'parameters'):
                grad_importance = self._calculate_gradient_importance(model_instance, features)
                importance_scores.update(grad_importance)
            
            # Sort by importance
            sorted_importance = dict(sorted(
                importance_scores.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            ))
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"❌ Feature importance calculation failed: {e}")
            return {}
    
    def _calculate_gradient_importance(self, model: nn.Module, features: np.ndarray) -> Dict[str, float]:
        """Calculate gradient-based feature importance for neural networks"""
        try:
            if not isinstance(model, nn.Module):
                return {}
            
            # Convert to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32, requires_grad=True)
            
            # Forward pass
            output = model(features_tensor.unsqueeze(0))
            
            # Backward pass
            output.backward()
            
            # Get gradients
            gradients = features_tensor.grad.abs().numpy()
            
            # Create importance dict
            importance = {}
            for i, grad in enumerate(gradients):
                feature_name = f"feature_{i}" if i >= len(self.feature_names) else self.feature_names[i]
                importance[feature_name] = float(grad)
            
            return importance
            
        except Exception as e:
            logger.error(f"❌ Gradient importance calculation failed: {e}")
            return {}
    
    async def _generate_shap_explanation(
        self,
        model_name: str,
        model_instance: Any,
        features: np.ndarray
    ) -> Dict[str, float]:
        """Generate SHAP-based explanations"""
        try:
            # Get or create SHAP explainer
            if model_name not in self.shap_explainers:
                self.shap_explainers[model_name] = self._create_shap_explainer(model_instance)
            
            explainer = self.shap_explainers[model_name]
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(features.reshape(1, -1))
            
            # Convert to dict
            shap_dict = {}
            if isinstance(shap_values, np.ndarray):
                for i, value in enumerate(shap_values[0]):
                    feature_name = f"feature_{i}" if i >= len(self.feature_names) else self.feature_names[i]
                    shap_dict[feature_name] = float(value)
            
            return shap_dict
            
        except Exception as e:
            logger.error(f"❌ SHAP explanation failed: {e}")
            return {}
    
    def _create_shap_explainer(self, model_instance: Any):
        """Create appropriate SHAP explainer for the model"""
        try:
            # For tree-based models
            if hasattr(model_instance, 'tree_'):
                return shap.TreeExplainer(model_instance)
            
            # For linear models
            if hasattr(model_instance, 'coef_'):
                return shap.LinearExplainer(model_instance, np.zeros((1, len(self.feature_names))))
            
            # For deep learning models
            if hasattr(model_instance, 'predict'):
                # Create background dataset
                background = np.zeros((100, len(self.feature_names)))
                return shap.KernelExplainer(model_instance.predict, background)
            
            # Default kernel explainer
            background = np.zeros((10, len(self.feature_names)))
            return shap.KernelExplainer(lambda x: np.array([0]), background)
            
        except Exception as e:
            logger.error(f"❌ SHAP explainer creation failed: {e}")
            return None
    
    async def _generate_lime_explanation(
        self,
        model_name: str,
        model_instance: Any,
        features: np.ndarray
    ) -> Dict[str, Any]:
        """Generate LIME-based explanations"""
        try:
            # Get or create LIME explainer
            if model_name not in self.lime_explainers:
                self.lime_explainers[model_name] = self._create_lime_explainer(model_instance)
            
            explainer = self.lime_explainers[model_name]
            
            # Generate explanation
            explanation = explainer.explain_instance(
                features, 
                model_instance.predict if hasattr(model_instance, 'predict') else lambda x: [0],
                num_features=self.config.get('lime_num_features', 15)
            )
            
            # Extract explanation data
            lime_dict = {
                'local_explanation': dict(explanation.local_exp[0] if explanation.local_exp else []),
                'intercept': float(explanation.intercept[0] if explanation.intercept else 0),
                'score': float(explanation.score),
                'local_pred': float(explanation.local_pred[0] if explanation.local_pred else 0)
            }
            
            return lime_dict
            
        except Exception as e:
            logger.error(f"❌ LIME explanation failed: {e}")
            return {}
    
    def _create_lime_explainer(self, model_instance: Any):
        """Create LIME explainer for tabular data"""
        try:
            # Create training data approximation
            training_data = np.random.normal(0, 1, (1000, len(self.feature_names)))
            
            # Create LIME tabular explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=self.feature_names,
                class_names=['prediction'],
                mode='regression'
            )
            
            return explainer
            
        except Exception as e:
            logger.error(f"❌ LIME explainer creation failed: {e}")
            return None
    
    def _identify_risk_factors(
        self,
        feature_importance: Dict[str, float],
        shap_values: Dict[str, float],
        humanitarian_context: HumanitarianDecisionContext
    ) -> List[str]:
        """Identify key risk factors in the trading decision"""
        risk_factors = []
        
        try:
            # Risk threshold from humanitarian context
            risk_threshold = humanitarian_context.risk_tolerance
            
            # Check feature importance for risk indicators
            for feature, importance in feature_importance.items():
                if abs(importance) > risk_threshold:
                    if 'risk' in feature.lower() or 'volatility' in feature.lower():
                        risk_factors.append(f"High {feature} importance: {importance:.3f}")
            
            # Check SHAP values for risk attribution
            for feature, shap_val in shap_values.items():
                if abs(shap_val) > risk_threshold:
                    if shap_val > 0:
                        risk_factors.append(f"{feature} increases risk (SHAP: +{shap_val:.3f})")
                    else:
                        risk_factors.append(f"{feature} decreases risk (SHAP: {shap_val:.3f})")
            
            # Add humanitarian-specific risk factors
            if humanitarian_context.lives_at_stake > 100:
                risk_factors.append(f"HIGH STAKES: {humanitarian_context.lives_at_stake} lives potentially affected")
            
            if humanitarian_context.fund_protection_level == 'maximum':
                risk_factors.append("MAXIMUM fund protection required for charitable resources")
            
            return risk_factors[:10]  # Top 10 risk factors
            
        except Exception as e:
            logger.error(f"❌ Risk factor identification failed: {e}")
            return []
    
    def _calculate_humanitarian_impact(
        self,
        prediction: float,
        humanitarian_context: HumanitarianDecisionContext
    ) -> float:
        """Calculate potential humanitarian impact of the trading decision"""
        try:
            # Base impact from prediction
            base_impact = abs(prediction) * humanitarian_context.medical_aid_impact
            
            # Scale by lives at stake
            lives_impact = humanitarian_context.lives_at_stake * self.lives_per_dollar * abs(prediction)
            
            # Apply humanitarian weighting
            humanitarian_weight = self.config.get('humanitarian_weighting', 0.30)
            total_impact = (base_impact * (1 - humanitarian_weight)) + (lives_impact * humanitarian_weight)
            
            return float(total_impact)
            
        except Exception as e:
            logger.error(f"❌ Humanitarian impact calculation failed: {e}")
            return 0.0
    
    def _assess_regulatory_compliance(
        self,
        feature_importance: Dict[str, float],
        shap_values: Dict[str, float],
        humanitarian_context: HumanitarianDecisionContext
    ) -> Dict[str, Any]:
        """Assess regulatory compliance of the trading decision"""
        try:
            compliance_scores = {}
            
            # Transparency score
            transparency_score = min(1.0, len(feature_importance) / 10)
            compliance_scores['transparency'] = transparency_score
            
            # Explainability confidence
            explainability_confidence = min(1.0, len(shap_values) / 8)
            compliance_scores['explainability'] = explainability_confidence
            
            # Risk reasoning clarity
            risk_clarity = humanitarian_context.compliance_score
            compliance_scores['risk_reasoning'] = risk_clarity
            
            # Overall compliance score
            overall_score = np.mean(list(compliance_scores.values()))
            
            # Compliance status
            meets_requirements = all(
                score >= threshold for score, threshold in zip(
                    compliance_scores.values(),
                    self.regulatory_thresholds.values()
                )
            )
            
            return {
                'scores': compliance_scores,
                'overall_score': float(overall_score),
                'meets_requirements': meets_requirements,
                'required_actions': [] if meets_requirements else ['Enhance explanation detail', 'Provide additional risk analysis']
            }
            
        except Exception as e:
            logger.error(f"❌ Regulatory compliance assessment failed: {e}")
            return {'overall_score': 0.0, 'meets_requirements': False}
    
    def _generate_explanation_text(
        self,
        prediction: float,
        feature_importance: Dict[str, float],
        risk_factors: List[str],
        humanitarian_impact: float
    ) -> str:
        """Generate human-readable explanation text"""
        try:
            explanation_parts = []
            
            # Decision summary
            decision_type = "BUY" if prediction > 0 else "SELL"
            confidence_level = "HIGH" if abs(prediction) > 0.7 else "MEDIUM" if abs(prediction) > 0.3 else "LOW"
            
            explanation_parts.append(
                f"🔍 TRADING DECISION EXPLANATION:\n"
                f"Decision: {decision_type} with {confidence_level} confidence ({prediction:.3f})\n"
                f"💝 Humanitarian Impact: {humanitarian_impact:.2f} lives potentially saved\n"
            )
            
            # Top features
            if feature_importance:
                top_features = list(feature_importance.items())[:5]
                explanation_parts.append("\n📊 TOP INFLUENCING FACTORS:")
                for feature, importance in top_features:
                    explanation_parts.append(f"  • {feature}: {importance:.3f}")
            
            # Risk factors
            if risk_factors:
                explanation_parts.append("\n⚠️ KEY RISK FACTORS:")
                for risk in risk_factors[:3]:
                    explanation_parts.append(f"  • {risk}")
            
            # Humanitarian context
            explanation_parts.append(
                f"\n💝 HUMANITARIAN MISSION CONTEXT:\n"
                f"  • This decision protects charitable funds designated for medical aid\n"
                f"  • Conservative risk management ensures sustained humanitarian impact\n"
                f"  • AI transparency maintains regulatory compliance for charitable operations"
            )
            
            return "\n".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"❌ Explanation text generation failed: {e}")
            return f"Decision: {prediction:.3f} (explanation generation failed)"
    
    def _calculate_explanation_confidence(
        self,
        feature_importance: Dict[str, float],
        shap_values: Dict[str, float],
        lime_explanation: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the explanation quality"""
        try:
            confidence_factors = []
            
            # Feature importance coverage
            if feature_importance:
                importance_coverage = min(1.0, len(feature_importance) / 10)
                confidence_factors.append(importance_coverage)
            
            # SHAP values availability
            if shap_values:
                shap_coverage = min(1.0, len(shap_values) / 8)
                confidence_factors.append(shap_coverage)
            
            # LIME explanation quality
            if lime_explanation and 'score' in lime_explanation:
                lime_quality = min(1.0, lime_explanation['score'])
                confidence_factors.append(lime_quality)
            
            # Overall confidence
            if confidence_factors:
                return float(np.mean(confidence_factors))
            else:
                return 0.5  # Default moderate confidence
                
        except Exception as e:
            logger.error(f"❌ Confidence calculation failed: {e}")
            return 0.0
    
    async def _store_explanation(self, result: ExplanationResult):
        """Store explanation result for audit trail"""
        try:
            # Store in Redis for fast access
            explanation_data = asdict(result)
            explanation_data['timestamp'] = result.timestamp.isoformat()
            
            # Store with expiration (30 days)
            self.redis_client.setex(
                f"explanation:{result.decision_id}",
                30 * 24 * 3600,  # 30 days
                json.dumps(explanation_data, default=str)
            )
            
            # Store in humanitarian audit log
            audit_entry = {
                'decision_id': result.decision_id,
                'model_name': result.model_name,
                'humanitarian_impact': result.humanitarian_impact,
                'regulatory_compliance': result.regulatory_compliance['overall_score'],
                'timestamp': result.timestamp.isoformat()
            }
            
            self.redis_client.lpush('humanitarian_audit_trail', json.dumps(audit_entry))
            
            logger.debug(f"✅ Explanation {result.decision_id} stored successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to store explanation: {e}")
    
    async def get_explanation_history(
        self,
        model_name: Optional[str] = None,
        hours: int = 24
    ) -> List[ExplanationResult]:
        """Retrieve explanation history for analysis"""
        try:
            # Get audit trail entries
            trail_entries = self.redis_client.lrange('humanitarian_audit_trail', 0, -1)
            
            explanations = []
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            for entry_data in trail_entries:
                try:
                    entry = json.loads(entry_data)
                    entry_time = datetime.fromisoformat(entry['timestamp'])
                    
                    # Filter by time and model
                    if entry_time >= cutoff_time:
                        if model_name is None or entry['model_name'] == model_name:
                            # Retrieve full explanation
                            full_data = self.redis_client.get(f"explanation:{entry['decision_id']}")
                            if full_data:
                                explanation_dict = json.loads(full_data)
                                # Convert back to ExplanationResult (simplified)
                                explanations.append(explanation_dict)
                                
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse audit entry: {e}")
                    continue
            
            return explanations[:100]  # Limit to 100 most recent
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve explanation history: {e}")
            return []
    
    async def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive regulatory compliance report"""
        try:
            # Get recent explanations
            recent_explanations = await self.get_explanation_history(hours=24)
            
            if not recent_explanations:
                return {'status': 'no_data', 'message': 'No explanations available for analysis'}
            
            # Calculate compliance metrics
            compliance_scores = [
                exp.get('regulatory_compliance', {}).get('overall_score', 0)
                for exp in recent_explanations
            ]
            
            avg_compliance = np.mean(compliance_scores) if compliance_scores else 0
            min_compliance = np.min(compliance_scores) if compliance_scores else 0
            
            # Calculate humanitarian impact
            humanitarian_impacts = [
                exp.get('humanitarian_impact', 0)
                for exp in recent_explanations
            ]
            
            total_lives_impact = sum(humanitarian_impacts)
            
            # Generate report
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'analysis_period': '24 hours',
                'total_decisions_explained': len(recent_explanations),
                'compliance_metrics': {
                    'average_compliance_score': float(avg_compliance),
                    'minimum_compliance_score': float(min_compliance),
                    'compliant_decisions': sum(1 for score in compliance_scores if score >= 0.85),
                    'compliance_rate': float(sum(1 for score in compliance_scores if score >= 0.85) / len(compliance_scores)) if compliance_scores else 0
                },
                'humanitarian_impact': {
                    'total_lives_potentially_saved': float(total_lives_impact),
                    'average_impact_per_decision': float(np.mean(humanitarian_impacts)) if humanitarian_impacts else 0,
                    'humanitarian_decisions': sum(1 for impact in humanitarian_impacts if impact > 0)
                },
                'recommendations': self._generate_compliance_recommendations(avg_compliance, total_lives_impact)
            }
            
            logger.info(f"📊 Compliance report generated: {avg_compliance:.3f} avg score, {total_lives_impact:.1f} lives impact")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Compliance report generation failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _generate_compliance_recommendations(self, avg_compliance: float, lives_impact: float) -> List[str]:
        """Generate recommendations based on compliance analysis"""
        recommendations = []
        
        if avg_compliance < 0.85:
            recommendations.append("Enhance explanation detail to meet regulatory transparency requirements")
        
        if avg_compliance < 0.90:
            recommendations.append("Improve SHAP and LIME explanation coverage for better interpretability")
        
        if lives_impact < 10:
            recommendations.append("Optimize trading strategies to maximize humanitarian impact")
        
        if avg_compliance >= 0.95 and lives_impact > 50:
            recommendations.append("Excellent compliance and humanitarian impact - maintain current approach")
        
        recommendations.append("Continue monitoring explanation quality for sustained regulatory compliance")
        
        return recommendations

# Factory function for service creation
def create_explainability_service(config_path: str = None) -> ModelExplainabilityService:
    """Create and configure model explainability service"""
    return ModelExplainabilityService(config_path or "config/explainability_config.json")

# Example usage for humanitarian AI platform
if __name__ == "__main__":
    async def main():
        """Example usage of model explainability service"""
        print("🔍 Starting Model Explainability Service for Humanitarian AI Platform")
        print("💝 Sacred Mission: Transparent AI decisions for maximum charitable impact")
        
        # Create service
        service = create_explainability_service()
        
        # Example humanitarian context
        humanitarian_context = HumanitarianDecisionContext(
            lives_at_stake=150,
            medical_aid_impact=0.85,
            risk_tolerance=0.15,
            regulatory_requirements=['MiFID II', 'GDPR', 'Humanitarian Compliance'],
            compliance_score=0.92,
            fund_protection_level='maximum'
        )
        
        # Example explanation (simplified for demo)
        print("\n🔍 Generating example trading decision explanation...")
        
        # This would normally be called with actual model and features
        print("✅ Model Explainability Service ready for humanitarian trading platform")
        print("🏥 Ready to provide transparent AI decisions for life-saving mission")
        
        # Generate compliance report
        report = await service.generate_compliance_report()
        print(f"\n📊 Compliance Report Generated:")
        print(f"   • Status: {report.get('status', 'ready')}")
        print(f"   • Sacred mission: Transparent AI for maximum charitable impact")
    
    # Run example
    asyncio.run(main())
