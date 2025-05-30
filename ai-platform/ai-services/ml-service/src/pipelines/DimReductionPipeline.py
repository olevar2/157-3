"""
Dimensionality Reduction Pipeline for Feature Optimization

This module provides comprehensive dimensionality reduction techniques for
optimizing feature sets in machine learning models. It includes PCA, ICA,
and other advanced dimensionality reduction methods.

Key Features:
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)
- t-SNE for visualization
- UMAP for non-linear reduction
- Feature importance analysis
- Explained variance tracking
- Real-time dimensionality reduction

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ML libraries
try:
    from sklearn.decomposition import PCA, FastICA, TruncatedSVD
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_regression
    import umap
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Using mock implementations.")

logger = logging.getLogger(__name__)

class DimReductionMethod(Enum):
    """Dimensionality reduction methods."""
    PCA = "pca"
    ICA = "ica"
    TSNE = "tsne"
    UMAP = "umap"
    SVD = "svd"
    FEATURE_SELECTION = "feature_selection"

@dataclass
class ComponentAnalysis:
    """Analysis of principal components."""
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    components: np.ndarray
    feature_importance: Dict[str, float]
    optimal_components: int

@dataclass
class DimReductionResult:
    """Result from dimensionality reduction."""
    reduced_features: pd.DataFrame
    original_features: pd.DataFrame
    method: DimReductionMethod
    n_components: int
    explained_variance: float
    component_analysis: ComponentAnalysis
    feature_mapping: Dict[str, List[str]]
    computation_time: float
    reduction_ratio: float

class DimReductionPipeline:
    """
    Comprehensive Dimensionality Reduction Pipeline

    Provides various dimensionality reduction techniques for feature optimization
    in machine learning models with real-time processing capabilities.
    """

    def __init__(self,
                 method: DimReductionMethod = DimReductionMethod.PCA,
                 n_components: Optional[int] = None,
                 variance_threshold: float = 0.95,
                 max_workers: int = 4):
        """
        Initialize dimensionality reduction pipeline.

        Args:
            method: Reduction method to use
            n_components: Number of components (None for auto-selection)
            variance_threshold: Minimum variance to retain
            max_workers: Maximum worker threads
        """
        self.method = method
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.max_workers = max_workers

        # Initialize components
        self.reducer = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_selector = None
        self.is_fitted = False

        # Cache for performance
        self.feature_cache = {}
        self.component_cache = {}

        logger.info(f"DimReductionPipeline initialized with method: {method.value}")

    async def fit_transform(self,
                           X: pd.DataFrame,
                           y: Optional[pd.Series] = None) -> DimReductionResult:
        """
        Fit the dimensionality reduction model and transform data.

        Args:
            X: Input features
            y: Target variable (optional, for supervised methods)

        Returns:
            Dimensionality reduction result
        """
        start_time = datetime.now()
        logger.info(f"Starting dimensionality reduction with {self.method.value}...")

        # Validate input
        if X.empty:
            raise ValueError("Input features cannot be empty")

        # Prepare data
        X_processed = await self._prepare_data(X)

        # Determine optimal number of components
        if self.n_components is None:
            self.n_components = await self._determine_optimal_components(X_processed, y)

        # Fit and transform
        X_reduced = await self._fit_transform_method(X_processed, y)

        # Analyze components
        component_analysis = await self._analyze_components(X_processed, X_reduced)

        # Create feature mapping
        feature_mapping = self._create_feature_mapping(X.columns, X_reduced.columns)

        computation_time = (datetime.now() - start_time).total_seconds()
        reduction_ratio = X_reduced.shape[1] / X.shape[1]

        result = DimReductionResult(
            reduced_features=X_reduced,
            original_features=X,
            method=self.method,
            n_components=self.n_components,
            explained_variance=component_analysis.explained_variance_ratio.sum(),
            component_analysis=component_analysis,
            feature_mapping=feature_mapping,
            computation_time=computation_time,
            reduction_ratio=reduction_ratio
        )

        self.is_fitted = True
        logger.info(f"Dimensionality reduction completed in {computation_time:.2f}s. "
                   f"Reduced from {X.shape[1]} to {X_reduced.shape[1]} features "
                   f"({reduction_ratio:.2%} reduction)")

        return result

    async def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted reducer.

        Args:
            X: Input features to transform

        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        X_processed = await self._prepare_data(X)

        if SKLEARN_AVAILABLE and hasattr(self.reducer, 'transform'):
            X_reduced = self.reducer.transform(X_processed)
            return pd.DataFrame(X_reduced,
                              columns=[f'component_{i}' for i in range(X_reduced.shape[1])],
                              index=X.index)
        else:
            # Mock transformation
            n_components = min(self.n_components or 10, X_processed.shape[1])
            return X_processed.iloc[:, :n_components]

    async def _prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for dimensionality reduction."""
        # Handle missing values
        X_clean = X.fillna(X.mean())

        # Scale features if using PCA/ICA
        if self.method in [DimReductionMethod.PCA, DimReductionMethod.ICA] and self.scaler:
            if not hasattr(self.scaler, 'scale_'):
                X_scaled = pd.DataFrame(
                    self.scaler.fit_transform(X_clean),
                    columns=X_clean.columns,
                    index=X_clean.index
                )
            else:
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X_clean),
                    columns=X_clean.columns,
                    index=X_clean.index
                )
            return X_scaled

        return X_clean

    async def _determine_optimal_components(self,
                                          X: pd.DataFrame,
                                          y: Optional[pd.Series] = None) -> int:
        """Determine optimal number of components."""
        max_components = min(X.shape[0] - 1, X.shape[1])

        if self.method == DimReductionMethod.PCA and SKLEARN_AVAILABLE:
            # Use PCA to determine components for variance threshold
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            optimal_components = np.argmax(cumsum_var >= self.variance_threshold) + 1
            return min(optimal_components, max_components)

        elif self.method == DimReductionMethod.FEATURE_SELECTION and y is not None and SKLEARN_AVAILABLE:
            # Use statistical feature selection
            selector = SelectKBest(score_func=f_regression, k='all')
            selector.fit(X, y)
            scores = selector.scores_
            # Select features with scores above median
            threshold = np.median(scores)
            return min(sum(scores > threshold), max_components)

        # Default to 80% of features or 50, whichever is smaller
        return min(int(X.shape[1] * 0.8), 50, max_components)

    async def _fit_transform_method(self,
                                   X: pd.DataFrame,
                                   y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform using the specified method."""
        if SKLEARN_AVAILABLE:
            if self.method == DimReductionMethod.PCA:
                self.reducer = PCA(n_components=self.n_components, random_state=42)
                X_reduced = self.reducer.fit_transform(X)

            elif self.method == DimReductionMethod.ICA:
                self.reducer = FastICA(n_components=self.n_components, random_state=42, max_iter=1000)
                X_reduced = self.reducer.fit_transform(X)

            elif self.method == DimReductionMethod.SVD:
                self.reducer = TruncatedSVD(n_components=self.n_components, random_state=42)
                X_reduced = self.reducer.fit_transform(X)

            elif self.method == DimReductionMethod.TSNE:
                # t-SNE doesn't support transform, only fit_transform
                perplexity = min(30, (X.shape[0] - 1) // 3)
                self.reducer = TSNE(n_components=min(self.n_components, 3),
                                  perplexity=perplexity, random_state=42)
                X_reduced = self.reducer.fit_transform(X)

            elif self.method == DimReductionMethod.UMAP:
                try:
                    self.reducer = umap.UMAP(n_components=self.n_components, random_state=42)
                    X_reduced = self.reducer.fit_transform(X)
                except:
                    # Fallback to PCA if UMAP fails
                    logger.warning("UMAP failed, falling back to PCA")
                    self.reducer = PCA(n_components=self.n_components, random_state=42)
                    X_reduced = self.reducer.fit_transform(X)

            elif self.method == DimReductionMethod.FEATURE_SELECTION and y is not None:
                self.feature_selector = SelectKBest(score_func=f_regression, k=self.n_components)
                X_reduced = self.feature_selector.fit_transform(X, y)

            else:
                # Default to PCA
                self.reducer = PCA(n_components=self.n_components, random_state=42)
                X_reduced = self.reducer.fit_transform(X)
        else:
            # Mock implementation
            logger.warning("Using mock dimensionality reduction")
            X_reduced = X.iloc[:, :self.n_components].values

        # Create DataFrame with appropriate column names
        if self.method == DimReductionMethod.FEATURE_SELECTION and hasattr(self.feature_selector, 'get_support'):
            selected_features = X.columns[self.feature_selector.get_support()]
            columns = selected_features.tolist()
        else:
            columns = [f'component_{i}' for i in range(X_reduced.shape[1])]

        return pd.DataFrame(X_reduced, columns=columns, index=X.index)

    async def _analyze_components(self,
                                 X_original: pd.DataFrame,
                                 X_reduced: pd.DataFrame) -> ComponentAnalysis:
        """Analyze the components and their importance."""
        if SKLEARN_AVAILABLE and hasattr(self.reducer, 'explained_variance_ratio_'):
            explained_variance_ratio = self.reducer.explained_variance_ratio_
            components = self.reducer.components_
        elif SKLEARN_AVAILABLE and hasattr(self.reducer, 'singular_values_'):
            # For SVD
            total_var = np.sum(self.reducer.singular_values_ ** 2)
            explained_variance_ratio = (self.reducer.singular_values_ ** 2) / total_var
            components = self.reducer.components_
        else:
            # Mock analysis
            n_components = X_reduced.shape[1]
            explained_variance_ratio = np.linspace(0.3, 0.05, n_components)
            explained_variance_ratio = explained_variance_ratio / explained_variance_ratio.sum()
            components = np.random.randn(n_components, X_original.shape[1])

        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Calculate feature importance
        feature_importance = {}
        if len(components.shape) == 2:
            for i, feature in enumerate(X_original.columns):
                # Sum of absolute component loadings weighted by explained variance
                importance = np.sum(np.abs(components[:, i]) * explained_variance_ratio)
                feature_importance[feature] = float(importance)
        else:
            # Equal importance for mock data
            for feature in X_original.columns:
                feature_importance[feature] = 1.0 / len(X_original.columns)

        # Find optimal number of components for variance threshold
        optimal_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        optimal_components = min(optimal_components, len(explained_variance_ratio))

        return ComponentAnalysis(
            explained_variance_ratio=explained_variance_ratio,
            cumulative_variance=cumulative_variance,
            components=components,
            feature_importance=feature_importance,
            optimal_components=optimal_components
        )

    def _create_feature_mapping(self,
                               original_features: pd.Index,
                               reduced_features: pd.Index) -> Dict[str, List[str]]:
        """Create mapping between original and reduced features."""
        mapping = {}

        if self.method == DimReductionMethod.FEATURE_SELECTION:
            # Direct mapping for feature selection
            for reduced_feature in reduced_features:
                if reduced_feature in original_features:
                    mapping[reduced_feature] = [reduced_feature]
        else:
            # Component-based mapping
            for i, reduced_feature in enumerate(reduced_features):
                if hasattr(self.reducer, 'components_') and len(self.reducer.components_.shape) == 2:
                    # Find top contributing original features for this component
                    component_weights = np.abs(self.reducer.components_[i])
                    top_indices = np.argsort(component_weights)[-5:]  # Top 5 contributors
                    top_features = [original_features[idx] for idx in top_indices]
                    mapping[reduced_feature] = top_features
                else:
                    # Mock mapping
                    mapping[reduced_feature] = original_features[:5].tolist()

        return mapping

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before getting feature importance")

        if hasattr(self, 'component_analysis'):
            return self.component_analysis.feature_importance
        return {}

    def get_explained_variance(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get explained variance ratios and cumulative variance."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before getting explained variance")

        if hasattr(self, 'component_analysis'):
            return (self.component_analysis.explained_variance_ratio,
                   self.component_analysis.cumulative_variance)
        return np.array([]), np.array([])

    async def optimize_components(self,
                                 X: pd.DataFrame,
                                 y: Optional[pd.Series] = None,
                                 max_components: Optional[int] = None) -> Dict[str, float]:
        """
        Optimize the number of components based on various criteria.

        Args:
            X: Input features
            y: Target variable (optional)
            max_components: Maximum components to test

        Returns:
            Dictionary with optimal components for different criteria
        """
        if max_components is None:
            max_components = min(50, X.shape[1])

        results = {}

        # Test different numbers of components
        for n_comp in range(2, max_components + 1, 2):
            temp_pipeline = DimReductionPipeline(
                method=self.method,
                n_components=n_comp,
                variance_threshold=self.variance_threshold
            )

            try:
                result = await temp_pipeline.fit_transform(X, y)
                results[n_comp] = result.explained_variance
            except Exception as e:
                logger.warning(f"Failed to test {n_comp} components: {e}")
                continue

        if not results:
            return {'optimal_components': min(10, X.shape[1])}

        # Find optimal based on different criteria
        optimal_results = {}

        # 95% variance threshold
        for n_comp, variance in results.items():
            if variance >= 0.95:
                optimal_results['variance_95'] = n_comp
                break

        # 90% variance threshold
        for n_comp, variance in results.items():
            if variance >= 0.90:
                optimal_results['variance_90'] = n_comp
                break

        # Elbow method (largest improvement drop)
        variances = list(results.values())
        improvements = [variances[i] - variances[i-1] for i in range(1, len(variances))]
        if improvements:
            elbow_idx = np.argmin(improvements)
            optimal_results['elbow_method'] = list(results.keys())[elbow_idx + 1]

        # Default recommendation
        optimal_results['recommended'] = optimal_results.get('variance_95',
                                                           optimal_results.get('variance_90',
                                                                             min(20, X.shape[1])))

        return optimal_results
