"""
Sentiment Pipeline for Market Sentiment Analysis

This module provides comprehensive sentiment analysis capabilities for trading
applications, integrating news feeds, social media, and other text sources
to generate sentiment features for ML models.

Key Features:
- Multi-source sentiment analysis
- Real-time sentiment processing
- Sentiment aggregation and scoring
- Integration with trading models
- Historical sentiment tracking
- Sentiment-based feature engineering

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import asyncio
import re
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
try:
    from textblob import TextBlob
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    NLTK_AVAILABLE = True
    TRANSFORMERS_AVAILABLE = True

    # Download required NLTK data
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)

except ImportError:
    NLTK_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = False
    logging.warning("NLP libraries not available. Using mock implementations.")

logger = logging.getLogger(__name__)

class SentimentSource(Enum):
    """Sources of sentiment data."""
    NEWS = "news"
    TWITTER = "twitter"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    FINANCIAL_REPORTS = "financial_reports"
    ANALYST_REPORTS = "analyst_reports"
    GENERAL = "general"

class SentimentModel(Enum):
    """Sentiment analysis models."""
    TEXTBLOB = "textblob"
    VADER = "vader"
    FINBERT = "finbert"
    ROBERTA = "roberta"
    CUSTOM = "custom"

@dataclass
class SentimentConfig:
    """Configuration for sentiment pipeline."""
    models: List[SentimentModel] = field(default_factory=lambda: [SentimentModel.VADER, SentimentModel.TEXTBLOB])
    sources: List[SentimentSource] = field(default_factory=lambda: [SentimentSource.NEWS, SentimentSource.TWITTER])
    aggregation_method: str = "weighted_average"
    time_window: int = 24  # hours
    min_confidence: float = 0.1
    max_texts_per_source: int = 1000
    language: str = "en"
    currency_pairs: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY"])

@dataclass
class SentimentResult:
    """Result from sentiment analysis."""
    sentiment_scores: Dict[str, float]
    confidence_scores: Dict[str, float]
    source_breakdown: Dict[SentimentSource, Dict[str, float]]
    aggregated_sentiment: float
    sentiment_trend: float
    processed_texts: int
    computation_time: float
    timestamp: datetime

class SentimentPipeline:
    """
    Comprehensive Sentiment Analysis Pipeline

    Processes text data from multiple sources to generate sentiment features
    for trading models with real-time capabilities.
    """

    def __init__(self, config: SentimentConfig = None):
        """
        Initialize sentiment pipeline.

        Args:
            config: Sentiment analysis configuration
        """
        self.config = config or SentimentConfig()
        self.models = {}
        self.sentiment_history = []
        self.source_weights = {
            SentimentSource.NEWS: 0.4,
            SentimentSource.TWITTER: 0.2,
            SentimentSource.REDDIT: 0.15,
            SentimentSource.TELEGRAM: 0.1,
            SentimentSource.FINANCIAL_REPORTS: 0.1,
            SentimentSource.ANALYST_REPORTS: 0.05
        }

        # Initialize models
        self._initialize_models()

        logger.info(f"SentimentPipeline initialized with models: {[m.value for m in self.config.models]}")

    def _initialize_models(self):
        """Initialize sentiment analysis models."""
        for model in self.config.models:
            try:
                if model == SentimentModel.TEXTBLOB and NLTK_AVAILABLE:
                    self.models[model] = self._textblob_sentiment
                elif model == SentimentModel.VADER and NLTK_AVAILABLE:
                    self.models[model] = SentimentIntensityAnalyzer()
                elif model == SentimentModel.FINBERT and TRANSFORMERS_AVAILABLE:
                    self.models[model] = pipeline(
                        "sentiment-analysis",
                        model="ProsusAI/finbert",
                        tokenizer="ProsusAI/finbert"
                    )
                elif model == SentimentModel.ROBERTA and TRANSFORMERS_AVAILABLE:
                    self.models[model] = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                    )
                else:
                    # Mock model
                    self.models[model] = self._mock_sentiment

            except Exception as e:
                logger.warning(f"Failed to initialize {model.value}: {e}. Using mock model.")
                self.models[model] = self._mock_sentiment

    async def analyze_sentiment(self,
                               texts: Dict[SentimentSource, List[str]],
                               metadata: Optional[Dict] = None) -> SentimentResult:
        """
        Analyze sentiment from multiple text sources.

        Args:
            texts: Dictionary mapping sources to text lists
            metadata: Optional metadata for analysis

        Returns:
            Comprehensive sentiment analysis result
        """
        start_time = datetime.now()
        logger.info("Starting sentiment analysis...")

        # Validate input
        if not texts:
            raise ValueError("No texts provided for sentiment analysis")

        # Process each source
        source_results = {}
        total_processed = 0

        for source, text_list in texts.items():
            if source in self.config.sources:
                # Limit texts per source
                limited_texts = text_list[:self.config.max_texts_per_source]
                source_result = await self._analyze_source_sentiment(source, limited_texts)
                source_results[source] = source_result
                total_processed += len(limited_texts)

        # Aggregate results
        aggregated_sentiment = await self._aggregate_sentiment(source_results)

        # Calculate sentiment scores for each model
        sentiment_scores = {}
        confidence_scores = {}

        for model in self.config.models:
            model_scores = []
            model_confidences = []

            for source_result in source_results.values():
                if model.value in source_result:
                    model_scores.append(source_result[model.value]['sentiment'])
                    model_confidences.append(source_result[model.value]['confidence'])

            if model_scores:
                sentiment_scores[model.value] = np.mean(model_scores)
                confidence_scores[model.value] = np.mean(model_confidences)

        # Calculate sentiment trend
        sentiment_trend = await self._calculate_sentiment_trend(aggregated_sentiment)

        computation_time = (datetime.now() - start_time).total_seconds()

        result = SentimentResult(
            sentiment_scores=sentiment_scores,
            confidence_scores=confidence_scores,
            source_breakdown=source_results,
            aggregated_sentiment=aggregated_sentiment,
            sentiment_trend=sentiment_trend,
            processed_texts=total_processed,
            computation_time=computation_time,
            timestamp=datetime.now()
        )

        # Store in history
        self.sentiment_history.append(result)

        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=self.config.time_window * 2)
        self.sentiment_history = [
            r for r in self.sentiment_history
            if r.timestamp > cutoff_time
        ]

        logger.info(f"Sentiment analysis completed in {computation_time:.2f}s. "
                   f"Processed {total_processed} texts. "
                   f"Aggregated sentiment: {aggregated_sentiment:.3f}")

        return result

    async def _analyze_source_sentiment(self,
                                       source: SentimentSource,
                                       texts: List[str]) -> Dict[str, Dict[str, float]]:
        """Analyze sentiment for a specific source."""
        source_results = {}

        for model in self.config.models:
            model_sentiments = []
            model_confidences = []

            for text in texts:
                if not text or len(text.strip()) < 10:
                    continue

                try:
                    sentiment, confidence = await self._get_model_sentiment(model, text)

                    if confidence >= self.config.min_confidence:
                        model_sentiments.append(sentiment)
                        model_confidences.append(confidence)

                except Exception as e:
                    logger.warning(f"Error analyzing text with {model.value}: {e}")
                    continue

            if model_sentiments:
                source_results[model.value] = {
                    'sentiment': np.mean(model_sentiments),
                    'confidence': np.mean(model_confidences),
                    'count': len(model_sentiments)
                }

        return source_results

    async def _get_model_sentiment(self,
                                  model: SentimentModel,
                                  text: str) -> Tuple[float, float]:
        """Get sentiment score from a specific model."""
        # Clean text
        cleaned_text = self._clean_text(text)

        if model == SentimentModel.TEXTBLOB:
            return await self._textblob_sentiment(cleaned_text)
        elif model == SentimentModel.VADER:
            return await self._vader_sentiment(cleaned_text)
        elif model == SentimentModel.FINBERT:
            return await self._finbert_sentiment(cleaned_text)
        elif model == SentimentModel.ROBERTA:
            return await self._roberta_sentiment(cleaned_text)
        else:
            return await self._mock_sentiment(cleaned_text)

    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)

        return text

    async def _textblob_sentiment(self, text: str) -> Tuple[float, float]:
        """Get sentiment using TextBlob."""
        if not NLTK_AVAILABLE:
            return await self._mock_sentiment(text)

        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity  # -1 to 1
            confidence = abs(blob.sentiment.subjectivity)  # 0 to 1
            return sentiment, confidence
        except:
            return 0.0, 0.0

    async def _vader_sentiment(self, text: str) -> Tuple[float, float]:
        """Get sentiment using VADER."""
        if SentimentModel.VADER not in self.models:
            return await self._mock_sentiment(text)

        try:
            scores = self.models[SentimentModel.VADER].polarity_scores(text)
            sentiment = scores['compound']  # -1 to 1
            confidence = max(scores['pos'], scores['neg'], scores['neu'])
            return sentiment, confidence
        except:
            return 0.0, 0.0

    async def _finbert_sentiment(self, text: str) -> Tuple[float, float]:
        """Get sentiment using FinBERT."""
        if SentimentModel.FINBERT not in self.models:
            return await self._mock_sentiment(text)

        try:
            result = self.models[SentimentModel.FINBERT](text)[0]
            label = result['label'].lower()
            confidence = result['score']

            # Convert to -1 to 1 scale
            if label == 'positive':
                sentiment = confidence
            elif label == 'negative':
                sentiment = -confidence
            else:  # neutral
                sentiment = 0.0

            return sentiment, confidence
        except:
            return 0.0, 0.0

    async def _roberta_sentiment(self, text: str) -> Tuple[float, float]:
        """Get sentiment using RoBERTa."""
        if SentimentModel.ROBERTA not in self.models:
            return await self._mock_sentiment(text)

        try:
            result = self.models[SentimentModel.ROBERTA](text)[0]
            label = result['label'].lower()
            confidence = result['score']

            # Convert to -1 to 1 scale
            if 'positive' in label:
                sentiment = confidence
            elif 'negative' in label:
                sentiment = -confidence
            else:  # neutral
                sentiment = 0.0

            return sentiment, confidence
        except:
            return 0.0, 0.0

    async def _mock_sentiment(self, text: str) -> Tuple[float, float]:
        """Mock sentiment analysis."""
        # Simple keyword-based sentiment
        positive_words = ['good', 'great', 'excellent', 'positive', 'up', 'rise', 'gain', 'profit', 'bull']
        negative_words = ['bad', 'terrible', 'negative', 'down', 'fall', 'loss', 'bear', 'crash', 'decline']

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        total_words = len(text.split())
        if total_words == 0:
            return 0.0, 0.0

        sentiment = (pos_count - neg_count) / max(total_words, 1)
        sentiment = max(-1.0, min(1.0, sentiment * 10))  # Scale and clamp
        confidence = min(1.0, (pos_count + neg_count) / max(total_words, 1) * 5)

        return sentiment, confidence

    async def _aggregate_sentiment(self,
                                  source_results: Dict[SentimentSource, Dict[str, Dict[str, float]]]) -> float:
        """Aggregate sentiment across sources and models."""
        if self.config.aggregation_method == "weighted_average":
            return await self._weighted_average_aggregation(source_results)
        elif self.config.aggregation_method == "simple_average":
            return await self._simple_average_aggregation(source_results)
        else:
            return await self._weighted_average_aggregation(source_results)

    async def _weighted_average_aggregation(self,
                                          source_results: Dict[SentimentSource, Dict[str, Dict[str, float]]]) -> float:
        """Aggregate using weighted average."""
        total_weighted_sentiment = 0.0
        total_weight = 0.0

        for source, models_data in source_results.items():
            source_weight = self.source_weights.get(source, 0.1)

            # Average across models for this source
            source_sentiments = []
            source_confidences = []

            for model_data in models_data.values():
                source_sentiments.append(model_data['sentiment'])
                source_confidences.append(model_data['confidence'])

            if source_sentiments:
                # Weight by confidence
                weighted_sentiment = np.average(source_sentiments, weights=source_confidences)
                avg_confidence = np.mean(source_confidences)

                # Apply source weight and confidence weight
                final_weight = source_weight * avg_confidence
                total_weighted_sentiment += weighted_sentiment * final_weight
                total_weight += final_weight

        return total_weighted_sentiment / total_weight if total_weight > 0 else 0.0

    async def _simple_average_aggregation(self,
                                        source_results: Dict[SentimentSource, Dict[str, Dict[str, float]]]) -> float:
        """Aggregate using simple average."""
        all_sentiments = []

        for models_data in source_results.values():
            for model_data in models_data.values():
                all_sentiments.append(model_data['sentiment'])

        return np.mean(all_sentiments) if all_sentiments else 0.0

    async def _calculate_sentiment_trend(self, current_sentiment: float) -> float:
        """Calculate sentiment trend based on history."""
        if len(self.sentiment_history) < 2:
            return 0.0

        # Get recent sentiments
        recent_sentiments = [r.aggregated_sentiment for r in self.sentiment_history[-10:]]
        recent_sentiments.append(current_sentiment)

        # Calculate trend using linear regression slope
        if len(recent_sentiments) >= 3:
            x = np.arange(len(recent_sentiments))
            y = np.array(recent_sentiments)

            # Simple linear regression
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)

            return slope

        return 0.0

    def get_sentiment_features(self,
                              lookback_hours: int = 24) -> pd.DataFrame:
        """
        Get sentiment features for ML models.

        Args:
            lookback_hours: Hours to look back for features

        Returns:
            DataFrame with sentiment features
        """
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_results = [
            r for r in self.sentiment_history
            if r.timestamp > cutoff_time
        ]

        if not recent_results:
            # Return empty features
            return pd.DataFrame()

        features = []
        timestamps = []

        for result in recent_results:
            feature_dict = {
                'aggregated_sentiment': result.aggregated_sentiment,
                'sentiment_trend': result.sentiment_trend,
                'processed_texts': result.processed_texts,
                'confidence_avg': np.mean(list(result.confidence_scores.values())) if result.confidence_scores else 0.0
            }

            # Add model-specific features
            for model, score in result.sentiment_scores.items():
                feature_dict[f'sentiment_{model}'] = score

            # Add source-specific features
            for source, source_data in result.source_breakdown.items():
                for model, model_data in source_data.items():
                    feature_dict[f'sentiment_{source.value}_{model}'] = model_data['sentiment']
                    feature_dict[f'confidence_{source.value}_{model}'] = model_data['confidence']

            features.append(feature_dict)
            timestamps.append(result.timestamp)

        df = pd.DataFrame(features, index=timestamps)

        # Add rolling statistics
        if len(df) > 1:
            df['sentiment_ma_1h'] = df['aggregated_sentiment'].rolling('1H').mean()
            df['sentiment_ma_6h'] = df['aggregated_sentiment'].rolling('6H').mean()
            df['sentiment_volatility'] = df['aggregated_sentiment'].rolling('1H').std()
            df['sentiment_momentum'] = df['aggregated_sentiment'].diff()

        return df.fillna(0)