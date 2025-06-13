"""
News Article Sentiment Analyzer for Platform3

A comprehensive news analysis system that processes financial news articles,
extracts sentiment, relevance, and impact indicators for market prediction.
Includes NLP processing, sentiment scoring, and news impact modeling.

Author: Platform3 Development Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import re
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

try:
    import requests
    from urllib.parse import urlencode
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


class NewsArticle:
    """
    News Article Sentiment and Impact Analyzer
    
    This analyzer processes financial news articles to extract:
    - Sentiment scores (positive, negative, neutral)
    - Market relevance and impact
    - Entity extraction (companies, sectors)
    - Temporal impact modeling
    - News volume and frequency analysis
    - Market moving news detection
    
    Features:
    - Multiple sentiment analysis engines
    - Financial keyword detection
    - Impact scoring based on source credibility
    - Time-decay modeling for news impact
    - Aggregated sentiment signals
    """
    
    def __init__(self,
                 sources: List[str] = None,
                 financial_keywords: List[str] = None,
                 sentiment_threshold: float = 0.1,
                 impact_decay_hours: float = 24.0,
                 volume_window: int = 24,
                 credibility_weights: Dict[str, float] = None):
        """
        Initialize News Article Analyzer
        
        Args:
            sources: List of news sources to monitor
            financial_keywords: Keywords for financial relevance
            sentiment_threshold: Minimum sentiment magnitude
            impact_decay_hours: Hours for news impact decay
            volume_window: Hours for news volume analysis
            credibility_weights: Source credibility weights
        """
        self.sources = sources or [
            'reuters', 'bloomberg', 'wsj', 'cnbc', 'marketwatch',
            'yahoo_finance', 'seeking_alpha', 'benzinga'
        ]
        
        self.financial_keywords = financial_keywords or [
            'earnings', 'revenue', 'profit', 'loss', 'merger', 'acquisition',
            'ipo', 'dividend', 'split', 'buyback', 'guidance', 'forecast',
            'beats', 'misses', 'upgrade', 'downgrade', 'target', 'rating',
            'federal reserve', 'fed', 'interest rate', 'inflation', 'gdp',
            'unemployment', 'housing', 'manufacturing', 'consumer confidence',
            'trade war', 'tariff', 'brexit', 'election', 'policy', 'regulation'
        ]
        
        self.sentiment_threshold = sentiment_threshold
        self.impact_decay_hours = impact_decay_hours
        self.volume_window = volume_window
        
        self.credibility_weights = credibility_weights or {
            'reuters': 1.0,
            'bloomberg': 1.0,
            'wsj': 0.95,
            'cnbc': 0.8,
            'marketwatch': 0.7,
            'yahoo_finance': 0.6,
            'seeking_alpha': 0.5,
            'benzinga': 0.6,
            'unknown': 0.3
        }
        
        # News storage
        self.news_articles = []
        self.processed_articles = []
        self.sentiment_history = []
        
        # Initialize NLP tools
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP processing tools"""
        self.sentiment_analyzer = None
        self.lemmatizer = None
        self.stop_words = set()
        
        if NLTK_AVAILABLE:
            try:
                # Try to download required NLTK data
                nltk.download('vader_lexicon', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
                
            except Exception as e:
                print(f"Warning: NLTK initialization failed: {e}")
        
        if not NLTK_AVAILABLE:
            print("Warning: NLTK not available. Using basic sentiment analysis.")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_financial_relevance(self, text: str) -> float:
        """
        Calculate financial relevance score
        
        Args:
            text: Article text
            
        Returns:
            Relevance score (0-1)
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        keyword_count = 0
        total_words = len(text_lower.split())
        
        if total_words == 0:
            return 0.0
        
        # Count financial keywords
        for keyword in self.financial_keywords:
            keyword_count += text_lower.count(keyword.lower())
        
        # Calculate relevance score
        relevance = min(1.0, keyword_count / max(1, total_words / 50))
        
        return relevance
    
    def _analyze_sentiment_nltk(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using NLTK VADER
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment scores
        """
        if not NLTK_AVAILABLE or not self.sentiment_analyzer:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'compound': scores['compound']
            }
        except Exception as e:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
    
    def _analyze_sentiment_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment scores
        """
        if not TEXTBLOB_AVAILABLE:
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def _analyze_sentiment_basic(self, text: str) -> Dict[str, float]:
        """
        Basic sentiment analysis using keyword matching
        
        Args:
            text: Text to analyze
            
        Returns:
            Basic sentiment scores
        """
        positive_words = [
            'buy', 'bull', 'bullish', 'gain', 'gains', 'profit', 'profits',
            'up', 'rise', 'rising', 'surge', 'rally', 'strong', 'strength',
            'beat', 'beats', 'outperform', 'upgrade', 'positive', 'optimistic',
            'growth', 'expand', 'expansion', 'record', 'high', 'peak'
        ]
        
        negative_words = [
            'sell', 'bear', 'bearish', 'loss', 'losses', 'down', 'fall',
            'falling', 'drop', 'decline', 'crash', 'weak', 'weakness',
            'miss', 'misses', 'underperform', 'downgrade', 'negative',
            'pessimistic', 'recession', 'crisis', 'low', 'bottom'
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_words = len(words)
        
        if total_words == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
        
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        neutral_score = 1.0 - positive_score - negative_score
        compound_score = positive_score - negative_score
        
        return {
            'positive': positive_score,
            'negative': negative_score,
            'neutral': max(0.0, neutral_score),
            'compound': compound_score
        }
    
    def _calculate_impact_score(self, article: Dict) -> float:
        """
        Calculate market impact score for article
        
        Args:
            article: Article data
            
        Returns:
            Impact score (0-1)
        """
        # Base impact from sentiment magnitude
        sentiment = article.get('sentiment', {})
        sentiment_magnitude = abs(sentiment.get('compound', 0.0))
        
        # Relevance factor
        relevance = article.get('relevance', 0.0)
        
        # Source credibility
        source = article.get('source', 'unknown')
        credibility = self.credibility_weights.get(source, 0.3)
        
        # Recency factor (news impact decays over time)
        publish_time = article.get('publish_time', datetime.now())
        if isinstance(publish_time, str):
            try:
                publish_time = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
            except:
                publish_time = datetime.now()
        
        hours_ago = (datetime.now() - publish_time).total_seconds() / 3600
        time_decay = np.exp(-hours_ago / self.impact_decay_hours)
        
        # Calculate overall impact
        impact = sentiment_magnitude * relevance * credibility * time_decay
        
        return min(1.0, impact)
    
    def process_article(self, article_data: Dict) -> Dict[str, Any]:
        """
        Process a single news article
        
        Args:
            article_data: Raw article data
            
        Returns:
            Processed article with sentiment and impact
        """
        # Extract and clean text
        title = self._clean_text(article_data.get('title', ''))
        content = self._clean_text(article_data.get('content', ''))
        
        # Combine title and content (title weighted more)
        full_text = f"{title} {title} {content}"  # Title appears twice for emphasis
        
        if not full_text.strip():
            return None
        
        # Calculate financial relevance
        relevance = self._extract_financial_relevance(full_text)
        
        # Skip if not financially relevant
        if relevance < 0.1:
            return None
        
        # Analyze sentiment using available methods
        sentiment_scores = {}
        
        # NLTK VADER
        nltk_sentiment = self._analyze_sentiment_nltk(full_text)
        sentiment_scores['nltk'] = nltk_sentiment
        
        # TextBlob
        textblob_sentiment = self._analyze_sentiment_textblob(full_text)
        sentiment_scores['textblob'] = textblob_sentiment
        
        # Basic sentiment
        basic_sentiment = self._analyze_sentiment_basic(full_text)
        sentiment_scores['basic'] = basic_sentiment
        
        # Aggregate sentiment
        compound_scores = []
        if 'compound' in nltk_sentiment:
            compound_scores.append(nltk_sentiment['compound'])
        if 'polarity' in textblob_sentiment:
            compound_scores.append(textblob_sentiment['polarity'])
        if 'compound' in basic_sentiment:
            compound_scores.append(basic_sentiment['compound'])
        
        if compound_scores:
            avg_compound = np.mean(compound_scores)
        else:
            avg_compound = 0.0
        
        # Create processed article
        processed = {
            'title': title,
            'content': content[:1000],  # Truncate content
            'source': article_data.get('source', 'unknown'),
            'publish_time': article_data.get('publish_time', datetime.now()),
            'url': article_data.get('url', ''),
            'relevance': relevance,
            'sentiment': {
                'compound': avg_compound,
                'positive': np.mean([s.get('positive', 0) for s in [nltk_sentiment, basic_sentiment]]),
                'negative': np.mean([s.get('negative', 0) for s in [nltk_sentiment, basic_sentiment]]),
                'neutral': np.mean([s.get('neutral', 1) for s in [nltk_sentiment, basic_sentiment]]),
                'detailed': sentiment_scores
            },
            'processed_time': datetime.now()
        }
        
        # Calculate impact score
        processed['impact'] = self._calculate_impact_score(processed)
        
        return processed
    
    def add_article(self, article_data: Dict) -> bool:
        """
        Add and process a news article
        
        Args:
            article_data: Article data
            
        Returns:
            Success status
        """
        try:
            processed = self.process_article(article_data)
            if processed:
                self.processed_articles.append(processed)
                
                # Keep only recent articles
                cutoff_time = datetime.now() - timedelta(hours=self.volume_window * 2)
                self.processed_articles = [
                    a for a in self.processed_articles 
                    if a['processed_time'] > cutoff_time
                ]
                
                return True
            return False
            
        except Exception as e:
            print(f"Error processing article: {e}")
            return False
    
    def add_articles_batch(self, articles: List[Dict]) -> int:
        """
        Add multiple articles in batch
        
        Args:
            articles: List of article data
            
        Returns:
            Number of successfully processed articles
        """
        processed_count = 0
        for article in articles:
            if self.add_article(article):
                processed_count += 1
        
        return processed_count
    
    def get_sentiment_signal(self, lookback_hours: float = 24.0) -> Dict[str, float]:
        """
        Calculate aggregated sentiment signal
        
        Args:
            lookback_hours: Hours to look back for articles
            
        Returns:
            Sentiment signal metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        # Filter recent articles
        recent_articles = [
            a for a in self.processed_articles
            if a['processed_time'] > cutoff_time and a['impact'] > 0.1
        ]
        
        if not recent_articles:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'article_count': 0,
                'impact_weighted_sentiment': 0.0,
                'news_volume': 0
            }
        
        # Calculate metrics
        sentiments = [a['sentiment']['compound'] for a in recent_articles]
        impacts = [a['impact'] for a in recent_articles]
        
        # Simple average sentiment
        avg_sentiment = np.mean(sentiments)
        
        # Impact-weighted sentiment
        total_impact = sum(impacts)
        if total_impact > 0:
            weighted_sentiment = sum(s * i for s, i in zip(sentiments, impacts)) / total_impact
        else:
            weighted_sentiment = avg_sentiment
        
        # Confidence based on article count and impact
        confidence = min(1.0, len(recent_articles) / 10.0 + total_impact / 5.0)
        
        return {
            'sentiment_score': avg_sentiment,
            'confidence': confidence,
            'article_count': len(recent_articles),
            'impact_weighted_sentiment': weighted_sentiment,
            'news_volume': len(recent_articles),
            'total_impact': total_impact
        }
    
    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate news sentiment signals for backtesting
        
        Args:
            data: Market data (used for timing)
            
        Returns:
            Array of sentiment signals
        """
        signals = np.zeros(len(data))
        
        # For backtesting, we would need historical news data
        # This is a simplified implementation
        
        # Simulate some news events (in real implementation, use actual news data)
        np.random.seed(42)
        
        for i in range(len(data)):
            # Simulate random news events
            if np.random.random() < 0.1:  # 10% chance of news event
                # Create simulated article
                sentiment_value = np.random.normal(0, 0.3)  # Random sentiment
                impact = np.random.exponential(0.2)  # Random impact
                
                article = {
                    'title': f'Market News {i}',
                    'content': 'Simulated news content for backtesting',
                    'source': 'simulated',
                    'publish_time': data.index[i] if hasattr(data, 'index') else datetime.now(),
                    'url': 'http://example.com'
                }
                
                # Process article
                processed = self.process_article(article)
                if processed:
                    # Override with simulated values
                    processed['sentiment']['compound'] = sentiment_value
                    processed['impact'] = min(1.0, impact)
                    
                    self.add_article(article)
            
            # Get current signal
            signal_data = self.get_sentiment_signal(lookback_hours=24)
            signals[i] = signal_data['impact_weighted_sentiment'] * signal_data['confidence']
        
        return signals
    
    def get_news_summary(self) -> Dict[str, Any]:
        """
        Get summary of recent news analysis
        
        Returns:
            News analysis summary
        """
        if not self.processed_articles:
            return {'status': 'No articles processed'}
        
        recent_signal = self.get_sentiment_signal()
        
        # Top articles by impact
        top_articles = sorted(
            self.processed_articles[-50:],  # Last 50 articles
            key=lambda x: x['impact'],
            reverse=True
        )[:5]
        
        # Source distribution
        sources = {}
        for article in self.processed_articles[-100:]:  # Last 100 articles
            source = article['source']
            sources[source] = sources.get(source, 0) + 1
        
        return {
            'current_signal': recent_signal,
            'total_articles_processed': len(self.processed_articles),
            'top_impact_articles': [
                {
                    'title': a['title'][:100],
                    'source': a['source'],
                    'impact': a['impact'],
                    'sentiment': a['sentiment']['compound']
                }
                for a in top_articles
            ],
            'source_distribution': sources,
            'processing_stats': {
                'avg_relevance': np.mean([a['relevance'] for a in self.processed_articles[-50:]]) if self.processed_articles else 0,
                'avg_impact': np.mean([a['impact'] for a in self.processed_articles[-50:]]) if self.processed_articles else 0
            }
        }


# Test and example usage
if __name__ == "__main__":
    print("Testing News Article Sentiment Analyzer...")
    
    # Initialize analyzer
    news_analyzer = NewsArticle(
        sentiment_threshold=0.05,
        impact_decay_hours=12.0,
        volume_window=24
    )
    
    print(f"Initialized with {len(news_analyzer.sources)} sources")
    print(f"Financial keywords: {len(news_analyzer.financial_keywords)}")
    print(f"NLP tools available: NLTK={NLTK_AVAILABLE}, TextBlob={TEXTBLOB_AVAILABLE}")
    
    # Test with sample articles
    sample_articles = [
        {
            'title': 'Apple Reports Record Quarterly Earnings, Beats Analyst Expectations',
            'content': 'Apple Inc. today announced financial results for its fiscal 2024 first quarter ended December 30, 2023. The company posted quarterly revenue of $119.6 billion, up 2 percent year over year, and quarterly earnings per diluted share of $2.18, up 16 percent year over year.',
            'source': 'reuters',
            'publish_time': datetime.now() - timedelta(hours=2),
            'url': 'https://example.com/apple-earnings'
        },
        {
            'title': 'Federal Reserve Raises Interest Rates by 0.25%, Signals More Hikes Ahead',
            'content': 'The Federal Reserve raised its benchmark interest rate by a quarter percentage point on Wednesday and signaled that more increases are likely this year as policymakers work to bring down inflation.',
            'source': 'bloomberg',
            'publish_time': datetime.now() - timedelta(hours=5),
            'url': 'https://example.com/fed-rates'
        },
        {
            'title': 'Tesla Stock Tumbles on Production Concerns and Delivery Miss',
            'content': 'Tesla shares fell sharply in premarket trading after the electric vehicle maker reported lower-than-expected deliveries for the quarter, raising concerns about production capacity and demand.',
            'source': 'cnbc',
            'publish_time': datetime.now() - timedelta(hours=8),
            'url': 'https://example.com/tesla-stock'
        },
        {
            'title': 'Weather Update: Sunny Skies Expected This Weekend',
            'content': 'The weather forecast shows sunny conditions for the weekend with temperatures in the 70s. Perfect weather for outdoor activities.',
            'source': 'weather.com',
            'publish_time': datetime.now() - timedelta(hours=1),
            'url': 'https://example.com/weather'
        }
    ]
    
    print(f"\\nProcessing {len(sample_articles)} sample articles...")
    
    processed_count = news_analyzer.add_articles_batch(sample_articles)
    print(f"Successfully processed {processed_count} articles")
    
    # Test sentiment analysis
    for i, article in enumerate(sample_articles[:3]):  # Test first 3 articles
        processed = news_analyzer.process_article(article)
        if processed:
            print(f"\\nArticle {i+1}: {article['title'][:50]}...")
            print(f"  Relevance: {processed['relevance']:.3f}")
            print(f"  Sentiment: {processed['sentiment']['compound']:.3f}")
            print(f"  Impact: {processed['impact']:.3f}")
            print(f"  Source credibility: {news_analyzer.credibility_weights.get(processed['source'], 0.3)}")
    
    # Test signal generation
    print("\\nTesting sentiment signals...")
    signal_data = news_analyzer.get_sentiment_signal(lookback_hours=24)
    
    print("Current Sentiment Signal:")
    for key, value in signal_data.items():
        print(f"  {key}: {value}")
    
    # Test with market data simulation
    print("\\nTesting with simulated market data...")
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    market_data = pd.DataFrame({
        'date': dates,
        'close': 100 + np.cumsum(np.random.randn(50) * 0.02)
    })
    
    signals = news_analyzer.calculate(market_data)
    print(f"Generated {len(signals)} signals")
    print(f"Signal range: [{signals.min():.6f}, {signals.max():.6f}]")
    print(f"Non-zero signals: {np.count_nonzero(signals)}")
    print(f"Average absolute signal: {np.mean(np.abs(signals)):.6f}")
    
    # News summary
    print("\\nNews Analysis Summary:")
    summary = news_analyzer.get_news_summary()
    
    if 'current_signal' in summary:
        print("Current Signal:")
        for key, value in summary['current_signal'].items():
            print(f"  {key}: {value}")
    
    if 'top_impact_articles' in summary:
        print("\\nTop Impact Articles:")
        for article in summary['top_impact_articles']:
            print(f"  {article['title']} (Impact: {article['impact']:.3f})")
    
    print("\\nNews Article Sentiment Analyzer test completed successfully!")