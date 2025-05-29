"""
Market Sentiment Analysis Module
Advanced sentiment analysis for forex trading platform

This module provides comprehensive sentiment analysis capabilities including:
- Real-time news feed analysis
- Social media sentiment monitoring
- Multi-source sentiment aggregation
- AI-powered sentiment classification

Author: Platform3 Development Team
Version: 1.0.0
"""

from .SentimentAnalyzer import (
    SentimentAnalyzer,
    SentimentData,
    SentimentConfig,
    SentimentScore
)

from .NewsScraper import (
    NewsScraper,
    NewsArticle,
    ScraperConfig
)

from .SocialMediaIntegrator import (
    SocialMediaIntegrator,
    SocialMediaPost,
    SocialMediaConfig
)

__all__ = [
    # Main sentiment analyzer
    'SentimentAnalyzer',
    'SentimentData',
    'SentimentConfig',
    'SentimentScore',
    
    # News scraping
    'NewsScraper',
    'NewsArticle',
    'ScraperConfig',
    
    # Social media integration
    'SocialMediaIntegrator',
    'SocialMediaPost',
    'SocialMediaConfig'
]

__version__ = "1.0.0"
__author__ = "Platform3 Development Team"
