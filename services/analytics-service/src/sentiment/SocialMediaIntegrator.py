#!/usr/bin/env python3
"""
Social Media Integrator for Market Sentiment Analysis
Integration with social media platforms for forex trading sentiment
Supports Twitter, Reddit, and Telegram for comprehensive market sentiment

Author: Platform3 Development Team
Version: 1.0.0
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import re
import hashlib
from collections import defaultdict, deque
import threading

# Third-party imports
import redis
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import requests
from requests.auth import HTTPBasicAuth
import tweepy
import praw
from telethon import TelegramClient
import asyncpraw

@dataclass
class SocialMediaPost:
    """Social media post data structure"""
    platform: str
    post_id: str
    author: str
    content: str
    timestamp: datetime
    likes: int = 0
    shares: int = 0
    comments: int = 0
    hashtags: List[str] = None
    mentions: List[str] = None
    url: Optional[str] = None
    hash_id: str = None
    
    def __post_init__(self):
        if self.hashtags is None:
            self.hashtags = []
        if self.mentions is None:
            self.mentions = []
        
        # Generate unique hash for deduplication
        if self.hash_id is None:
            content_hash = hashlib.md5(
                f"{self.platform}{self.post_id}{self.content}".encode('utf-8')
            ).hexdigest()
            self.hash_id = content_hash
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "post_id": self.post_id,
            "author": self.author,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "likes": self.likes,
            "shares": self.shares,
            "comments": self.comments,
            "hashtags": self.hashtags,
            "mentions": self.mentions,
            "url": self.url,
            "hash_id": self.hash_id
        }

@dataclass
class SocialMediaConfig:
    """Configuration for social media integration"""
    # Twitter API credentials
    twitter_bearer_token: Optional[str] = None
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    twitter_access_token: Optional[str] = None
    twitter_access_token_secret: Optional[str] = None
    
    # Reddit API credentials
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: str = "ForexTradingBot/1.0"
    
    # Telegram API credentials
    telegram_api_id: Optional[int] = None
    telegram_api_hash: Optional[str] = None
    telegram_phone: Optional[str] = None
    
    # Search parameters
    forex_keywords: List[str] = None
    currency_hashtags: List[str] = None
    max_posts_per_platform: int = 100
    update_interval: int = 600  # 10 minutes
    
    # Database settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "forex_trading"
    postgres_user: str = "forex_admin"
    postgres_password: str = "ForexSecure2025!"
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = "RedisSecure2025!"
    
    def __post_init__(self):
        if self.forex_keywords is None:
            self.forex_keywords = [
                'forex', 'fx', 'currency', 'trading', 'eurusd', 'gbpusd', 'usdjpy',
                'audusd', 'usdcad', 'usdchf', 'nzdusd', 'eurgbp', 'eurjpy',
                'gbpjpy', 'central bank', 'fed', 'ecb', 'boe', 'boj', 'interest rates'
            ]
        
        if self.currency_hashtags is None:
            self.currency_hashtags = [
                '#forex', '#fx', '#trading', '#eurusd', '#gbpusd', '#usdjpy',
                '#dollar', '#euro', '#pound', '#yen', '#centralbank', '#fed'
            ]

class SocialMediaIntegrator:
    """
    Social media integration for forex sentiment analysis
    Monitors Twitter, Reddit, and Telegram for market sentiment
    """
    
    def __init__(self, config: SocialMediaConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.running = False
        
        # Initialize API clients
        self.twitter_client = None
        self.reddit_client = None
        self.telegram_client = None
        
        self._init_api_clients()
        self._init_connections()
        
        # Post cache for deduplication
        self.processed_posts = set()
        
        # Statistics
        self.stats = {
            "posts_collected": 0,
            "posts_stored": 0,
            "duplicates_filtered": 0,
            "errors": 0,
            "platform_stats": defaultdict(int),
            "start_time": None
        }
        
        # Thread safety
        self.lock = threading.RLock()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger("SocialMediaIntegrator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_api_clients(self):
        """Initialize social media API clients"""
        try:
            # Initialize Twitter client
            if self.config.twitter_bearer_token:
                self.twitter_client = tweepy.Client(
                    bearer_token=self.config.twitter_bearer_token,
                    consumer_key=self.config.twitter_api_key,
                    consumer_secret=self.config.twitter_api_secret,
                    access_token=self.config.twitter_access_token,
                    access_token_secret=self.config.twitter_access_token_secret,
                    wait_on_rate_limit=True
                )
                self.logger.info("Twitter client initialized")
            
            # Initialize Reddit client
            if self.config.reddit_client_id and self.config.reddit_client_secret:
                self.reddit_client = praw.Reddit(
                    client_id=self.config.reddit_client_id,
                    client_secret=self.config.reddit_client_secret,
                    user_agent=self.config.reddit_user_agent
                )
                self.logger.info("Reddit client initialized")
            
            # Initialize Telegram client (placeholder)
            if self.config.telegram_api_id and self.config.telegram_api_hash:
                # Note: Telegram client initialization would require phone verification
                # This is a placeholder for production implementation
                self.logger.info("Telegram client placeholder initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing API clients: {e}")
    
    def _init_connections(self):
        """Initialize database connections"""
        try:
            # PostgreSQL connection pool
            self.pg_pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=8,
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            
            # Redis connection
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                decode_responses=True
            )
            
            self.logger.info("Database connections initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connections: {e}")
            raise
    
    async def start(self):
        """Start social media monitoring"""
        self.logger.info("Starting social media integrator...")
        self.running = True
        self.stats["start_time"] = time.time()
        
        # Load processed posts from cache
        await self._load_processed_posts()
        
        # Start monitoring tasks
        tasks = []
        
        if self.twitter_client:
            tasks.append(asyncio.create_task(self._monitor_twitter()))
        
        if self.reddit_client:
            tasks.append(asyncio.create_task(self._monitor_reddit()))
        
        if self.telegram_client:
            tasks.append(asyncio.create_task(self._monitor_telegram()))
        
        tasks.append(asyncio.create_task(self._monitor_performance()))
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in social media monitoring: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop social media monitoring"""
        self.logger.info("Stopping social media integrator...")
        self.running = False
        self._close_connections()
    
    async def _monitor_twitter(self):
        """Monitor Twitter for forex-related posts"""
        while self.running:
            try:
                self.logger.info("Monitoring Twitter for forex content...")
                
                # Search for forex-related tweets
                for keyword in self.config.forex_keywords[:5]:  # Limit to avoid rate limits
                    try:
                        tweets = self.twitter_client.search_recent_tweets(
                            query=f"{keyword} -is:retweet lang:en",
                            max_results=20,
                            tweet_fields=['created_at', 'author_id', 'public_metrics', 'entities']
                        )
                        
                        if tweets.data:
                            for tweet in tweets.data:
                                post = self._parse_twitter_post(tweet)
                                if post and await self._store_post(post):
                                    self.stats["platform_stats"]["twitter"] += 1
                        
                        # Rate limiting
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        self.logger.error(f"Error searching Twitter for '{keyword}': {e}")
                        self.stats["errors"] += 1
                
                # Wait before next monitoring cycle
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in Twitter monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def _parse_twitter_post(self, tweet) -> Optional[SocialMediaPost]:
        """Parse Twitter tweet into SocialMediaPost"""
        try:
            # Extract hashtags and mentions
            hashtags = []
            mentions = []
            
            if hasattr(tweet, 'entities') and tweet.entities:
                if 'hashtags' in tweet.entities:
                    hashtags = [tag['tag'] for tag in tweet.entities['hashtags']]
                if 'mentions' in tweet.entities:
                    mentions = [mention['username'] for mention in tweet.entities['mentions']]
            
            # Get engagement metrics
            likes = 0
            shares = 0
            comments = 0
            
            if hasattr(tweet, 'public_metrics') and tweet.public_metrics:
                likes = tweet.public_metrics.get('like_count', 0)
                shares = tweet.public_metrics.get('retweet_count', 0)
                comments = tweet.public_metrics.get('reply_count', 0)
            
            post = SocialMediaPost(
                platform="twitter",
                post_id=tweet.id,
                author=str(tweet.author_id),
                content=tweet.text,
                timestamp=tweet.created_at,
                likes=likes,
                shares=shares,
                comments=comments,
                hashtags=hashtags,
                mentions=mentions,
                url=f"https://twitter.com/user/status/{tweet.id}"
            )
            
            return post
            
        except Exception as e:
            self.logger.error(f"Error parsing Twitter post: {e}")
            return None
    
    async def _monitor_reddit(self):
        """Monitor Reddit for forex-related posts"""
        while self.running:
            try:
                self.logger.info("Monitoring Reddit for forex content...")
                
                # Monitor forex-related subreddits
                subreddits = ['forex', 'trading', 'SecurityAnalysis', 'investing', 'wallstreetbets']
                
                for subreddit_name in subreddits:
                    try:
                        subreddit = self.reddit_client.subreddit(subreddit_name)
                        
                        # Get hot posts
                        for submission in subreddit.hot(limit=10):
                            if self._is_forex_related(submission.title + " " + submission.selftext):
                                post = self._parse_reddit_post(submission)
                                if post and await self._store_post(post):
                                    self.stats["platform_stats"]["reddit"] += 1
                        
                        # Rate limiting
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        self.logger.error(f"Error monitoring subreddit '{subreddit_name}': {e}")
                        self.stats["errors"] += 1
                
                # Wait before next monitoring cycle
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in Reddit monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def _parse_reddit_post(self, submission) -> Optional[SocialMediaPost]:
        """Parse Reddit submission into SocialMediaPost"""
        try:
            content = f"{submission.title}\n{submission.selftext}"
            
            post = SocialMediaPost(
                platform="reddit",
                post_id=submission.id,
                author=str(submission.author) if submission.author else "deleted",
                content=content,
                timestamp=datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                likes=submission.score,
                shares=0,  # Reddit doesn't have shares
                comments=submission.num_comments,
                url=f"https://reddit.com{submission.permalink}"
            )
            
            return post
            
        except Exception as e:
            self.logger.error(f"Error parsing Reddit post: {e}")
            return None
    
    async def _monitor_telegram(self):
        """Monitor Telegram for forex-related content"""
        while self.running:
            try:
                # Placeholder for Telegram monitoring
                # In production, this would monitor forex-related Telegram channels
                self.logger.info("Telegram monitoring placeholder - implement with Telethon")
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in Telegram monitoring: {e}")
                await asyncio.sleep(300)
    
    def _is_forex_related(self, text: str) -> bool:
        """Check if text is related to forex trading"""
        text_lower = text.lower()
        
        # Check for forex keywords
        keyword_count = sum(1 for keyword in self.config.forex_keywords if keyword in text_lower)
        
        return keyword_count >= 1
    
    async def _store_post(self, post: SocialMediaPost) -> bool:
        """Store social media post in database"""
        try:
            # Check for duplicates
            if post.hash_id in self.processed_posts:
                self.stats["duplicates_filtered"] += 1
                return False
            
            # Check Redis cache for recent duplicates
            cache_key = f"social_post:{post.hash_id}"
            if self.redis_client.exists(cache_key):
                self.stats["duplicates_filtered"] += 1
                return False
            
            # Store in PostgreSQL
            conn = self.pg_pool.getconn()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO social_media_posts 
                        (platform, post_id, author, content, timestamp, likes, shares, 
                         comments, hashtags, mentions, url, hash_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (hash_id) DO NOTHING
                    """, (
                        post.platform,
                        post.post_id,
                        post.author,
                        post.content,
                        post.timestamp,
                        post.likes,
                        post.shares,
                        post.comments,
                        json.dumps(post.hashtags),
                        json.dumps(post.mentions),
                        post.url,
                        post.hash_id
                    ))
                    
                    if cursor.rowcount > 0:
                        conn.commit()
                        
                        # Cache in Redis
                        self.redis_client.setex(cache_key, 86400, "1")  # 24 hours
                        
                        # Add to processed set
                        self.processed_posts.add(post.hash_id)
                        
                        self.stats["posts_stored"] += 1
                        return True
                    else:
                        self.stats["duplicates_filtered"] += 1
                        return False
                        
            finally:
                self.pg_pool.putconn(conn)
                
        except Exception as e:
            self.logger.error(f"Error storing social media post: {e}")
            self.stats["errors"] += 1
            return False
    
    async def _load_processed_posts(self):
        """Load recently processed post hashes from database"""
        try:
            conn = self.pg_pool.getconn()
            try:
                with conn.cursor() as cursor:
                    # Load posts from last 24 hours
                    cursor.execute("""
                        SELECT hash_id FROM social_media_posts 
                        WHERE timestamp > %s
                    """, (datetime.now(timezone.utc) - timedelta(hours=24),))
                    
                    for row in cursor.fetchall():
                        self.processed_posts.add(row[0])
                    
                    self.logger.info(f"Loaded {len(self.processed_posts)} processed posts")
                    
            finally:
                self.pg_pool.putconn(conn)
                
        except Exception as e:
            self.logger.error(f"Error loading processed posts: {e}")
    
    async def _monitor_performance(self):
        """Monitor and log performance metrics"""
        while self.running:
            await asyncio.sleep(600)  # Report every 10 minutes
            
            if self.stats["start_time"]:
                runtime = time.time() - self.stats["start_time"]
                posts_per_hour = (self.stats["posts_collected"] / runtime) * 3600
                
                self.logger.info(
                    f"Social Media Performance: "
                    f"{posts_per_hour:.1f} posts/hour, "
                    f"Stored: {self.stats['posts_stored']}, "
                    f"Duplicates: {self.stats['duplicates_filtered']}, "
                    f"Platform stats: {dict(self.stats['platform_stats'])}"
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get social media integration statistics"""
        with self.lock:
            stats = self.stats.copy()
            
            if stats["start_time"]:
                runtime = time.time() - stats["start_time"]
                stats["runtime_hours"] = runtime / 3600
                stats["posts_per_hour"] = (stats["posts_collected"] / runtime) * 3600 if runtime > 0 else 0
            
            return stats
    
    def _close_connections(self):
        """Close database connections"""
        try:
            if hasattr(self, 'redis_client'):
                self.redis_client.close()
            
            if hasattr(self, 'pg_pool'):
                self.pg_pool.closeall()
            
            self.logger.info("Connections closed")
            
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")


# Example usage
async def main():
    """Main function for testing social media integrator"""
    config = SocialMediaConfig()
    integrator = SocialMediaIntegrator(config)
    
    # Start integrator
    integrator_task = asyncio.create_task(integrator.start())
    
    # Run for a short time
    await asyncio.sleep(60)
    integrator.stop()
    
    # Print statistics
    stats = integrator.get_stats()
    print(f"Social media statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
