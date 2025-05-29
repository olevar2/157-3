#!/usr/bin/env python3
"""
News Scraper for Market Sentiment Analysis
High-performance news feed scraping and processing for forex trading platform
Optimized for real-time financial news analysis

Author: Platform3 Development Team
Version: 1.0.0
"""

import asyncio
import aiohttp
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import re
from urllib.parse import urljoin, urlparse
import hashlib

# Third-party imports
import feedparser
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import redis
import psycopg2
from psycopg2.pool import ThreadedConnectionPool

@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    content: str
    url: str
    source: str
    published_date: datetime
    author: Optional[str] = None
    tags: List[str] = None
    hash_id: str = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        
        # Generate unique hash for deduplication
        if self.hash_id is None:
            content_hash = hashlib.md5(
                f"{self.title}{self.url}".encode('utf-8')
            ).hexdigest()
            self.hash_id = content_hash
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "published_date": self.published_date.isoformat(),
            "author": self.author,
            "tags": self.tags,
            "hash_id": self.hash_id
        }

@dataclass
class ScraperConfig:
    """Configuration for news scraper"""
    # RSS feeds to monitor
    rss_feeds: List[Dict[str, str]] = None
    
    # Web scraping settings
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    delay_between_requests: float = 1.0
    
    # Content filtering
    min_content_length: int = 100
    max_content_length: int = 10000
    
    # Database settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "forex_trading"
    postgres_user: str = "forex_admin"
    postgres_password: str = "ForexSecure2025!"
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = "RedisSecure2025!"
    
    # Update frequency
    update_interval: int = 300  # 5 minutes
    
    def __post_init__(self):
        if self.rss_feeds is None:
            self.rss_feeds = [
                {
                    "name": "Reuters Business",
                    "url": "https://feeds.reuters.com/reuters/businessNews",
                    "priority": "high"
                },
                {
                    "name": "Bloomberg Markets",
                    "url": "https://feeds.bloomberg.com/markets/news.rss",
                    "priority": "high"
                },
                {
                    "name": "ForexFactory",
                    "url": "https://www.forexfactory.com/rss.php",
                    "priority": "medium"
                },
                {
                    "name": "FXStreet",
                    "url": "https://www.fxstreet.com/rss/news",
                    "priority": "medium"
                },
                {
                    "name": "MarketWatch",
                    "url": "https://feeds.marketwatch.com/marketwatch/topstories/",
                    "priority": "medium"
                }
            ]

class NewsScraper:
    """
    High-performance news scraper for financial market data
    Handles RSS feeds and web scraping with deduplication
    """
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.running = False
        
        # Initialize HTTP session with retry strategy
        self.session = self._create_session()
        
        # Initialize database connections
        self._init_connections()
        
        # Article cache for deduplication
        self.processed_articles = set()
        
        # Statistics
        self.stats = {
            "articles_scraped": 0,
            "articles_stored": 0,
            "duplicates_filtered": 0,
            "errors": 0,
            "start_time": None,
            "source_stats": {}
        }
        
        # Financial keywords for relevance filtering
        self.financial_keywords = [
            'forex', 'currency', 'exchange rate', 'central bank', 'fed', 'ecb', 'boe', 'boj',
            'interest rate', 'monetary policy', 'inflation', 'gdp', 'employment', 'unemployment',
            'trade', 'tariff', 'brexit', 'election', 'crisis', 'recession', 'bull', 'bear',
            'dollar', 'euro', 'pound', 'yen', 'franc', 'aussie', 'loonie', 'kiwi'
        ]
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger("NewsScraper")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy"""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=self.config.retry_attempts,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers to appear as a regular browser
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        return session
    
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
        """Start the news scraping engine"""
        self.logger.info("Starting news scraper...")
        self.running = True
        self.stats["start_time"] = time.time()
        
        # Load processed articles from cache
        await self._load_processed_articles()
        
        # Start scraping tasks
        tasks = [
            asyncio.create_task(self._scrape_rss_feeds()),
            asyncio.create_task(self._monitor_performance())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Error in news scraping: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the news scraper"""
        self.logger.info("Stopping news scraper...")
        self.running = False
        self._close_connections()
    
    async def _scrape_rss_feeds(self):
        """Scrape all configured RSS feeds"""
        while self.running:
            try:
                # Create semaphore to limit concurrent requests
                semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
                
                # Process all feeds concurrently
                tasks = []
                for feed_config in self.config.rss_feeds:
                    task = asyncio.create_task(
                        self._process_rss_feed(feed_config, semaphore)
                    )
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Wait before next scraping cycle
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in RSS feed scraping: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _process_rss_feed(self, feed_config: Dict[str, str], semaphore: asyncio.Semaphore):
        """Process a single RSS feed"""
        async with semaphore:
            try:
                feed_url = feed_config["url"]
                feed_name = feed_config["name"]
                
                self.logger.debug(f"Processing RSS feed: {feed_name}")
                
                # Parse RSS feed
                loop = asyncio.get_event_loop()
                feed = await loop.run_in_executor(None, feedparser.parse, feed_url)
                
                articles_processed = 0
                
                for entry in feed.entries[:20]:  # Process last 20 entries
                    try:
                        article = await self._parse_rss_entry(entry, feed_name)
                        
                        if article and self._is_relevant_article(article):
                            if await self._store_article(article):
                                articles_processed += 1
                        
                        # Rate limiting
                        await asyncio.sleep(self.config.delay_between_requests)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing RSS entry: {e}")
                        self.stats["errors"] += 1
                
                # Update source statistics
                if feed_name not in self.stats["source_stats"]:
                    self.stats["source_stats"][feed_name] = 0
                self.stats["source_stats"][feed_name] += articles_processed
                
                self.logger.debug(f"Processed {articles_processed} articles from {feed_name}")
                
            except Exception as e:
                self.logger.error(f"Error processing RSS feed {feed_config['name']}: {e}")
                self.stats["errors"] += 1
    
    async def _parse_rss_entry(self, entry, source: str) -> Optional[NewsArticle]:
        """Parse RSS entry into NewsArticle"""
        try:
            # Extract basic information
            title = getattr(entry, 'title', '')
            url = getattr(entry, 'link', '')
            
            if not title or not url:
                return None
            
            # Parse published date
            published_date = datetime.now(timezone.utc)
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                published_date = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
            
            # Extract content
            content = ""
            if hasattr(entry, 'summary'):
                content = BeautifulSoup(entry.summary, 'html.parser').get_text()
            elif hasattr(entry, 'description'):
                content = BeautifulSoup(entry.description, 'html.parser').get_text()
            
            # Try to get full article content if summary is short
            if len(content) < self.config.min_content_length:
                full_content = await self._scrape_full_article(url)
                if full_content:
                    content = full_content
            
            # Extract author
            author = getattr(entry, 'author', None)
            
            # Extract tags
            tags = []
            if hasattr(entry, 'tags'):
                tags = [tag.term for tag in entry.tags if hasattr(tag, 'term')]
            
            article = NewsArticle(
                title=title.strip(),
                content=content.strip(),
                url=url,
                source=source,
                published_date=published_date,
                author=author,
                tags=tags
            )
            
            return article
            
        except Exception as e:
            self.logger.error(f"Error parsing RSS entry: {e}")
            return None
    
    async def _scrape_full_article(self, url: str) -> Optional[str]:
        """Scrape full article content from URL"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.session.get(url, timeout=self.config.request_timeout)
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Try common article selectors
                article_selectors = [
                    'article', '.article-content', '.story-body', 
                    '.entry-content', '.post-content', '.content'
                ]
                
                for selector in article_selectors:
                    article_element = soup.select_one(selector)
                    if article_element:
                        text = article_element.get_text()
                        if len(text) > self.config.min_content_length:
                            return text.strip()
                
                # Fallback: get all paragraph text
                paragraphs = soup.find_all('p')
                text = ' '.join([p.get_text() for p in paragraphs])
                
                if len(text) > self.config.min_content_length:
                    return text.strip()
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error scraping full article {url}: {e}")
            return None
    
    def _is_relevant_article(self, article: NewsArticle) -> bool:
        """Check if article is relevant to forex trading"""
        text = f"{article.title} {article.content}".lower()
        
        # Check for financial keywords
        keyword_count = sum(1 for keyword in self.financial_keywords if keyword in text)
        
        # Article is relevant if it contains at least 2 financial keywords
        # or if it's from a high-priority financial source
        return keyword_count >= 2 or any(
            source in article.source.lower() 
            for source in ['reuters', 'bloomberg', 'forexfactory', 'fxstreet']
        )
    
    async def _store_article(self, article: NewsArticle) -> bool:
        """Store article in database if not duplicate"""
        try:
            # Check for duplicates
            if article.hash_id in self.processed_articles:
                self.stats["duplicates_filtered"] += 1
                return False
            
            # Check Redis cache for recent duplicates
            cache_key = f"article:{article.hash_id}"
            if self.redis_client.exists(cache_key):
                self.stats["duplicates_filtered"] += 1
                return False
            
            # Store in PostgreSQL
            conn = self.pg_pool.getconn()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO news_articles 
                        (title, content, url, source, published_date, author, tags, hash_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (hash_id) DO NOTHING
                    """, (
                        article.title,
                        article.content,
                        article.url,
                        article.source,
                        article.published_date,
                        article.author,
                        json.dumps(article.tags),
                        article.hash_id
                    ))
                    
                    if cursor.rowcount > 0:
                        conn.commit()
                        
                        # Cache in Redis
                        self.redis_client.setex(cache_key, 86400, "1")  # 24 hours
                        
                        # Add to processed set
                        self.processed_articles.add(article.hash_id)
                        
                        self.stats["articles_stored"] += 1
                        return True
                    else:
                        self.stats["duplicates_filtered"] += 1
                        return False
                        
            finally:
                self.pg_pool.putconn(conn)
                
        except Exception as e:
            self.logger.error(f"Error storing article: {e}")
            self.stats["errors"] += 1
            return False
    
    async def _load_processed_articles(self):
        """Load recently processed article hashes from database"""
        try:
            conn = self.pg_pool.getconn()
            try:
                with conn.cursor() as cursor:
                    # Load articles from last 24 hours
                    cursor.execute("""
                        SELECT hash_id FROM news_articles 
                        WHERE published_date > %s
                    """, (datetime.now(timezone.utc) - timedelta(hours=24),))
                    
                    for row in cursor.fetchall():
                        self.processed_articles.add(row[0])
                    
                    self.logger.info(f"Loaded {len(self.processed_articles)} processed articles")
                    
            finally:
                self.pg_pool.putconn(conn)
                
        except Exception as e:
            self.logger.error(f"Error loading processed articles: {e}")
    
    async def _monitor_performance(self):
        """Monitor and log performance metrics"""
        while self.running:
            await asyncio.sleep(300)  # Report every 5 minutes
            
            if self.stats["start_time"]:
                runtime = time.time() - self.stats["start_time"]
                articles_per_hour = (self.stats["articles_scraped"] / runtime) * 3600
                
                self.logger.info(
                    f"News Scraper Performance: "
                    f"{articles_per_hour:.1f} articles/hour, "
                    f"Stored: {self.stats['articles_stored']}, "
                    f"Duplicates: {self.stats['duplicates_filtered']}, "
                    f"Errors: {self.stats['errors']}"
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scraper statistics"""
        stats = self.stats.copy()
        
        if stats["start_time"]:
            runtime = time.time() - stats["start_time"]
            stats["runtime_hours"] = runtime / 3600
            stats["articles_per_hour"] = (stats["articles_scraped"] / runtime) * 3600 if runtime > 0 else 0
        
        return stats
    
    def _close_connections(self):
        """Close database connections"""
        try:
            if hasattr(self, 'session'):
                self.session.close()
            
            if hasattr(self, 'redis_client'):
                self.redis_client.close()
            
            if hasattr(self, 'pg_pool'):
                self.pg_pool.closeall()
            
            self.logger.info("Connections closed")
            
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")


# Example usage
async def main():
    """Main function for testing news scraper"""
    config = ScraperConfig()
    scraper = NewsScraper(config)
    
    # Start scraper
    scraper_task = asyncio.create_task(scraper.start())
    
    # Run for a short time
    await asyncio.sleep(30)
    scraper.stop()
    
    # Print statistics
    stats = scraper.get_stats()
    print(f"Scraper statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
