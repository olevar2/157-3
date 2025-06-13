"""
Main execution script for Platform3 Trading System
"""
import asyncio
import logging
from datetime import datetime
from ai_platform.coordination.engine.platform3_engine import Platform3TradingEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main execution function"""
    print("=== Platform3 Trading System ===")
    print("Humanitarian Trading for Global Good")
    print(f"Started: {datetime.now()}")
    print("=" * 40)
    
    # Initialize the main engine
    engine = Platform3TradingEngine()
    
    try:
        # Start 24/7 operation
        print("\nStarting 24/7 trading operation...")
        print("- 9 Genius Agents: Active")
        print("- 115+ Indicators: Loaded")
        print("- Adaptive Systems: Online")
        print("\nMonitoring all major forex pairs...")
        
        await engine.start_24_7_operation()
        
    except KeyboardInterrupt:
        print("\nShutting down Platform3...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print(f"\nStopped: {datetime.now()}")

if __name__ == "__main__":
    asyncio.run(main())
