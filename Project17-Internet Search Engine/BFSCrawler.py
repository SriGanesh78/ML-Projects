#!/usr/bin/env python3.10
"""
BFS Web Crawler - Fixed version for Project 17
This script demonstrates web crawling with crawl4ai library.
"""

import asyncio
import sys
import os

# Add error handling for missing dependencies
try:
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
    from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
    print("✅ crawl4ai imported successfully")
except ImportError as e:
    print(f"❌ Error importing crawl4ai: {e}")
    print("💡 To install crawl4ai, run: pip install crawl4ai")
    print("💡 Or use the Jupyter notebook version instead")
    sys.exit(1)

async def main():
    """Main crawling function with enhanced error handling"""
    print("🚀 Starting BFS Web Crawler...")

    try:
        # Configure a 2-level deep crawl
        config = CrawlerRunConfig(
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_depth=2,
                include_external=False
            ),
            scraping_strategy=LXMLWebScrapingStrategy(),
            verbose=True
        )

        print("📊 Configuration:")
        print(f"   - Max Depth: 2")
        print(f"   - Include External: False")
        print(f"   - Scraping Strategy: LXML")
        print("-" * 50)

        async with AsyncWebCrawler() as crawler:
            print("🌐 Starting crawl of https://www.wikipedia.org...")
            results = await crawler.arun("https://www.wikipedia.org", config=config)

            if results:
                print(f"✅ Successfully crawled {len(results)} pages")
                print("\n📋 CRAWLING RESULTS:")
                print("=" * 60)

                # Show first 3 results with more details
                for i, result in enumerate(results[:3], 1):
                    print(f"\n{i}. URL: {result.url}")
                    print(f"   Depth: {result.metadata.get('depth', 0)}")
                    print(f"   Title: {result.metadata.get('title', 'No title')[:100]}...")
                    print(f"   Content Length: {len(result.markdown or ''):,} characters")
                    print("-" * 40)

                if len(results) > 3:
                    print(f"\n... and {len(results) - 3} more pages")

            else:
                print("❌ No results returned from crawler")

    except Exception as e:
        print(f"❌ Error during crawling: {e}")
        print("💡 This might be due to network issues or website restrictions")
        return False

    return True

def run_crawler():
    """Wrapper function to handle asyncio properly"""
    try:
        # Check if we're in a Jupyter environment
        if 'ipykernel' in sys.modules:
            print("📓 Detected Jupyter environment")
            print("💡 Use 'await main()' instead of running this script directly")
            return

        # Run in regular Python environment
        result = asyncio.run(main())
        if result:
            print("\n🎉 Crawling completed successfully!")
        else:
            print("\n❌ Crawling failed")

    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print("❌ Error: Cannot run asyncio.run() in this environment")
            print("💡 This script is designed for regular Python execution")
            print("💡 For Jupyter notebooks, use the notebook version instead")
        else:
            print(f"❌ Runtime error: {e}")

if __name__ == "__main__":
    run_crawler()