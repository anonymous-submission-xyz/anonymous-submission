"""
Research Gap Analyzer
Analyzes academic literature to identify research gaps using LLMs
"""

import os
import json
import time
import logging
import argparse
import requests
import xml.etree.ElementTree as ET
import gc
import torch  # Добавил в глобальные импорты
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_gap.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Article:
    """Data class representing a research paper"""
    title: Optional[str]
    abstract: Optional[str]
    authors: List[str]
    source: str
    meta: Dict
    text: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'source': self.source,
            'meta': self.meta,
            'has_full_text': bool(self.text and len(self.text) > 100)
        }


class Config:
    """Application configuration from environment"""
    def __init__(self):
        self.core_api_key = os.getenv('CORE_API_KEY')
        self.model_name = os.getenv('MODEL_NAME', 'microsoft/Phi-3-mini-4k-instruct')
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '30'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.max_workers = int(os.getenv('MAX_WORKERS', '4'))


class ArxivSearcher:
    """arXiv API searcher"""
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search(self, query: str, max_results: int = 50) -> List[Article]:
        """Search arXiv for papers matching query"""
        base_url = "http://export.arxiv.org/api/query"
        articles = []
        start = 0
        batch_size = 100
        
        while start < max_results:
            try:
                params = {
                    "search_query": f"all:{query}",
                    "start": start,
                    "max_results": min(batch_size, max_results - start)
                }
                
                response = self.session.get(
                    base_url, 
                    params=params, 
                    timeout=self.config.request_timeout
                )
                
                if response.status_code != 200:
                    logger.error(f"arXiv HTTP {response.status_code}")
                    break
                
                root = ET.fromstring(response.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall("atom:entry", ns)
                
                if not entries:
                    break
                
                for entry in entries:
                    title = entry.find("atom:title", ns)
                    summary = entry.find("atom:summary", ns)
                    id_elem = entry.find("atom:id", ns)
                    
                    authors = []
                    for author in entry.findall("atom:author", ns):
                        name = author.find("atom:name", ns)
                        if name is not None:
                            authors.append(name.text)
                    
                    articles.append(Article(
                        title=title.text.strip() if title is not None else None,
                        abstract=summary.text.strip() if summary is not None else "",
                        authors=authors,
                        source="arxiv",
                        meta={"url": id_elem.text if id_elem is not None else None},
                        text=None
                    ))
                
                start += len(entries)
                time.sleep(1)  # Be nice to the API
                    
            except Exception as e:
                logger.error(f"arXiv search error: {e}")
                break
        
        logger.info(f"arXiv found {len(articles)} articles")
        return articles


def search_articles(query: str, max_results: int = 50) -> List[Article]:
    """Search for articles across all configured sources"""
    config = Config()
    
    searchers = [ArxivSearcher(config)]
    all_articles = []
    
    logger.info(f"Searching: '{query}' (max: {max_results})")
    
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(s.search, query, max_results): s.__class__.__name__
            for s in searchers
        }
        
        for future in as_completed(futures):
            try:
                articles = future.result()
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Search error: {e}")
    
    # Remove duplicates by title
    seen = set()
    unique = []
    for article in all_articles:
        if article.title and article.title.lower() not in seen:
            seen.add(article.title.lower())
            unique.append(article)
    
    logger.info(f"Found {len(unique)} unique articles")
    return unique[:max_results]

def analyze_with_llm(query: str, articles: List[Article], model_name: str) -> Dict:
    """
    Analyze research gaps using HF pipeline
    """
    start_time = time.time()
    
    # Prepare articles for analysis
    articles_data = [a.to_dict() for a in articles[:20]]
    
    # Build prompt
    articles_text = ""
    for i, article in enumerate(articles_data, 1):
        articles_text += f"\nArticle {i}: {article['title']}\n"
        articles_text += f"Abstract: {article['abstract'][:500]}\n"
    
    prompt = f"""Analyze these academic articles and identify research gaps:

{articles_text}

Provide a brief analysis of:
1. Main research themes
2. Key gaps or limitations
3. Suggestions for future research

Analysis:"""
    
    pipe = None
    try:
        logger.info(f"Loading model: {model_name}")
        # Создаем pipeline без параметров генерации
        pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
        )
        
        logger.info("Generating analysis...")
        # Параметры генерации передаем при вызове
        result = pipe(
            prompt,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        analysis = result[0]['generated_text'].replace(prompt, '').strip()
        
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        analysis = f"Analysis failed: {str(e)}"
    finally:
        # Clean up
        if pipe is not None:
            del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    # Prepare result
    result = {
        "metadata": {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "articles_analyzed": len(articles_data),
            "total_articles_found": len(articles),
            "processing_time_seconds": round(time.time() - start_time, 2)
        },
        "analysis": analysis,
        "articles": articles_data
    }
    
    return result

def save_results(results: Dict, output_dir: str = "results") -> Path:
    """Save analysis results to JSON file"""
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_slug = results["metadata"]["query"].replace(" ", "_")[:30]
    filename = f"analysis_{query_slug}_{timestamp}.json"
    filepath = out_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {filepath}")
    return filepath


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Research Gap Analyzer")
    parser.add_argument("query", nargs="?", help="Research query to analyze")
    parser.add_argument("--max-results", type=int, default=20, help="Maximum articles to fetch")
    parser.add_argument("--model", help="Override model from config")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # If no query provided, use interactive mode
    if not args.query:
        args.query = input("Enter research query: ").strip()
        if not args.query:
            logger.error("No query provided")
            return
    
    # Setup config
    config = Config()
    model_name = args.model or config.model_name
    
    logger.info(f"Starting analysis for: '{args.query}'")
    logger.info(f"Using model: {model_name}")
    
    # Search for articles
    articles = search_articles(args.query, args.max_results)
    
    if not articles:
        logger.error("No articles found")
        return
    
    # Analyze with LLM
    results = analyze_with_llm(args.query, articles, model_name)
    
    # Save results
    save_results(results, args.output_dir)
    
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
