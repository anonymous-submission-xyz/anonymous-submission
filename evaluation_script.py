#!/usr/bin/env python3
"""
Research Gap Analysis - Dataset Evaluation Script
Evaluates model performance on pre-collected dataset
Saves raw model responses for later parsing
"""

import os
import json
import logging
import argparse
import gc
import time
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import torch
from dotenv import load_dotenv

# Import from main module
import sys
sys.path.append(str(Path(__file__).parent.parent))
from research_gap_module import (
    Article, Config, logger, analyze_with_llm
)

# Load environment
load_dotenv()

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")
warnings.filterwarnings("ignore", message="`flash-attention` package not found")


class JudgeEvaluator:
    """LLM-as-a-Judge for evaluating ResearchGapBot responses"""
    
    JUDGE_PROMPT = """You are an automatic expert judge for evaluating ResearchGapBot responses.

You are given a pair:
Question: {question}
Answer: {answer}

Evaluate the answer according to the following criteria (scale 1–10):
- Relevance — how well the answer addresses the question
- Coverage — how comprehensively the topic is covered
- Factuality — factual correctness, absence of hallucinations
- Clarity — clarity and logical structure of the presentation
- Novelty — usefulness for identifying research gaps / new ideas

Then calculate:
- TotalScore = average of the five criteria
- Percent = (TotalScore − 1) / 9 × 100
- Acceptable = true if TotalScore ≥ 7 and Factuality ≥ 6, otherwise false

Return strictly one JSON object in the following format — without any other text:
{{
"Relevance": int,
"Coverage": int,
"Factuality": int,
"Clarity": int,
"Novelty": int,
"TotalScore": float,
"Percent": float,
"Acceptable": bool,
"Confidence": float,
"Evaluation": "brief explanation (up to 120 words)"
}}

Now evaluate:
Question: {question}
Answer: {answer}"""
    
    def __init__(self):
        self.config = Config()
        self.output_dir = Path(os.getenv('OUTPUT_DIR', 'results'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Use pipeline for judge model (same as main module)
        self.judge_pipeline = None
        self.is_working = False
        self._load_judge_model()
    
    def _load_judge_model(self):
        """Load judge model using pipeline (same as main module)"""
        try:
            from transformers import pipeline
            
            judge_model = os.getenv('JUDGE_MODEL_NAME', 'microsoft/Phi-3-mini-4k-instruct')
            logger.info(f"Loading judge model: {judge_model}")
            
            # Pipeline без параметров генерации
            self.judge_pipeline = pipeline(
                "text-generation",
                model=judge_model,
                device_map="auto",
            )
            
            self.is_working = True
            logger.info("Judge model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load judge model: {e}")
            logger.error("Judge model is not working - will return None for all evaluations")
            self.is_working = False
            self.judge_pipeline = None

    def evaluate_single(self, question: str, answer: str) -> Optional[Dict]:
        """
        Просто генерируем ответ, JSON не парсим
        """
        try:
            prompt = self.JUDGE_PROMPT.format(question=question, answer=answer)

            result = self.judge_pipeline(
                prompt,
                max_new_tokens=1000,
                temperature=0.01,
                do_sample=False,
                pad_token_id=self.judge_pipeline.tokenizer.eos_token_id
            )

            response = result[0]['generated_text'].replace(prompt, '').strip()

            return {
                "raw_response": response,
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None

    def cleanup(self):
        """Clean up judge model"""
        if self.judge_pipeline:
            del self.judge_pipeline
            self.judge_pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()


def load_dataset(data_file: str, max_articles_per_query: int = 10) -> Dict[str, List[Article]]:
    """
    Load articles from JSON and group by query
    """
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        items = data if isinstance(data, list) else data.get("data", [])
        
        logger.info(f"Loaded {len(items)} items from {data_file}")
        
        articles_by_query = defaultdict(list)
        
        for item in items:
            try:
                # Extract query from meta
                meta = item.get("meta", {})
                query = meta.get("query") or item.get("query") or "unknown"
                
                # Create article
                article = Article(
                    title=item.get("title"),
                    abstract=item.get("abstract", ""),
                    authors=item.get("authors", []),
                    source=item.get("source", "unknown"),
                    meta=meta,
                    text=item.get("text"),
                )
                
                if len(articles_by_query[query]) < max_articles_per_query:
                    articles_by_query[query].append(article)
                
            except Exception as e:
                logger.warning(f"Failed to process item: {e}")
                continue
        
        # Log stats
        for query, articles in articles_by_query.items():
            logger.info(f"Query '{query}': {len(articles)} articles")
        
        return dict(articles_by_query)
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {}


def evaluate_dataset(data_file: str, max_articles: int = 10) -> List[Dict]:
    """
    Evaluate entire dataset - loads models sequentially to avoid OOM
    """
    config = Config()
    
    # Load dataset
    articles_by_query = load_dataset(data_file, max_articles)
    
    if not articles_by_query:
        logger.error("No articles loaded")
        return [], []
    
    all_results = []
    all_logs = []
    
    # Process each query with clean memory
    for query, articles in articles_by_query.items():
        logger.info(f"\nProcessing query: {query} ({len(articles)} articles)")
        
        # Clear memory before each query
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Step 1: Generate analysis (loads main model)
        logger.info("Loading main model for analysis...")
        try:
            analysis_result = analyze_with_llm(
                query=query,
                articles=articles,
                model_name=config.model_name
            )
        except Exception as e:
            logger.error(f"Analysis failed for query '{query}': {e}")
            continue
        
        # Clean up main model aggressively
        logger.info("Cleaning up main model...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Step 2: Initialize judge for this query (fresh)
        logger.info("Loading judge model...")
        judge = JudgeEvaluator()
        
        if not judge.is_working:
            logger.error("Judge model failed to load - skipping evaluation")
            judge.cleanup()
            continue
        
        # Prepare for evaluation
        question = f"What are the research gaps in the field of '{query}'?"
        answer = analysis_result.get("analysis", "")
        
        # Evaluate
        evaluation = judge.evaluate_single(question, answer)
        
        # Clean up judge
        judge.cleanup()
        
        # Only add to results if evaluation succeeded
        if evaluation is not None:
            result = {
                "query": query,
                "articles_count": len(articles),
                "model": config.model_name,
                "timestamp": datetime.now().isoformat(),
                "evaluation": evaluation
            }
            
            all_results.append(result)
            
            # Collect full log
            log_entry = {
                "query": query,
                "articles": [a.to_dict() for a in articles],
                "analysis": answer,
                "evaluation": evaluation
            }
            all_logs.append(log_entry)
            
            logger.info(f"Got raw response for query: {query}")
        else:
            logger.warning(f"Skipping query '{query}' - evaluation failed")
        
        # Final cleanup before next query
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    return all_results, all_logs


def export_results(results: List[Dict], logs: List[Dict], output_dir: str = "results") -> Path:
    """
    Export results - only JSON with raw responses, no CSV
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not results:
        logger.warning("No results to export")
        return None
    
    model_name = results[0]['model'].replace('/', '_')
    
    # JSON with full logs (all raw data)
    log_file = out_dir / f"evaluation_raw_{model_name}_{timestamp}.json"
    
    full_output = {
        "metadata": {
            "timestamp": timestamp,
            "model": model_name,
            "total_queries": len(results),
            "judge_model": os.getenv('JUDGE_MODEL_NAME', 'microsoft/Phi-3-mini-4k-instruct'),
        },
        "results": results,
        "logs": logs
    }
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Raw evaluation data saved to {log_file}")
    
    return log_file


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Evaluate Research Gap Analyzer on dataset")
    parser.add_argument("--data-file", required=True, help="JSON file with articles")
    parser.add_argument("--max-articles", type=int, default=10, help="Max articles per query")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Log GPU info
    config = Config()
    logger.info(f"Starting evaluation for model: {config.model_name}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available, running on CPU (will be slow)")
    
    start_time = time.time()
    
    # Run evaluation
    results, logs = evaluate_dataset(args.data_file, args.max_articles)
    
    if not results:
        logger.error("No results generated - check if judge model is working")
        return 1
    
    # Export only JSON
    log_file = export_results(results, logs, args.output_dir)
    
    elapsed = time.time() - start_time
    
    # Final summary
    logger.info("=" * 50)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Queries evaluated: {len(results)}")
    logger.info(f"Total time: {elapsed:.2f} seconds")
    logger.info(f"Raw data saved to: {log_file}")
    logger.info("=" * 50)
    
    return 0


if __name__ == "__main__":
    exit(main())
