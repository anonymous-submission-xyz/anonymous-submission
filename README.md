## ResearchGapBot: Discourse-Aware Research Gap Detection

ResearchGapBot is an open-source framework for automatic research gap detection in academic literature. 
The system is designed for local deployment in scientific environments.

The tool combines discourse-aware analysis focusing on titles and abstracts with Retrieval-Augmented Generation (RAG)
using local LLMs (1-7B parameters). It integrates real-time paper retrieval via the arXiv API and provides a complete 
benchmarking suite for evaluating model performance.

Benchmark results show that Llama-3.2-1B achieves the best performance-quality trade-off, demonstrating that large 
proprietary models are not necessary for effective academic text analysis.

#### Usage

Basic analysis:

```python
python research_gap_module.py "quantum machine learning"
```

Specifications:
- `--max-results`: top-K results from arXiv API
- `--model`: model name from Hugging Face hub
- `--output-dir`: custom output directory

Examples:

```python
python research_gap_module.py "attention mechanisms" --max-results 15 --model "meta-llama/Llama-3.2-1B"
```

```python
python research_gap_module.py "climate change" --output-dir ./my_results
```

#### Python API

```python
from research_gap_module import search_articles, analyze_with_llm

# Search for papers
articles = search_articles("transformer architectures", max_results=10)

# Analyze with specific model
results = analyze_with_llm(
    query="transformer architectures",
    articles=articles,
    model_name="meta-llama/Llama-3.2-1B"
)

print(results["analysis"])
```

#### Running Benchmarks

```bash
python -m research_gap_module.evaluate --models \
    "meta-llama/Llama-3.2-1B" \
    "Qwen/Qwen2.5-1.5B-Instruct" \
    "microsoft/Phi-3-mini-4k-instruct" \
    "google/gemma-3-1b-it" \
    "codellama/CodeLlama-7b-Instruct-hf"
```

##### Requirements
- Python 3.11+
- PyTorch with CUDA support
- 8GB+ GPU VRAM (for 7B models)

#### Tested Models
- `meta-llama/Llama-3.2-1B` (recommended)
- `Qwen/Qwen2.5-1.5B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`
- `google/gemma-3-1b-it`
- `codellama/CodeLlama-7b-Instruct-hf`
- `microsoft/phi-2` (tested, known pad_token_id issue)

#### Memory Management
Sequential model loading with CUDA cache clearing to handle multiple models within 8GB VRAM constraints. 
Models are loaded, used, and unloaded per query to prevent out-of-memory errors.

### Benchmarking

Quality assessment uses an LLM-as-a-judge protocol across five dimensions. Results are aggregated over 14 diverse research queries covering natural sciences, social sciences, and humanities.

#### Test Configuration
- **Judge Model:** `google/gemma-3-1b-it`
- **Generation Parameters:** `max_new_tokens=500`, `temperature=0.7`
- **Topics:** Anthropology, Economics, Engineering, Ethnography, Folklore studies, Mycology, Narratology, Oceanology, Paleontology, Particle physics, Planetology, Political science, Sociology, Urban studies

#### Key Findings
- Model size does not correlate with output quality. The 1B Llama 3.2 outperforms larger models.
- Speed varies minimally among 1-1.5B class models.
- Code-specialized models underperform on general academic analysis tasks.

> This codebase accompanies a paper currently under review. The benchmark dataset was collected via arXiv API search. All evaluation data and analysis scripts are included. For questions or issues, please open a GitHub ticket.
