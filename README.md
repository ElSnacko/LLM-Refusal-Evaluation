<div align="center">

# 🛡️ LLM Refusal Evaluation

**A comprehensive benchmark suite for evaluating LLM refusal behavior on safety and sensitive topics**


[![arXiv](https://img.shields.io/badge/arXiv-2512.16602-b31b1b.svg)](https://arxiv.org/abs/2512.16602)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-llm--refusal--evaluation-yellow)](https://huggingface.co/datasets/MultiverseComputingCAI/llm-refusal-evaluation)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)     
<br>
[![Multiverse Computing](https://img.shields.io/badge/Multiverse_Computing-purple)](https://multiversecomputing.com)

</div>

---

## 📖 Overview

**LLM Refusal Evaluation** is an inference-time evaluation framework for measuring refusal behavior in Large Language Models. Unlike traditional pattern-based refusal detection, this library uses an **LLM-as-a-judge** approach to accurately identify sophisticated refusal patterns—including government-aligned narratives, topic deflection, information omission, and propaganda replacement.

The methodology is based on the paper [**"Refusal Steering: Fine-grained Control over LLM Refusal Behaviour for Sensitive Topics"**](https://arxiv.org/abs/2512.16602).

### ✨ Key Features

- **🎯 LLM-as-a-Judge Detection** — Captures nuanced refusals that pattern-matching misses
- **📊 Confidence Scoring** — Probability-weighted refusal scores for fine-grained analysis
- **🔬 Multi-benchmark Suite** — Safety, Chinese-sensitive, and sanity-check datasets
- **⚡ vLLM-powered** — Efficient batch inference with tensor parallelism
- **📈 Automatic Metrics** — Generates histograms, compliance/rejection percentages, and per-category statistics with bootstrap confidence intervals
- **🏷️ Category Preservation** — Auto-detects dataset categories (including multi-label boolean columns) and propagates them through the entire pipeline
- **⚖️ Balanced Sampling** — `--samples-per-category N` for manageable runs on large datasets
- **🔌 Dataset Adapters** — Built-in column mappings for BeaverTails, WildJailbreak, and SORRY-Bench; load any HuggingFace dataset via CLI
- **🔍 Audit Trail** — Every output entry includes `source_dataset`, `source_row_index`, `prompt_hash`, and `classification_method` for full traceability
- **✂️ Truncated Generation** — `--max-new-tokens` CLI override for fast pilot runs
- **📋 Compliance Quality** — Automatic quality scoring for compliant responses (lexical diversity, hedge phrase detection)
- **🔗 Merge Utility** — Combine results from multiple runs with prompt-hash deduplication

---

## 🧪 Evaluation Methodology

The evaluation pipeline works in three stages:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Generate    │ ──▶ │  2. Judge       │ ──▶ │  3. Aggregate   │
│     Answers     │     │     Responses   │     │     Scores      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
   K samples per           LLM-as-a-judge         Softmax-weighted
   prompt with             classifies each        refusal confidence
   log-probabilities       as refusal/not         scores per prompt
```

### Refusal Confidence Score

For each prompt, we sample `K` answers and compute a **refusal confidence score** `c(x) ∈ [-1, 1]`:

- **`c(x) > 0`** → Model tends to refuse
- **`c(x) < 0`** → Model tends to comply
- **`c(x) ≈ 0`** → Uncertain/mixed behavior

The score is weighted by answer probability using softmax over log-probabilities, emphasizing more likely completions.

---

## 📦 Installation

This project uses [**uv**](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/CompactifAI/LLM-Refusal-Evaluation.git
cd LLM-Refusal-Evaluation

# If uv is NOT available in your system
pip install uv
# Or
curl -LsSf https://astral.sh/uv/install.sh | sh


# Install dependencies
uv sync
```

---

## 🚀 Quick Start

### Evaluate a Model

Run evaluation with a YAML configuration file:

```bash
uv run python -m src.compute_refusal_score --config configs/Qwen3-4B-Instruct-2507.yaml
```

Or with conda/pip (set `PYTHONPATH` so `src` is importable):

```bash
cd LLM-Refusal-Evaluation
PYTHONPATH=. python src/compute_refusal_score.py --config configs/Qwen3-4B-Instruct-2507.yaml
```

### Quick Pilot Run

Run a fast pilot on BeaverTails with 20 samples per category and truncated generation:

```bash
PYTHONPATH=. python src/compute_refusal_score.py \
  --config configs/my-model.yaml \
  --custom-dataset PKU-Alignment/BeaverTails-Evaluation \
  --dataset-split test \
  --samples-per-category 20 \
  --max-new-tokens 512 \
  --seed 42
```

### Example Output Structure

```
results/Qwen3-4B-Instruct-2507/
├── jailbreakbench/
│   ├── answers.json                          # Generated model responses
│   ├── judge_scores.json                     # LLM judge classifications
│   ├── censor_scores.json                    # Aggregated refusal scores
│   ├── censor_scores_metrics.json            # Compliance/rejection percentages + per-category stats
│   └── censor_scores_answer_censor_score.jpg # Score distribution histogram
├── sorrybench/
│   └── ...
└── ...
```

Each entry in `censor_scores.json` now includes:
- `category` — harm category label(s) from the source dataset (when configured)
- `source_dataset`, `source_split`, `source_row_index` — full provenance
- `prompt_hash` — SHA256 hash for deduplication and traceability
- `classification_method` — `"judge"` (LLM-as-a-judge)
- `compliance_quality` — quality score for compliant responses (0-1)

---

## ⚙️ Configuration

Create a YAML config file to specify your evaluation:

```yaml
# Dataset splits to evaluate
dataset_splits:
  # Simple string form — uses built-in Iker/refusal-evaluation dataset
  - jailbreakbench
  - sorrybench

  # Dict form — any HuggingFace dataset with explicit column mappings
  - name: "beavertails"
    dataset_id: "PKU-Alignment/BeaverTails-Evaluation"
    split: "test"
    prompt_column: "prompt"
    category_column: "auto"    # auto-detect boolean category columns

  # Known datasets get automatic column mappings (adapters)
  - dataset_id: "allenai/wildjailbreak"
    split: "train"
    # adapter auto-applies: prompt_column="vanilla", category_column="risk_category"

# Model under evaluation
model:
  name_or_path: "Qwen/Qwen3.5-9B"
  max_model_len: 16384
  max_new_tokens: 8192
  thinking-string: </think>    # reasoning end token, i.e "</think>"
  num_return_sequences: 5  # Number of samples per prompt
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  batch_size: 512

# Judge model configuration
judge_model:
  name_or_path: "openai/gpt-oss-20b"
  max_model_len: 24576
  max_new_tokens: 8192
  num_return_sequences: 1
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  batch_size: 512

# Infrastructure settings
gpu_memory_utilization: 0.95
tensor_parallel_size: "auto"  # Use all available GPUs
continue_from_checkpoint: true

# Output directory
output_dir: "results/my-model-evaluation"
```

### Configuration Options

| Parameter | Description |
|-----------|-------------|
| `dataset_splits` | List of benchmark datasets (strings or dicts) |
| `dataset_splits[].dataset_id` | HuggingFace dataset identifier |
| `dataset_splits[].name` | Custom output directory name for this split |
| `dataset_splits[].prompt_column` | Column name for prompts (auto-detected for known datasets) |
| `dataset_splits[].category_column` | Column for categories. Use `"auto"` for boolean column auto-detection (e.g., BeaverTails) |
| `model.name_or_path` | HuggingFace model ID or local path |
| `model.thinking-string` | Token that separates reasoning from answer (e.g., `"</think>"`) |
| `model.num_return_sequences` | Number of answer samples per prompt (default: 5) |
| `judge_model.name_or_path` | Model used for refusal classification |
| `tensor_parallel_size` | Number of GPUs (`"auto"` = use all) |
| `continue_from_checkpoint` | Resume from previous run if files exist |

### CLI Options

These flags override or extend the YAML config:

```bash
python src/compute_refusal_score.py --config configs/my-model.yaml [OPTIONS]
```

| Flag | Description |
|------|-------------|
| `--custom-dataset HF_ID` | Override config's dataset_splits with a single HuggingFace dataset |
| `--prompt-column COL` | Prompt column for `--custom-dataset` (default: auto-detect or `"prompt"`) |
| `--category-column COL` | Category column for `--custom-dataset` (use `"auto"` for boolean auto-detection) |
| `--dataset-split SPLIT` | Dataset split for `--custom-dataset` (default: `"train"`) |
| `--samples-per-category N` | Sample N prompts per category for balanced runs |
| `--seed INT` | Random seed for balanced sampling (default: 42) |
| `--max-new-tokens INT` | Override max generation length (e.g., 50 for fast pilot runs) |
| `--model-type {instruct,base}` | Warns if truncated generation is used with a base model |

### Dataset Adapters

Built-in adapters auto-detect column mappings for known datasets. These apply only when `prompt_column` or `category_column` is not explicitly set:

| Dataset | prompt_column | category_column |
|---------|---------------|-----------------|
| `PKU-Alignment/BeaverTails*` | `prompt` | `auto` (boolean columns) |
| `allenai/wildjailbreak` | `vanilla` | `risk_category` |
| `sorry-bench/*` | `prompt` | `category` |

### Merge Utility

Combine results from multiple runs with prompt-hash deduplication:

```bash
python merge_results.py \
  --input-dirs results/run1 results/run2 \
  --output-dir results/merged

# Without deduplication
python merge_results.py \
  --input-dirs results/run1 results/run2 \
  --output-dir results/merged \
  --no-dedup
```

---

## 📊 Benchmark Datasets

All datasets are available at [🤗 MultiverseComputingCAI/llm-refusal-evaluation](https://huggingface.co/datasets/MultiverseComputingCAI/llm-refusal-evaluation).

### 🔴 Safety Benchmarks

Models **should refuse** these prompts.

| Dataset | Description | Source |
|---------|-------------|--------|
| `jailbreakbench` | Jailbreak robustness evaluation | [arXiv:2404.01318](https://arxiv.org/abs/2404.01318) |
| `sorrybench` | 440 unsafe prompts across 44 categories | [arXiv:2406.14598](https://arxiv.org/abs/2406.14598) |
| `xstest_unsafe` | 200 unsafe prompts | [arXiv:2308.01263](https://arxiv.org/abs/2308.01263) |
| `harmbench_sampled` | 256 sampled red-teaming prompts | [arXiv:2402.04249](https://arxiv.org/abs/2402.04249) |
| `adversarial_unsafe_prompts` | Harmful prompts with China-topic injections | [NVIDIA Dataset](https://huggingface.co/datasets/nvidia/Nemotron-Content-Safety-Reasoning-Dataset) |

### 🟡 Chinese Sensitive Topics

Models **may refuse** due to alignment, but uncensored models should engage.

| Dataset | Description | Source |
|---------|-------------|--------|
| `ccp_sensitive_sampled` | Topics likely censored by China-aligned models | [promptfoo/CCP-sensitive-prompts](https://huggingface.co/datasets/promptfoo/CCP-sensitive-prompts) |
| `deccp_censored` | Prompts triggering refusals in Qwen2-7B-Instruct | [augmxnt/deccp](https://huggingface.co/datasets/augmxnt/deccp) |

### 🟢 Sanity Check

Models **should NOT refuse** these prompts.

| Dataset | Description | Source |
|---------|-------------|--------|
| `xstest_safe` | 250 safe prompts across 10 types | [arXiv:2308.01263](https://arxiv.org/abs/2308.01263) |
| `general_prompts` | 100 random non-sensitive prompts | [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) |

---

## 📈 Understanding Results

### Metrics Output

After evaluation, each split generates a `_metrics.json` file:

```json
{
  "answer_censor_score": {
    "compliance_pct": 85.2,
    "neutral_pct": 5.1,
    "rejection_pct": 9.7
  },
  "per_category": {
    "violence": {
      "total": 50,
      "refusal": 45,
      "compliant": 3,
      "uncertain": 2,
      "mean_score": 0.7842,
      "bootstrap_ci_95": [0.6521, 0.9163],
      "recommendation": "sufficient"
    },
    "controversial_topics,politics": {
      "total": 18,
      "refusal": 0,
      "compliant": 18,
      "uncertain": 0,
      "mean_score": -0.9512,
      "bootstrap_ci_95": [-1.0, -0.8514],
      "recommendation": "borderline — recommend 12+ additional prompts"
    }
  }
}
```

The `per_category` section (present when categories are configured) includes per-category refusal/compliance counts, bootstrap 95% confidence intervals for the mean score, and a recommendation on whether the sample size is sufficient for stable steering vectors.

### Expected Behavior by Dataset Type

| Dataset Type | Expected Rejection % | Description |
|--------------|---------------------|-------------|
| Safety Benchmarks | **High (>80%)** | Model should refuse harmful requests |
| Chinese Sensitive | **Varies** | Depends on model alignment |
| Sanity Check | **Low (<10%)** | Model should not over-refuse benign requests |

---

## 🔍 Refusal Detection Categories

The LLM judge identifies 12 categories of refusal behavior:

1. **Direct/Explicit Refusal** — Clear rejection statements
2. **Deflection with Alternative** — Offers safer alternatives
3. **Misinformation/Narrative Manipulation** — Distorted facts
4. **Reframing/Sanitization** — Avoids harmful substance
5. **Feigned Ignorance** — Pretends not to know
6. **Extreme Vagueness** — Minimal, uninformative responses
7. **Topic Redirection** — Narrows scope dramatically
8. **Conditional/Hypothetical Framing** — Excessive hedging
9. **Moral Lecturing** — Extended ethical commentary
10. **Propaganda Replacement** — Official narrative enforcement
11. **Question Invalidation** — Dismisses premise
12. **Non-Sensical Response** — Corrupted or looping output

---

## 📚 Citation

If you use this library in your research, please cite:

```bibtex
@misc{garciaferrero2025Refusal,
      title={Refusal Steering: Fine-grained Control over LLM Refusal Behaviour for Sensitive Topics}, 
      author={Iker García-Ferrero and David Montero and Roman Orus},
      year={2025},
      eprint={2512.16602},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.16602}, 
}
```



