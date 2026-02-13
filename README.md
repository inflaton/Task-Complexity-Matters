# Task Complexity Matters: An Empirical Study of Reasoning in LLMs for Sentiment Analysis

**Code, datasets & experiments for the paper:** *Task Complexity Matters: An Empirical Study of Reasoning in LLMs for Sentiment Analysis*

This work extends [*Explainable Sentiment Analysis with DeepSeek-R1*](https://ieeexplore.ieee.org/document/11181065) ([code](https://github.com/inflaton/Explainable-Sentiment-Analysis-with-DeepSeek-R1)) with the first large-scale, cross-architectural evaluation of reasoning in LLMs for sentiment analysis.

We systematically evaluate **504 configurations** across seven model families, three sentiment datasets of varying complexity, and seven few-shot settings to answer: *Do reasoning capabilities justify their computational overhead for sentiment tasks of varying complexity?*

---

## Key Findings

- **Reasoning is task-complexity dependent**: binary classification degrades up to −19.9 F1 pp, while 27-class emotion recognition gains up to +16.0 pp.
- **Distilled reasoning variants underperform** base models by 3–18 pp on simpler tasks, though few-shot prompting enables partial recovery.
- **Few-shot learning** yields more robust and consistent improvements than reasoning modes.
- **Pareto frontier analysis** shows base models dominate efficiency–performance trade-offs; reasoning is justified only for complex emotion recognition despite 2.1×–54× computational overhead.
- **Qualitative error analysis** reveals reasoning degrades simpler tasks through systematic over-deliberation.

---

## Models Evaluated

| Family | Models | Reasoning Type |
|--------|--------|---------------|
| **DeepSeek-R1** | Full (671B), distilled 8B/14B/32B/70B | Built-in (RL-trained) |
| **DeepSeek-V3** | Base model for R1-Full | Base (no reasoning) |
| **LLaMA** | 3.1-8B, 3.3-70B | Base (distillation targets) |
| **Qwen2.5** | 14B, 32B | Base (distillation targets) |
| **Qwen3** | 4B, 8B, 14B, 32B | Adaptive thinking (T/N modes) |
| **Granite3.3** | 2B, 8B | Conditional reasoning (T/N modes) |
| **Magistral** | 24B | RL-based reasoning (T/N modes) |

---

## Datasets

Three benchmarks spanning increasing complexity levels:

| Dataset | Classes | Complexity | Description |
|---------|---------|------------|-------------|
| **IMDB Movie Reviews** | 2 | Simple | Binary sentiment (positive/negative) |
| **Amazon Reviews** | 5 | Moderate | Five-class (strongly negative → strongly positive) |
| **GoEmotions** | 27 | High | 27 emotion categories (single-label subset) |

---

## Repository Structure

```
llm_toolkit/              # Core evaluation code
├── eval_openai.py             # Evaluate OpenAI-compatible APIs
├── eval_gemini.py             # Evaluate Google Gemini API
├── data_utils.py              # Prompt templates, parsing, metrics, plotting, dataset prep
└── llm_utils.py               # Tokenization, batching, API helpers

dataset/                  # Ground-truth data + documentation
├── amazon_reviews*.csv        # Amazon Reviews (5-class)
├── imdb_reviews*.csv          # IMDB Reviews (binary)
├── GoEmotions*.csv            # GoEmotions (27-class)
└── GoEmotions/                # Raw GoEmotions data + label mappings

notebooks/                # Analysis & evaluation notebooks
├── 00_EDA.ipynb                    # Exploratory data analysis
├── 01–04*.ipynb                    # DeepSeek & base model analysis
├── 05a–05f*.ipynb                  # Qwen3, Granite3.3, Magistral analysis
├── 11–13*.ipynb                    # Evaluation runners
├── 14–18*.ipynb                    # Comparative & few-shot analysis
└── 19_PAKDD_paper.ipynb            # paper results generation

results/                  # Predictions, metrics & visualizations
├── *_results.csv                   # Raw model predictions with timing
├── *_metrics.csv                   # Computed metrics per configuration
├── *_metrics.png                   # Visualization plots
└── *.pdf                           # Confusion matrices & analyses

pakdd/                    # PAKDD paper-specific artifacts
├── latex/                          # LaTeX source (main.tex, references.bib)
├── review/                         # Peer review materials
├── pareto_frontier*.pdf            # Pareto frontier figures
└── *_metrics.csv                   # Curated metrics for paper tables

scripts/                  # Shell scripts for batch evaluation
├── eval-qwen3.sh              # Qwen3 4B/8B/14B/32B (T/N modes)
├── eval-granite.sh            # Granite3.3-2B/8B (T/N modes)
├── eval-magistral.sh          # Magistral-24B (T/N modes)
├── eval-ollama-model.sh       # Generic Ollama model evaluation
├── eval-imdb.sh               # IMDB dataset evaluation
├── eval-goemotions.sh         # GoEmotions dataset evaluation
├── eval-all.sh                # Run all evaluations
└── create-*.sh                # Ollama model creation helpers
```

---

## Setup

**Python**: 3.10+ recommended

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment

```bash
cp .env.example .env
```

Set keys and paths in `.env`:
- `OPENAI_API_KEY` — for OpenAI models (or set to `Ollama` for local models)
- `DEEPSEEK_API_KEY` + `BASE_URL` — for DeepSeek API endpoints
- `GEMINI_API_KEY` — for Google Gemini models
- `DATA_PATH` — input dataset CSV file
- `RESULTS_PATH` — output CSV path
- `BASE_URL` — Ollama endpoint (default: `http://localhost:11434/v1`)
- `NUM_CTX` — context window size

### Set up Ollama models

Create all required Ollama models for evaluation:

```bash
bash scripts/create-all-models.sh
```

This script creates custom Ollama models with the necessary configurations for Qwen3, Granite3.3, and Magistral model families.

### Infrastructure

- **DeepSeek-R1 (full)** and **DeepSeek-V3**: via official DeepSeek APIs
- **All other models**: locally via [Ollama](https://ollama.com/) v0.6.8 on NVIDIA H100 GPU (96GB VRAM, Ubuntu 24.04.1 LTS)

---

## Running Evaluations

### Single model evaluation

```bash
bash scripts/eval-ollama-model.sh <model_name>
```

### Full evaluation suites

```bash
# Qwen3 family (4B, 8B, 14B, 32B × thinking/non-thinking × 3 datasets)
bash scripts/eval-qwen3.sh

# Granite3.3 family (2B, 8B × thinking/non-thinking × 3 datasets)
bash scripts/eval-granite.sh

# Magistral (24B × thinking/non-thinking × 3 datasets)
bash scripts/eval-magistral.sh

# All evaluations
bash scripts/eval-all.sh
```

### Analysis notebooks

Results analysis and figure generation are in `notebooks/`. Key notebooks for the extended paper:
- `notebooks/05a–05f*.ipynb` — Model-specific analysis for Qwen3, Granite3.3, Magistral
- `notebooks/20_PAKDD_paper.ipynb` — Paper table and figure generation

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If you use this repository or its results, please cite:

```bibtex
@inproceedings{huang2025task,
  title={Task Complexity Matters: An Empirical Study of Reasoning in LLMs for Sentiment Analysis},
  author={Huang, Donghao and Wang, Zhaoxia},
  booktitle={Proceedings of PAKDD},
  year={2025}
}
```

This work builds on the following earlier study:

```bibtex
@article{huang2025explainable,
  author={Huang, Donghao and Wang, Zhaoxia},
  journal={IEEE Intelligent Systems},
  title={Explainable Sentiment Analysis With DeepSeek-R1: Performance, Efficiency, and Few-Shot Learning},
  year={2025},
  volume={40},
  number={6},
  pages={52-63},
  doi={10.1109/MIS.2025.3614967}
}
```
