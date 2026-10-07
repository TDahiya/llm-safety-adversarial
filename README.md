# LLM Safety Consistency Under Adversarial Paraphrasing

> MSc Computing (AI Strand) Practicum — Dublin City University, 2025–2026

A systematic evaluation of whether large language models maintain consistent safety guardrails when prompts are reworded without changing their underlying intent. We paraphrase 329 red-team seed prompts across four linguistic transformation strategies, query eight frontier LLMs, and measure behavioural consistency using the ISCE metric family.

## Research Question

*Does surface-level linguistic variation cause frontier LLMs to inconsistently refuse or comply with harmful prompts — and which models are most robust?*

## Pipeline

```
Seed Prompts (329)
     │
     ├── Template Method (rule-based, deterministic)
     └── LLM Method (Qwen 3:14b via Ollama)
           │
           ▼
  Stage 2 · Prompt Validation (LLM-as-a-Judge)
           │
           ▼
  Stage 3 · Paraphrasing × 4 strategies
           │  • Lexical Substitution
           │  • Logic Reversal
           │  • Syntactic Restructuring
           │  • Contextual Embedding
           │
           ▼
  Stage 4 · LLM Querying (8 models, fresh sessions, no history)
           │
           ▼
  Stage 5 · G-EVAL Judge (3-judge panel, majority vote, dissent → human review)
           │
           ▼
        ISCE Metrics
```

## Models Evaluated

| Model | Provider |
|-------|----------|
| Claude Sonnet 4 | Anthropic |
| GPT-4o | OpenAI |
| DeepSeek-V4-Flash | DeepSeek |
| Mistral-Large-3 675B | Mistral AI |
| Qwen 3.5 | Alibaba |
| Gemini 2.5 Flash | Google |
| GLM 5.2 | Zhipu AI |
| Nemotron 3 Ultra | NVIDIA |

## ISCE Metrics

| Metric | Full Name | Description |
|--------|-----------|-------------|
| **MSA** | Mean Safety Accuracy | Overall refusal rate across all prompt variants |
| **UCR** | Under-refusal Consistency Rate | How often a model correctly refuses across all paraphrases of the same seed |
| **BUA** | Behaviour Under Ambiguity | Consistency when prompt intent is ambiguous |
| **ORR** | Over-refusal Rate | Benign prompts incorrectly refused |
| **IIR** | Intent Invariance Rate | Response consistency across paraphrase variants of the same seed |

## Repository Structure

```
src/
├── Data/                               # Datasets and results
│   ├── atomic_redteam_full.csv         # Source: Atomic Red Team (329 techniques)
│   ├── seed_prompts_combined_tagged.csv
│   ├── template_generated_seed_prompts.json
│   ├── llm_generated_prompts.json
│   └── geval_results_wide.xlsx         # Full G-EVAL results (8 models × all prompts)
└── Scripts/
    ├── 1_data_curation/                # Stage 1 · Seed prompt generation
    ├── 2_checking_prompts/             # Stage 2 · Prompt validation (LLM-as-a-Judge)
    ├── 3_paraphrasing/                 # Stage 3 · Four-strategy paraphrase pipeline
    ├── 4_querying/                     # Stage 4 · LLM querying
    └── 5_evaluation/                   # Stage 5 · G-EVAL judging + ISCE metrics
docs/
├── Final Paper/                        # Submitted research paper (PDF)
├── literature review/                  # Literature review (PDF)
└── proposal/                           # Project proposal (PDF)
```

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install pandas numpy openpyxl nltk spacy sentence-transformers openai anthropic requests
python -m spacy download en_core_web_sm
```

### API Keys

Create a `.env` file in the project root (never commit this):

```
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
OLLAMA_API_KEY=your_key
```

Stage 1 LLM method also requires [Ollama](https://ollama.ai) running locally with `qwen3:14b` pulled:

```bash
ollama pull qwen3:14b
```

## Usage

Run stages in order — each stage's output feeds the next.

### Stage 1 — Seed Prompt Generation

```bash
# Deterministic template method (no LLM required)
python src/Scripts/1_data_curation/template_based_seed_prompts_generator.py

# LLM-based method (requires Ollama + qwen3:14b)
python src/Scripts/1_data_curation/llm_based_seed_prompts_generator.py
```

### Stages 2–5

Open and run each Jupyter notebook in order inside `src/Scripts/`:

| Stage | Notebook | Description |
|-------|----------|-------------|
| 2 | `2_checking_prompts/LLM-As-A-Judge.ipynb` | Validate seed prompt quality |
| 3 | `3_paraphrasing/paraphrase_pipeline.ipynb` | Generate four paraphrase variants per seed |
| 4 | `4_querying/llm_query.ipynb` | Query all 8 models with all prompt variants |
| 5a | `5_evaluation/geval_judge.ipynb` | Run G-EVAL 3-judge scoring |
| 5b | `5_evaluation/geval_fix.ipynb` | Resolve dissenting votes |
| 5c | `5_evaluation/isce_eval.ipynb` | Compute ISCE metrics |

## Paper

Full paper: [LLM Safety Consistency Under Paraphrased OS Instructions](docs/Final%20Paper/LLM_Safety_Consistency_Under_Paraphrased_OS_Instructions.pdf)

Submitted: 19 July 2026, Dublin City University

## Authors

**Tanishq Dahiya** · [tanishq.dahiya2@mail.dcu.ie](mailto:tanishq.dahiya2@mail.dcu.ie)  
**Pragati Anant Tamboli** · [pragatianant.tamboli2@mail.dcu.ie](mailto:pragatianant.tamboli2@mail.dcu.ie)

## Disclaimer

This project is academic research on LLM safety alignment. All prompts were studied to *identify* guardrail weaknesses and improve model safety, not to cause harm. Experiments were conducted under DCU research ethics guidelines.
