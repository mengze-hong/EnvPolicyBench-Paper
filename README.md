# EnvPolicyBench

**A Self-Verifiable Benchmark for Predicting Environmental Outcomes from Policy Legislation**

> Anonymous ACL submission  
> Paper: [`policyeval_bench.tex`](policyeval_bench.tex)

---

## What is this?

**EnvPolicyBench** is the first *self-verifiable* benchmark for environmental policy NLP — every label is derived from **real-world indicator trajectories** measured after a law is enacted, not from human annotation.

We link **30,008 English-language environmental legislative records** from 179 countries (FAO FAOLEX, 1993--2020) to three-year post-enactment trajectories of **25 macro-environmental and economic indicators** (OWID, World Bank). Labels are produced via domain-aligned composite scoring combined with a pseudo-DiD regional baseline.

---

## Tasks

| Task | Description | Classes |
|------|-------------|---------|
| **T1 -- Direction** | Will this policy lead to improvement, no change, or deterioration? | 3-class (imp / neu / det) |
| **T2 -- Magnitude** | How large is the expected effect? | 4-class ordinal (0--3) |
| **T3 -- Geographic Transfer** | 18-country zero-shot holdout (3 per region) | same as T1 |
| **T4 -- Temporal OOD** | Macro-shock years (2007--2009, 2019--2020) | same as T1 |

---

## Key Results

### Main Results (Test set, n=1,826)

| Model | T1 Acc | T1 F1 | T1 kappa | T2 F1 |
|-------|:---:|:---:|:---:|:---:|
| TF-IDF + LR | 0.533 | 0.364 | 0.145 | 0.242 |
| Indicator-only MLP | 0.466 | 0.419 | 0.191 | 0.129 |
| Qwen3-8B zero-shot | --- | 0.206 | -0.042 | --- |
| Qwen3-8B SFT | --- | 0.388 | 0.000 | --- |
| **EnvPolicy-Dual (Qwen3-8B, ours)** | **0.596** | **0.412** | **0.281** | **0.225** |

### Zero-shot LLM Baselines (26 frontier models)

All 26 frontier LLMs tested on a stratified sample (n=498). Best performers:

| Model | T1 Acc | T1 F1 | T1 kappa | T4 OOD Acc | T4 OOD F1 |
|-------|:---:|:---:|:---:|:---:|:---:|
| Kimi-K2.5 | 0.321 | 0.293 | -0.018 | 0.317 | 0.285 |
| Gemini 2.5 Pro | 0.321 | 0.283 | -0.018 | 0.355 | 0.308 |
| DeepSeek-R1 | 0.311 | 0.283 | -0.033 | 0.343 | 0.302 |
| DeepSeek-V3.2 | 0.303 | 0.259 | -0.045 | **0.369** | **0.317** |

> Zero-shot LLMs (26 frontier models tested including GPT-5/5.5, Claude Opus 4.5/4.7, Gemini 3.1 Pro, DeepSeek-R1/V4-Pro) all fall **below** TF-IDF baselines on T1 F1 -- showing that policy outcome prediction requires grounding in indicator data, not just language understanding.

---

## Dataset

| Split | Records | Notes |
|-------|--------:|-------|
| Train | 17,704 | Country-year grouped, no leakage |
| Validation | ~3,000 | Grouped by (country, year) |
| Test | ~6,000 | Held-out |
| Test OOD (macro-shocks) | 5,789 | Policy years 2007--2009, 2019--2020 |
| T3 Holdout | ~3,000 | 18 countries, zero-shot |

Each record contains:
- Policy title + abstract text (English)
- 15-dimensional pre-enactment indicator vector
- Validity mask (which indicators are non-NaN)
- T1 direction label, T2 magnitude label, continuous composite score
- Label quality fields: `label_confidence` (high/medium/low), `label_consistent`, `label_tier` (Gold/Silver/Bronze)
- Legal lineage: `is_amendment`, `lineage_depth`
- Metadata: country code, policy year, ISO region, domain, FAOLEX URL

---

## The EnvPolicy-Dual Model

A gated cross-modal architecture fusing:
- **Text encoder**: Qwen3-8B (frozen, only fusion layers trained; 2.09M trainable parameters)
- **Indicator MLP**: masked MLP over 15 environmental/economic indicators
- **Fusion**: cross-attention gate between text and indicator representations

Training: AdamW + OneCycleLR, batch size 4, max seq len 256, 5 epochs on NVIDIA H20 GPUs. All seeds set to 42.

---

## Repository Structure

```
EnvPolicyBench-Paper/
│
├── README.md                        # This file
│
├── policyeval_bench.tex             # Main paper (LaTeX source)
├── references.bib                   # Bibliography
│
├── acl.sty                          # ACL conference style
├── acl_natbib.bst                   # ACL BibTeX style
│
└── figures/                         # LaTeX table sources
    ├── table_main_results.tex       #   Main results (T1 + T2, all models)
    ├── table_llm_zeroshot.tex       #   Zero-shot LLM results (26 models, T4 OOD)
    ├── table_ablation.tex           #   Ablation study
    ├── table_dataset_stats.tex      #   Dataset statistics
    └── table_domain_breakdown.tex   #   Domain breakdown
```

---

## How to Compile the Paper

```bash
# Option 1: latexmk (recommended)
latexmk -pdf policyeval_bench.tex

# Option 2: manual pdflatex
pdflatex policyeval_bench.tex
bibtex policyeval_bench
pdflatex policyeval_bench.tex
pdflatex policyeval_bench.tex

# Option 3: Overleaf
# Upload all files maintaining the same folder structure (root + figures/).
# Set policyeval_bench.tex as the main document.
```

Output: `policyeval_bench.pdf`

---

## How to Run the Code

> The full code (referenced in Appendix B of the paper) will be released upon de-anonymization.

```
src/
├── data/
│   └── preprocess.py      # Full preprocessing pipeline
└── models/
    └── train.py           # Training loop for all models
```

**Preprocessing**:
```bash
python src/data/preprocess.py \
    --faolex_raw data/raw/faolex_records.jsonl \
    --indicators data/raw/owid_worldbank_indicators.csv \
    --out_dir data/processed/
```

**Training EnvPolicy-Dual**:
```bash
python src/models/train.py \
    --model envpolicy_dual \
    --data_dir data/processed/ \
    --encoder qwen3-8b \
    --epochs 5 \
    --batch_size 4 \
    --seed 42
```

**Reproducibility notes**:
- All random seeds: 42 (`numpy`, `random`, `torch`)
- Country-year grouped splits prevent label leakage
- TF-IDF vocabulary fitted on training set only
- Indicator features z-score normalised using training-set statistics only
- Normalisation parameters saved to `norm_params.json`
- Macro-shock OOD split: years 2007--2009, 2019--2020

---

## Citation

```bibtex
@inproceedings{envpolicybench2026,
  title     = {{EnvPolicyBench}: A Self-Verifiable Benchmark for Predicting
               Environmental Outcomes from Policy Legislation},
  author    = {Anonymous},
  booktitle = {Proceedings of ACL 2026},
  year      = {2026},
}
```

---

## License

Data sourced from FAO FAOLEX (public domain), OWID (CC-BY), and World Bank Open Data (CC-BY 4.0).  
Code and paper: see LICENSE file (to be added upon de-anonymization).
