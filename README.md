# Domain Name Generator - Fine-tuned Llama-3.1-8B-Instruct

Fine-tuned Llama-3.1-8B-Instruct to generate creative domain names for businesses with built-in safety filtering.

## Features

- **Creative domain generation**: Generates brandable domains like "beanwise.co" and "fitwise.app"
- **Safety filtering**: Refuses inappropriate business requests
- **Multiple TLDs**: Uses modern extensions (.co, .io, .app, .studio, .pro)
- **Edge case handling**: Works with minimal inputs and complex descriptions

## Try It Live

**Demo**: [https://huggingface.co/spaces/Maikobi/domain-name-generator](https://huggingface.co/spaces/Maikobi/domain-name-generator)

Enter any business description and get 3 creative domain suggestions instantly.

## Results & Performance

**Model Performance:**
- 99.1% success rate on test set
- 100% safety compliance (blocks inappropriate requests)
- 7.6/10 average quality score (Claude evaluation)

**Key Improvements:**
- Baseline to Improved: +0.02 overall quality
- Reduced generic domains by 15%
- Better handling of minimal inputs
- Improved brandability and modern extensions

## Model & Dataset

**Model**: [Maikobi/domain-name-generator](https://huggingface.co/Maikobi/domain-name-generator) (improved model)  
**Dataset**: [Maikobi/domain-generation-dataset](https://huggingface.co/datasets/Maikobi/domain-generation-dataset)

- 1,667 examples across business categories
- Hybrid Claude API + manual curation approach
- 60/20/20 train/validation/test splits
- Comprehensive safety and edge case coverage

**For Reproducibility**: Use pre-generated datasets from HuggingFace/GitHub rather than regenerating, as Claude API produces different results each time. See [data/README.md](data/README.md) for detailed guidance.

## Technical Implementation

**Training Setup:**
- GPU: 16GB+ VRAM recommended (tested on A100/you can try on T4/L4)
- Memory optimization: 4-bit quantization + LoRA
- Evaluation: LLM-as-a-Judge (Claude-3-Haiku)

**Model Details:**
- **Base**: Llama-3.1-8B-Instruct  
- **Method**: LoRA (rank=8, alpha=16, 3 epochs)  
- **Dataset**: Hybrid Claude API + manual curation approach

**Safety Guardrails:**
- Training includes inappropriate request handling
- Refusal training with 15 safety examples (9 training, 3 validation, 3 test)
- Edge case testing (90 examples across 5 categories)

## Evaluation Framework

**Methodology:**
- LLM-as-a-Judge evaluation with Claude
- Edge case analysis across 6 categories
- Safety testing with inappropriate requests
- Systematic quality metrics (relevance, memorability, brandability)

**Test Categories:**
- Standard business descriptions (313 examples)
- Minimal inputs (3 examples)
- Buzzword-heavy descriptions (6 examples)
- Special characters (2 examples)
- Long descriptions (5 examples)
- Niche businesses (2 examples)
- Safety cases (3 examples)

## Usage Examples

**Single request:**
```json
{"business_description": "organic coffee shop in downtown area"}
```

**Batch requests:**
```json
[
  {"business_description": "organic coffee shop in downtown area"},
  {"business_description": "eco-friendly fashion brand"}
]
```

**Output format:**
```json
{
  "suggestions": ["beanwise.co", "brewcraft.coffee", "originroast.com"],
  "status": "success"
}
```

**Safety example:**
```json
{
  "business_description": "illegal drug marketplace",
  "suggestions": [],
  "status": "blocked",
  "message": "Request contains inappropriate content"
}
```

## Reproduction & Setup

**Quick Start:**
```bash
# 1. Clone repository
git clone https://github.com/aboualiee/domain-name-generator
cd domain-name-generator

# 2. Follow detailed setup instructions
# See SETUP.md for complete Colab-based reproduction
```

**Complete Instructions**: See **[SETUP.md](SETUP.md)** for:
- Step-by-step Colab setup (recommended)
- API key configuration
- Exact reproduction methodology
- Expected results and validation

## Important Notes

- **Reproducibility**: Use pre-generated datasets for exact reproduction. Claude API generates different results each run.
- **Requirements**: 16GB+ VRAM GPU recommended (tested on A100)
- **Setup**: Designed for Google Colab with automatic folder management

## Research & Experiments

**Experiment Tracking:**
- Baseline Model: [Run pgjuz8gi](https://wandb.ai/maikobi-epita/domain-generator/runs/pgjuz8gi)
- Improved Model: [Run mk3umccb](https://wandb.ai/maikobi-epita/domain-generator/runs/mk3umccb)

## Documentation

- **[SETUP.md](SETUP.md)** - Complete setup and reproduction instructions
- **[data/README.md](data/README.md)** - Dataset creation methodology and usage guidelines
- **[notebooks/](notebooks/)** - Training and evaluation notebooks

## Repository Structure

```
├── notebooks/
│   ├── domain_generator.ipynb          # Main training pipeline
│   └── dataset_creation.ipynb          # Dataset generation
├── data/
│   ├── README.md                       # Dataset documentation
│   ├── full_dataset.json              # Complete dataset (1,667 examples)
│   ├── train_data.json                 # Training examples (1,000)
│   ├── val_data.json                   # Validation examples (333)
│   ├── test_data.json                  # Test examples (334)
│   └── train_data_augmented.json       # Improved training data (884)
├── SETUP.md                            # Complete setup guide
└── README.md                           # This file
```

## License

MIT License