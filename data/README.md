# Dataset Creation README

## Overview
This dataset contains 1,667 examples for fine-tuning Llama-3.1-8B-Instruct for domain name generation with safety filtering. Generated using a hybrid manual and Claude API approach.

## Generation Methodology

### Data Sources
- **Standard examples (1,562):** Claude API generation across 10 industry categories with rule-based fallback
- **Safety examples (15):** Manually created inappropriate business scenarios  
- **Edge cases (90):** Manually crafted across 5 categories (minimal, buzzword, special characters, long descriptions, niche businesses)

### Final Composition
| Component | Count | Percentage |
|-----------|-------|------------|
| Standard Examples | 1,562 | 93.7% |
| Safety Examples | 15 | 0.9% |
| Edge Cases | 90 | 5.4% |
| **Total** | **1,667** | **100%** |

## Dataset Splits
Stratified 60/20/20 train/validation/test splits:

| Split | Count | Percentage |
|-------|-------|------------|
| Train | 1,000 | 60.0% |
| Validation | 333 | 20.0% |
| Test | 334 | 20.0% |

## Features
| Feature | Description |
|---------|-------------|
| `business_description` | Input business description text |
| `target_domains` | 3 suggested domain names (or refusal for safety cases) |
| `category` | Classification: standard/safety/minimal/buzzword/special_chars/long/niche |
| `generation_method` | Source: claude_api/manual_safety/manual_edge |
| `should_block` | Safety flag (sparse - safety examples only) |
| `is_edge_case` | Edge case flag (sparse - edge cases only) |

## Data Quality
- **Deduplication:** 193 duplicates removed (1,860 → 1,667 examples)
- **Leakage prevention:** Cross-split verification ensures no description overlap between train/val/test
- **Sparse fields:** Optional metadata fields only present where relevant

## Files
- `full_dataset.json` - Complete dataset (1,667 examples)
- `train_data.json` - Training split (1,000 examples)  
- `val_data.json` - Validation split (333 examples)
- `test_data.json` - Test split (334 examples)
- `dataset_metadata.json` - Generation statistics and metadata