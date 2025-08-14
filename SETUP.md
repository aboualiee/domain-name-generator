# Domain Name Generator - Setup Instructions for Reproducibility

## Colab-First Design (Recommended - Minimal Setup!)

### Why Colab?
This project is designed for **Google Colab** with automatic folder creation and dependency management. Everything "just works" once you follow the correct execution order!

This Colab design follows a streamlined workflow: Mount Drive, Configure Settings, Setup Environment, Add Datasets, Execute Training - with all other processes automated.

## Requirements
- Google Colab (recommended) or local GPU with 16GB+ VRAM  
- HuggingFace account and token
- Claude API key (for evaluation)
- Weights & Biases account (for experiment tracking)

### System Requirements
- GPU: 16GB+ VRAM (tested on A100)
- RAM: 32GB+ recommended
- Storage: 10GB for models and datasets
- CUDA: 11.8+ compatible

## Environment & Dependencies

### Step 1: Install Dependencies
```python
!pip install anthropic transformers datasets accelerate peft bitsandbytes
!pip install torch numpy pandas scikit-learn matplotlib tqdm wandb
```

### Step 2: Run Import Cell
```python
import os
import json
import random
import torch
import numpy as np
import anthropic
import time
# ... all other imports from the notebook
```

## Quick Start (Correct Execution Order)

### Step 3: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4: Define & Update Config Class
```python
@dataclass
class BaselineConfig:
    # Model config
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    hf_token: str = "YOUR_HUGGINGFACE_TOKEN_HERE"        # CHANGE THIS
    claude_api_key: str = "YOUR_CLAUDE_API_KEY_HERE"     # CHANGE THIS
    base_path: str = "/content/drive/MyDrive/domain_generator"
    
    # Training hyperparameters (for exact reproduction)
    max_length: int = 256
    max_new_tokens: int = 150
    temperature: float = 0.7
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    seed: int = 42  # CRITICAL for reproducibility
```

### Step 5: Run Helper Functions
```python
# Set seed function
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # ... rest of the function

# Setup folders function  
def setup_folders(path: str):
    os.makedirs(f"{path}/models", exist_ok=True)
    os.makedirs(f"{path}/freezes", exist_ok=True)
    os.makedirs(f"{path}/data", exist_ok=True)
    return path
```

### Step 6: Initialize Config & Create Folders
```python
# Initialize config
config = BaselineConfig()
set_seed(BaselineConfig.seed)

# This runs AFTER config is defined and creates folders automatically
base_path = setup_folders(config.base_path)
```

**Creates this structure automatically:**
```
/content/drive/MyDrive/domain_generator/
├── models/           # Auto-created - trained models saved here
├── freezes/          # Auto-created - experiment snapshots saved here
├── data/             # Auto-created - ADD DATASETS HERE MANUALLY
└── logs/             # Created during training
```

### Step 8: Weights & Biases Setup
```python
# Run wandb login cell - it will prompt for your API key
wandb.login()
# When prompted, enter your wandb API key from your wandb account

# Original experiment URLs for comparison:
# Baseline Model: https://wandb.ai/maikobi-epita/domain-generator/runs/pgjuz8gi
# Improved Model: https://wandb.ai/maikobi-epita/domain-generator/runs/mk3umccb
```

### Step 7: Add Datasets to Auto-Created `/data/` Folder

**For Exact Reproducibility (Recommended):**

**Option A: Download from GitHub:**
1. Download from: https://github.com/aboualiee/domain-name-generator/tree/main/data
2. Copy these files to: `/content/drive/MyDrive/domain_generator/data/`

**Option B: Download from HuggingFace:**
1. Download from: https://huggingface.co/datasets/Maikobi/domain-generation-dataset/tree/main
2. Copy these files to: `/content/drive/MyDrive/domain_generator/data/`

**Required Files:**
   - `test_data.json` (334 examples, final evaluation dataset)
   - `train_data.json` (1000 examples, original training data)
   - `val_data.json` (333 examples, validation data)

### Step 9: Load Datasets & View
```python
# Load datasets - runs AFTER datasets are added to /data/ folder
data_base = f"{config.base_path}/data"
with open(f"{data_base}/train_data.json") as f:
    train_data = json.load(f)
with open(f"{data_base}/val_data.json") as f:
    val_data = json.load(f)
with open(f"{data_base}/test_data.json") as f:
    test_data = json.load(f)

# View your datasets
print(f"Training: {len(train_data)} examples")
print(f"Validation: {len(val_data)} examples") 
print(f"Test: {len(test_data)} examples")
```

### Step 10: Run Training & Save Freeze Manifests
```python
# Run first training (baseline model)
baseline_path = train_model(config, train_generator_data, val_generator_data, "domain_generator_baseline_safety")

# IMPORTANT: After training completes, you'll see output like this:
# "Training complete. Model saved to /path/to/model"
# "Wandb run: https://wandb.ai/maikobi-epita/domain-generator/runs/abc123xyz"

# Copy the wandb URL from the training output above and paste it in the freeze function
# This saves a complete experiment manifest with all training details for reproducibility
freeze_model(
    model_path="models/baseline_v1",
    config=config,
    dataset_base="data/",
    wandb_run_url="https://wandb.ai/maikobi-epita/domain-generator/runs/abc123xyz",  # Replace with your actual URL
    model_name="baseline_safety",
    experiment_type="safety_training"
)

# Later, do the same for the improved model:
# 1. Run: improved_model_path = train_model(improved_config, augmented_train_data, val_data, "domain_generator_improved_v1")
# 2. Copy the new wandb URL from that training output
# 3. Run: freeze_model(..., wandb_run_url="YOUR_NEW_WANDB_URL_HERE", model_name="improved_v1", experiment_type="quality_improvement")
```

### Step 11: Evaluation, Augmentation & Improved Model Training

#### A. Evaluate Baseline Model
```python
# Run comprehensive evaluation on baseline model
baseline_evaluation = model_evaluator.evaluate_model(baseline_model, baseline_tokenizer, test_data, config)

# Analyze edge cases and failure modes
edge_analysis = analyze_edge_cases(baseline_evaluation)

# Display results
print("BASELINE MODEL EVALUATION RESULTS:")
print(f"Success Rate: {baseline_evaluation['metrics']['success_rate']:.1%}")
print(f"Overall Quality Score: {baseline_evaluation['metrics']['avg_overall_score']:.1f}/10")
```

#### B. Data Augmentation for Quality Improvement
```python
# Create augmented dataset to improve domain quality
augmented_train_data = augment_training_dataset(train_data, use_claude=False)

# This removes 150 generic examples and adds 34 quality examples
# Improves generic-to-creative domain ratio from 38% to 28% generic
```

#### C. Train Improved Model
```python
# Train second model with augmented data
improved_config = ImprovedConfig()
improved_model_path = train_model(improved_config, augmented_train_data, val_data, "domain_generator_improved_v1")

# Copy the wandb URL from improved model training output and freeze it
freeze_model(
    model_path="models/improved_v1",
    config=improved_config,
    dataset_base="data/",
    wandb_run_url="YOUR_IMPROVED_MODEL_WANDB_URL_HERE",  # Replace with actual URL
    model_name="improved_v1", 
    experiment_type="quality_improvement"
)
```

#### D. Evaluate & Compare Improved Model
```python
# Load and evaluate improved model
improved_model, improved_tokenizer = load_trained_model(improved_model_path, improved_config)
improved_evaluation = model_evaluator.evaluate_model(improved_model, improved_tokenizer, test_data, config)

# Compare baseline vs improved metrics
print("BASELINE vs IMPROVED COMPARISON:")
print(f"Baseline Quality: {baseline_evaluation['metrics']['avg_overall_score']:.1f}/10")
print(f"Improved Quality: {improved_evaluation['metrics']['avg_overall_score']:.1f}/10")

# Run detailed quality analysis
compare_baseline_vs_improved_detailed(baseline_model, baseline_tokenizer, improved_model, improved_tokenizer, config)
```
Execute training, evaluation, and freeze operations - everything auto-saves!

## Dataset Information

### Saved Datasets (Available at GitHub repo)

**test_data.json** (334 examples)
- Purpose: Final evaluation dataset used for all model comparisons
- Safety examples: 3 harmful requests for testing safety filtering
- Contains exact test cases used in research results

**train_data.json** (1000 examples) 
- Purpose: Original training dataset
- Safety examples: 9 examples for training-based safety
- Used for baseline model training

**val_data.json** (333 examples)
- Purpose: Validation dataset for training
- Safety examples: 3 examples
- Used for model validation during training

### Auto-Generated During Execution
```
/data/
├── test_data.json              # You add manually
├── train_data.json             # You add manually
├── val_data.json               # You add manually  
└── augmented_train_data.json   # Generated automatically during augmentation
```

### Column Descriptions
- **business_description**: Input business description for domain generation
- **target_domains**: Expected domain suggestions (3 domains per example)
- **category**: Example type (standard, safety, minimal, buzzword, etc.)
- **generation_method**: How example was created (claude_api, manual_safety, etc.)
- **should_block**: Boolean flag for content that should be refused (safety examples)

## Model Training & Freeze System

### What Gets Created Automatically

**Models Folder:**
```
/models/
├── domain_generator_baseline_safety_final/    # Auto-created during training
├── domain_generator_improved_v1_final/        # Auto-created during training
```

**Freezes Folder (Experiment Snapshots):**
```
/freezes/
├── safety_training_baseline_safety_[timestamp]/
│   └── manifest.json                          # Complete experiment metadata
└── quality_improvement_domain_generator_v1_[timestamp]/
    └── manifest.json                          # Complete experiment metadata
```

### Freeze Manifest Contents
Each freeze contains complete reproducibility metadata:
```json
{
  "tag": "safety_training_baseline_safety_20250813_042202",
  "model_path": "models/baseline_v1", 
  "wandb_run_url": "https://wandb.ai/maikobi-epita/domain-generator/runs/pgjuz8gi",
  "config": { /* all hyperparameters */ },
  "dataset_files": { /* paths to exact datasets */ },
  "dataset_sha256": { /* integrity hashes */ }
}
```

## Exact Reproduction Steps
1. **Install Dependencies** - Run pip install cell
2. **Run Imports** - Run the imports cell  
3. **Mount Drive** - Run drive mount cell
4. **Update Config** - Add API keys to BaselineConfig dataclass
5. **Run Helper Functions** - Run set_seed and setup_folders definitions
6. **Initialize Config** - Run config initialization and folder creation
7. **Add Datasets** - Copy JSON files to auto-created `/data/` folder
8. **Wandb Login** - Run wandb.login() and enter your API key when prompted
9. **Load & View Data** - Run dataset loading and verification
10. **Run Training** - Execute training cells, copy wandb URLs for freeze manifests
11. **Run Evaluation** - Execute evaluation framework
12. **Compare Results** - Against original wandb runs and freeze manifests

**Expected Results (on test set):**
- Baseline Model: 99.1% success rate, 7.6/10 overall quality score
- Improved Model: 99.1% success rate, 7.7/10 overall quality score  
- Safety Success Rate: 100% for both models

### What's Automated vs Manual

#### Fully Automated (Zero Setup)
- Dependencies installation via `!pip install`
- Folder structure creation via `setup_folders(config.base_path)`  
- Wandb configuration and experiment tracking
- Model training and saving to `/models/`
- Freeze manifest generation in `/freezes/`
- Dataset augmentation and processing
- Evaluation framework and metrics calculation

#### Manual Steps Required
- Mount Google Drive (1 cell)
- Update API keys in BaselineConfig dataclass (2 strings)
- Download and add 3 dataset files to auto-created `/data/` folder

### Pre-trained Models & Datasets
- **Improved Model**: https://huggingface.co/Maikobi/domain-name-generator
- **Dataset (HuggingFace)**: https://huggingface.co/datasets/Maikobi/domain-generation-dataset
- **Dataset (GitHub)**: https://github.com/aboualiee/domain-name-generator/tree/main/data

## File Structure After Complete Run
```
/content/drive/MyDrive/domain_generator/
├── models/
│   ├── domain_generator_baseline_safety_final/
│   └── domain_generator_improved_v1_final/
├── freezes/
│   ├── safety_training_baseline_safety_[timestamp]/
│   └── quality_improvement_domain_generator_v1_[timestamp]/
├── data/
│   ├── test_data.json
│   ├── train_data.json  
│   ├── val_data.json
│   └── augmented_train_data.json
└── logs/
    └── [training logs]
```

## Repository Links
- **Code**: https://github.com/aboualiee/domain-name-generator
- **Model**: https://huggingface.co/Maikobi/domain-name-generator  
- **Dataset (HuggingFace)**: https://huggingface.co/datasets/Maikobi/domain-generation-dataset
- **Dataset (GitHub)**: https://github.com/aboualiee/domain-name-generator/tree/main/data
- **Wandb Project**: https://wandb.ai/maikobi-epita/domain-generator