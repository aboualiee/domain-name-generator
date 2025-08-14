# Setup Instructions for Reproducibility

## Requirements:
- Google Colab (recommended) or local GPU with 16GB+ VRAM  
- HuggingFace account and token
- Claude API key (for evaluation)

## Before Running:
1. Replace API keys in the Config class below
2. Mount Google Drive or create local directories
3. Run cells sequentially

## Important: Dataset Reproducibility

**CRITICAL NOTE**: The dataset creation code uses Claude API which generates different results each time due to:
- API randomness and variability
- Different responses to the same prompts
- Potential changes in Claude's behavior over time

**For Exact Reproducibility**: 
- The exact datasets used in this research are saved in the `/data/` folder on GitHub: https://github.com/aboualiee/domain-name-generator/tree/main/data
- Download these files to reproduce the exact same results
- Skip the "Dataset Creation" section and load the saved datasets directly

**For New Experiments**:
- Run the dataset creation code to generate fresh synthetic data
- Results will be similar but not identical to the original research
- Useful for testing robustness and generalization

## Dataset Information

### Saved Datasets (available at https://github.com/aboualiee/domain-name-generator/tree/main/data):

**test_data.json** (247 examples, 101.9 KB)
- Purpose: Final evaluation dataset used for all model comparisons
- Columns (8): business_description, business_type, target_domains, complexity, location_type, category, generation_method, batch_number
- Safety examples: 4 harmful requests for testing safety filtering
- Contains the exact test cases used in the research results

**train_augmented_data.json** (823 examples, 329.7 KB) 
- Purpose: Enhanced training data for quality-improved model
- Columns (8): business_description, business_type, target_domains, complexity, location_type, category, generation_method, batch_number
- Safety examples: 9 examples
- Used for: Training the quality-enhanced model (iteration 2)

**safety_training_examples.json** (25 examples, 12.7 KB)
- Purpose: Training-based safety refusal examples
- Columns (6): business_description, target_domains, category, harm_type, generation_method, should_block
- Safety examples: 25 (all harmful requests with refusal responses)
- Categories: illegal_drugs, adult_content, violence, child_exploitation, fraud, hate_speech, piracy
- Used for: Training the safety-trained model (iteration 3)

### How to Use Saved Datasets:

# Clone the repository and load datasets
git clone https://github.com/aboualiee/domain-name-generator.git
cd domain-name-generator

# Load from downloaded repo (exact reproduction)
import json
with open("data/test_data.json", "r") as f:
    test_data = json.load(f)

### Column Descriptions:
* **business_description**: Input business description for domain generation
* **target_domains**: Expected domain suggestions (3 domains per example)
* **business_type**: Category (food_beverage, tech, professional, etc.)
* **complexity**: Word count of business description
* **location_type**: Geographic context (urban, suburban, rural, online)
* **category**: Example type (standard, safety, quality_enhanced)
* **generation_method**: How example was created (claude_api, manual_safety, etc.)
* **harm_type**: Type of harmful content (for safety examples only)
* **should_block**: Boolean flag for content that should be refused

### Model Training Data Progression:
* **Baseline Model**: 700 train examples (original dataset)
* **Quality-Enhanced Model**: 823 examples (700 + 123 quality examples)
* **Safety-Trained Model**: 848 examples (823 + 25 safety examples)

### Usage Options:
1. **Exact Reproduction**
   * Clone https://github.com/aboualiee/domain-name-generator
   * Use saved datasets from `/data/` folder
   * Run training/evaluation sections

2. **Fresh Experiments**
   * Run the dataset creation sections to generate new data
   * Compare results with original findings
   * Note: Results will vary due to API randomness

3. **Evaluation Only**
   * Load pre-trained model checkpoints
   * Run evaluation framework
   * Skip training sections

## Update these in the Config class with your actual API keys 

class Config:
    hf_token: str = "YOUR_HUGGINGFACE_TOKEN_HERE"
    claude_api_key: str = "YOUR_CLAUDE_API_KEY_HERE"
    
    # Paths (will be created automatically)
    base_path: str = "/content/drive/MyDrive/domain_generator"