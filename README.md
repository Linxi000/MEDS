# The Past Is Not Past: Memory-Enhanced Dynamic Reward Shaping

MEDS (Maximizing Exploration via Diversity Sampling) is a reinforcement learning recipe built on [veRL](https://github.com/volcengine/verl) for training Large Language Models (LLMs) with enhanced exploration through clustering-based diversity optimization.

## Overview

MEDS extends standard DAPO training by incorporating hidden state clustering to encourage diverse exploration during training. It extracts layer-wise hidden states, applies clustering algorithms HDBSCAN, and computes diversity-based rewards to improve model exploration.

## Key Features

- **Layer-wise Hidden State Analysis**: Extracts and analyzes hidden states from transformer layers during training
- **Clustering-Based Diversity Reward**: Uses HDBSCAN clustering to compute diversity penalties/rewards
- **Integration with veRL**: Built on top of the veRL framework for distributed RL training
- **Unified Math Data Processing**: Includes data preprocessing utilities for mathematical reasoning datasets
- **Ray-Based Distributed Training**: Supports multi-node training via Ray

## Project Structure

```
recipe/MEDS/
├── main_meds.py              # Main entry point for MEDS training
├── meds_ray_trainer.py       # Ray-based trainer implementation
├── extract_layer_logits.py   # Hidden state extraction utilities
├── layer_logits_utils.py     # Clustering and analysis utilities
├── unified_math_data.py      # Math dataset preprocessing
├── run_meds.sh               # Shell script for MEDS training
├── run_dapo.sh               # Shell script for DAPO baseline
├── config/
│   ├── dapo_trainer.yaml     # DAPO trainer configuration
│   └── dapo_megatron_trainer.yaml  # Megatron-based config
└── test_dapo_7b.sh           # Test script for 7B model
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MEDS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install veRL framework (ensure compatibility with your CUDA version).

## Usage

### Training with MEDS

Run the MEDS training script:

```bash
bash recipe/MEDS/run_meds.sh
```

Key configuration options in `run_meds.sh`:
- `cluster_method`: Clustering algorithm `hdbscan`
- `penalty_coef`: Coefficient for diversity penalty
- `use_layer_diff`: Whether to use layer difference for clustering
- `use_last_n_layers`: Number of last layers to use for clustering
- `cluster_penalty_target`: Target for penalty (`wrong`, `right`, `both`, `none`)

### Training with DAPO (Baseline)

Run the DAPO baseline:

```bash
bash recipe/MEDS/run_dapo.sh
```

## Configuration

The training configuration is managed via Hydra YAML files in `config/`:

- `dapo_trainer.yaml`: Standard DAPO trainer configuration
- `dapo_megatron_trainer.yaml`: Megatron-based trainer configuration

Key configuration sections:
- `data`: Dataset paths, batch sizes, prompt/response lengths
- `actor_rollout_ref`: Model configuration, rollout settings, PPO parameters
- `algorithm`: Advantage estimator, KL penalty, filtering options
- `reward_model`: Reward manager configuration
- `trainer`: Training loop settings, logging, checkpointing

## Data Format

MEDS uses the unified math data format. The `unified_math_data.py` script processes datasets into a standardized format:

```python
{
    "prompt": str,              # Problem text
    "solution": str,            # Full solution
    "ground_truth": str,        # Ground truth answer
    "data_source": str,         # Dataset source
    "ability": str,             # Problem ability level
}
```

## Key Components

### MEDS Trainer (`meds_ray_trainer.py`)

The `RayMEDSTrainer` class extends veRL's `RayPPOTrainer` with:
- KNN state management across epochs
- Hidden state extraction from transformer layers
- Integration with clustering utilities for diversity computation

### Layer Logits Extraction (`extract_layer_logits.py`)

Utilities for:
- Finding brace positions (`\boxed{}`) in model responses
- Computing absolute positions of key tokens
- Preparing batches for layer-wise analysis

### Clustering Utilities (`layer_logits_utils.py`)

Implements:
- HDBSCAN and HDBSCAN-KNN clustering
- Cluster penalty computation
- Logging and visualization of clustering results

## Citation

If you use MEDS in your research, please cite:

```bibtex

```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Acknowledgments

- Built on top of the [veRL](https://github.com/volcengine/verl) framework
- Inspired by [DAPO](https://github.com/BytedTS/DAPO) and other RL training recipes
- Uses [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) for clustering

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
