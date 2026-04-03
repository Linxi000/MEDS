# The Past Is Not Past: Memory-Enhanced Dynamic Reward Shaping

MEDS is a memory-enhanced RL training recipe for LLMs built on top of [veRL](https://github.com/volcengine/verl). Unlike standard memoryless reward designs, MEDS incorporates historical error signals into reward shaping, allowing the training process to recognize and discourage repeated mistakes.

To achieve this, MEDS reuses layer-wise logits from the forward pass as lightweight representations of reasoning behavior, clusters similar error patterns, and applies stronger penalties to repeated failures. This encourages broader exploration and leads to better reasoning performance and greater sampling diversity.

## Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Data Format](#data-format)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)


## Installation
Install the Python dependencies:
```bash
# Clone the MEDS repository
git clone https://github.com/Linxi000/MEDS.git
cd MEDS

# Create a new conda environment
conda create -n meds python=3.10
conda activate meds

# Install Python dependencies
pip install -r requirements.txt
```

MEDS is built on top of veRL. Please also install [veRL](https://github.com/volcengine/verl) and make sure your environment is compatible with the required CUDA version.

## Getting Started

### Training with MEDS

```bash
bash recipe/MEDS/run_meds.sh
```
You can modify key training options such as clustering method, penalty coefficient, and layer selection in `recipe/MEDS/run_meds.sh`.

### Training with DAPO (Baseline)

```bash
bash recipe/MEDS/run_dapo.sh
```

## Configuration

Training is configured through the shell scripts in `recipe/MEDS/` and the Hydra YAML files under `recipe/MEDS/config/`. These files control data loading, rollout settings, PPO training, reward shaping, and checkpointing.
The training configuration is managed via Hydra YAML files in `config/`.

Key configuration options in `run_meds.sh`:
- `cluster_method`: Clustering algorithm `hdbscan`
- `penalty_coef`: Coefficient for diversity penalty
- `use_layer_diff`: Whether to use layer difference for clustering
- `use_last_n_layers`: Number of last layers to use for clustering
- `cluster_penalty_target`: Target for penalty (`wrong`, `right`, `both`, `none`)

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
## Evaluation

这里要说一下咋跑的


## Citation

Our paper is available on [Arxiv](). If you find our code useful, please consider citing us!

```bibtex

```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on top of the [veRL](https://github.com/volcengine/verl) framework
- Inspired by [DAPO](https://github.com/BytedTS/DAPO) and other RL training recipes
- Uses [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) for clustering
