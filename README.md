# NEIROTERM

A project for training and testing neural network models on sequences (e.g. bioinformatics, DNA/RNA classification).

## Project structure

```
NeuralNetworks/
├── launcher.py
├── analyze_results.py
├── analysis_plots/
│ ├── best_balanced_acc_per_network.png
│ ├── bs_vs_balanced_acc_by_mode.png
│ ├── h1_vs_balanced_acc_by_mode.png
│ ├── h2_vs_balanced_acc_by_mode.png
│ ├── lr_vs_balanced_acc_by_mode.png
│ ├── param_corr_matrix.png
│ ├── ratio_vs_balanced_acc.png
│ └── recall_correlation.png
├── analysis_results/
│ ├── all_results.csv
│ ├── best_per_network.csv
│ └── overall_best.csv
├── ConvolutionalNeuralNetwork/
│ ├── CNN.py
│ ├── logs/
│ └── saved_models/
├── FullyConnectedPerceptron/
│ ├── Perceptron.py
│ ├── logs/
│ └── saved_models/
├── RecurrentNeuralNetwork/
│ ├── RNN.py
│ ├── logs/
│ └── saved_models/
└── utils/
    ├── __init__.py
    ├── DataUtils.py
    ├── WeightedBCE.py
    └── registry.py
TrainData/
├── TrainData.7z
├── balance_by_ratio.py
├── balance_by_species.py
├── pad_nucleotides.py
└── TrainData/
    └── BTFTrainData/ (folder with training data)
```

- **launcher.py** — entry point, automatically imports all models, displays available models with numbers, allows user to select specific models or run all, and runs selected models on all CSV files from the `../TrainData` folder.
- **analyze_results.py** — script for analyzing training logs from neural networks, parsing hyperparameters and metrics, generating plots and summary CSV files with best results per network and overall.
- **analysis_plots/** — directory containing generated plots for visualizing analysis results, such as balanced accuracy comparisons, hyperparameter correlations, and recall relationships.
- **analysis_results/** — directory with CSV files containing aggregated analysis data: all results, best per network, and overall best configuration.
- **ConvolutionalNeuralNetwork/** — implementation of Convolutional Neural Network (CNN) for sequence classification.
- **FullyConnectedPerceptron/** — implementation of Multi-Layer Perceptron (MLP) model for binary classification.
- **RecurrentNeuralNetwork/** — implementation of Recurrent Neural Network (RNN) for sequence data.
- **utils/** — auxiliary modules:
  - `DataUtils.py` — data preparation and encoding.
  - `WeightedBCE.py` — weighted BCE loss function for working with unbalanced classes.
  - `registry.py` — model registry (decorator for registering entry points).

## How to add your model

1. Create a new subfolder in the root of the project and add your `.py` file there.

2. Implement the launcher function, for example:
```python
from utils import register_model

@register_model("MyModel")
def MyModelRUN(data_csv: str, use_weighted_bce: bool):
# Your training logic
return {"accuracy": ..., "precision": ...}
```
3. launcher.py automatically imports all such functions.

## Preparing the data

- The data must be in CSV format with columns:
- `sequence` — a string (e.g. a nucleotide sequence)
- `class` — a label (0 or 1)
- All sequences must be the same length.
- CSV files are placed in the `../TrainData` folder.

## Launch

```bash
python launcher.py
```

- The script will automatically find all registered models, display them with numbers, and prompt the user to select specific models (by entering numbers separated by ';') or run all models (by entering 'all').
- Selected models will be run on all CSV files (including nested ones) from the `../TrainData` folder, with two loss function modes: regular BCE and weighted BCE (`pos_weight = neg / pos`).
- For each model and each file, training logs (TSV) and weights (H5) are saved in the corresponding model folder.
- In case of errors (e.g. missing data or model), an informative message is displayed.

## Dependencies

Python >= 3.8

Required packages:
- numpy >= 1.23.0          # vectorized operations and matrix math
- pandas >= 1.5.0          # CSV parsing and data handling
- scikit-learn >= 1.3.0    # preprocessing, metrics, data balancing
- tensorflow == 2.10.0     # core ML framework with Keras API, GPU-supported
- optuna                   # hyperparameter optimization framework
- seaborn                  # statistical data visualization
- matplotlib               # plotting and visualization

Install dependencies:
```bash
pip install -r requirements.txt
```

## Recommended: use a virtual environment to isolate dependencies
```bash
python -m venv .venv
source .venv/bin/activate      # on Unix/Mac
.\.venv\Scripts\activate       # on Windows
```

## Example output

```
Available models:
1. Perceptron
2. CNN
3. RNN
Enter model numbers separated by ';' or 'all' to run all: 1;3
Selected models: ['Perceptron', 'RNN']
=== Data file: TrainData/example.csv ===
--- Running Perceptron [BCE] ---
Final metrics:
accuracy : 0.9123
precision : 0.8765
recall : 0.8342
neg_recall : 0.9456
--- Running RNN [BCE] ---
Final metrics:
accuracy : 0.9234
precision : 0.8876
recall : 0.8453
neg_recall : 0.9567
```

## P.S.

To train the model, data from the repository were used:
https://github.com/BioinformaticsLabAtMUN/BacTermFinder

## License

MIT License
