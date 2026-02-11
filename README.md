# NeiroTerm V1.1.1

A project for training and testing neural network models on sequences. This version is compatible with Linux systems and TensorFlow 2.20.0+.

## Project Structure

```
NeuralNetworks/
├── launcher.py
├── analyze_results.py
├── analysis_plots/
│ ├── balanced_acc_boxplot.png
│ ├── best_val_f1_boxplot.png
│ ├── common_hyperparams_pairplot.png
│ ├── common_params_parallel_coords.png
│ ├── full_corr_matrix.png
│ ├── hyperparams_pairplot.png
│ └── [network]_corr_matrix.png
├── analysis_results/
│ ├── all_results_extended.csv
│ ├── best_per_network.csv
│ ├── best_per_dataset.csv
│ ├── overall_best.csv
│ └── top_10.csv
├── ConvolutionalNeuralNetwork/
│ ├── CNN.py
│ ├── optuna_cnn.db
│ ├── logs/
│ └── saved_models/
├── FullyConnectedPerceptron/
│ ├── Perceptron.py
│ ├── optuna_perceptron.db
│ ├── logs/
│ └── saved_models/
├── RecurrentNeuralNetwork/
│ ├── RNN.py
│ ├── optuna_rnn.db
│ ├── logs/
│ └── saved_models/
└── utils/
    ├── __init__.py
    ├── DataUtils.py
    ├── WeightedBCE.py
    ├── balanced_accuracy.py
    ├── focal_loss.py
    └── registry.py
TrainData/
├── TrainData.7z
├── balance_by_ratio.py
├── balance_by_species.py
├── pad_nucleotides.py
└── TrainData/
    └── BTFTrainData/ (folder with training data)
```

- **balance_by_ratio.py** — script for balancing datasets by class ratio (e.g., to address class imbalance).
- **balance_by_species.py** — script for balancing datasets by organism species (biological categories).
- **pad_nucleotides.py** — script for padding nucleotide sequences to the same length.
- **launcher.py** — entry point, automatically imports all models, displays available models with numbers, allows user to select specific models or run all, and runs selected models on all CSV files from the `../TrainData` folder.
- **analyze_results.py** — advanced script for analyzing neural network training logs, supports multiple model architectures (CNN, Perceptron, RNN), generates comprehensive plots (boxplots, pairplots, correlation matrices, parallel coordinates) and detailed CSV reports with extended metrics (F1, balanced accuracy, precision, recall).
- **analysis_plots/** — directory containing generated plots for visualizing analysis results: boxplots of balanced accuracy and F1 scores per network and loss function, violin plots, hyperparameter pairplots, correlation matrices, and parallel coordinates for parameter analysis.
- **analysis_results/** — directory with CSV files containing aggregated analysis data: extended results, best per network, best per dataset, overall best configuration, and top 10 results.
- **ConvolutionalNeuralNetwork/** — implementation of Convolutional Neural Network (CNN) for sequence classification with intelligent parameter filtering and cuDNN support.
- **FullyConnectedPerceptron/** — implementation of Multi-Layer Perceptron (MLP) model for binary classification.
- **RecurrentNeuralNetwork/** — implementation of Recurrent Neural Network (RNN) for sequence data.
- **utils/** — auxiliary modules:
  - `DataUtils.py` — data preparation and encoding.
  - `WeightedBCE.py` — weighted BCE loss function for handling imbalanced classes.
  - `balanced_accuracy.py` — balanced accuracy loss function.
  - `focal_loss.py` — focal loss function for hard examples.
  - `registry.py` — model registry (decorator for registering entry points).

## Adding Your Own Model

1. Create a new subfolder in the project root and add your `.py` file there.

2. Implement the launcher function, for example:
```python
from utils import register_model

@register_model("MyModel")
def MyModelRUN(data_csv: str, loss_type: str):
    # Your training logic
    return {"accuracy": ..., "precision": ...}
```
3. launcher.py automatically imports all such functions.

## Quickstart

Minimal set of commands to run:

```bash
# Clone the latest stable version (V1.1.1)
git clone -b linux https://github.com/pavel-bulyha/NEIROTERM.git
cd NEIROTERM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python NeuralNetworks/launcher.py
```

## Data Preparation

### Input CSV Format and Schema

Data must be in CSV format with required columns:

- `sequence` — string containing the sequence (e.g., nucleotide: `ATCGATCG...`)
- `class` — integer, class label (0 or 1)

All sequences in the file must be the same length. The file must be encoded in UTF-8-BOM (with BOM) or UTF-8.

#### Example CSV file:

```
sequence,class
ATCGATCGATCG,0
GCTAGCTAGCTA,1
TTTTAAAA,0
```

CSV files are placed in the `../TrainData` folder (relative to NeuralNetworks/).

## Running

```bash
python launcher.py
```

- The script will automatically find all registered models, display them with numbers, and prompt the user to select specific models (by entering numbers separated by ';') or run all (by entering 'all').
- Selected models will be run on all CSV files (including nested ones) from the `../TrainData` folder, with four loss function modes: BCE, weighted BCE (`pos_weight = neg / pos`), balanced accuracy, and focal loss.
- For each model and each file, training logs (TSV) and weights (H5) are saved in the corresponding model folder.
- In case of errors (e.g., missing data or model), an informative message is displayed.

## Output Details

### Training Logs
- **Location**: `NeuralNetworks/[ModelName]/logs/`
- **Format**: TSV files named like `{dataset}_{mode}_{params}.tsv`
- **Contents**: Columns per epoch:
  - `epoch` — epoch number (1-50)
  - `loss` — loss value on training
  - `accuracy` — accuracy on training
  - `precision` — precision on training
  - `recall` — recall on training
  - `neg_recall` — recall for negative class on training
  - `val_loss` — loss on validation
  - `val_accuracy` — accuracy on validation
  - `val_precision` — precision on validation
  - `val_recall` — recall on validation
  - `val_neg_recall` — recall for negative class on validation

  Example of first rows:
  ```
  epoch	loss	accuracy	precision	recall	neg_recall	val_loss	val_accuracy	val_precision	val_recall	val_neg_recall
  1	0.693	0.500	0.500	0.500	0.500	0.693	0.500	0.500	0.500	0.500
  2	0.690	0.525	0.525	0.525	0.525	0.690	0.525	0.525	0.525	0.525
  ...
  ```

  These logs are used by the `analyze_results.py` script to compute additional metrics (balanced accuracy, F1-score).

### Saved Models
- **Location**: `NeuralNetworks/[ModelName]/saved_models/`
- **Format**: Folders named like `{dataset}_{mode}_{params}/` containing `epoch_01.h5`, `epoch_02.h5`, ..., `epoch_50.h5`
- **Contents**: Keras weight files (.h5) for each training epoch.

### Analysis Reports
- **Location**: `NeuralNetworks/analysis_results/`
- **Files**:
  - `all_results_extended.csv` — all results with metrics
  - `best_per_network.csv` — best per network
  - `best_per_dataset.csv` — best per dataset
  - `overall_best.csv` — overall best
  - `top_10.csv` — top 10 results

### Analysis Plots
- **Location**: `NeuralNetworks/analysis_plots/`
- **Files**: PNG plots (boxplots, pairplots, correlation matrices, parallel coordinates)

## Analyzing Results

```bash
python analyze_results.py
```

- The script analyzes all training logs from model subfolders, generates plots and CSV reports in `analysis_plots/` and `analysis_results/`.

## System Requirements

- **Linux**
- **Python 3.10-3.13**
- **GPU with CUDA support** (optional)
- **TensorFlow 2.20.0+**

## Dependencies

Required packages:
- tensorflow >= 2.20.0     # core ML framework with Keras API, GPU support
- numpy >= 1.23.0          # vectorized operations and matrix math
- pandas >= 1.5.0          # CSV parsing and data handling
- scikit-learn >= 1.3.0    # preprocessing, metrics, data balancing
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
source .venv/bin/activate
```

## GPU Support

The project includes comprehensive GPU support with automatic fallback:

- **CNNOptunaOriginal**: Full GPU support with intelligent parameter filtering and cuDNN compatibility
- **PerceptronOptuna**: Full GPU support
- **RNNOptuna**: Full GPU support with cuDNN for performance

All models automatically detect GPUs, configure memory, and use mixed precision for performance.

## CUDA Requirements

For GPU support with TensorFlow 2.20.0+:

- **CUDA Toolkit 12.x**
- **cuDNN 9.x**
- **NVIDIA Driver 550+**

Check CUDA version:
```bash
nvcc --version
```

Check GPU devices:
```bash
python -c "import tensorflow as tf; print('GPU devices:', len(tf.config.list_physical_devices('GPU')))"
```

## Example Output

```
Available models:
1. CNNOptunaOriginal
2. PerceptronOptuna
3. RNNOptuna
Enter model numbers separated by ';' or 'all' to run all: 1;2
Selected models: ['CNNOptunaOriginal', 'PerceptronOptuna']

🎥 CNN GPU INITIALIZATION
============================================================
📊 Available devices:
   CPU devices: 1
   GPU devices: 1
🎮 GPU DEVICES DETECTED:
   GPU 0: /physical_device:GPU:0
         Device: NVIDIA GeForce RTX 4060 Ti
         Compute capability: (8, 9)
⚙️  GPU Configuration:
   - Memory growth: ENABLED (dynamic allocation)
   - Mixed precision: ENABLED (for CNN performance)
   - Status: ✅ GPU READY FOR TRAINING

🚀 Trial 0 - TRAINING START
   Architecture: Input(20,4) → Conv1D(10,1) → Dense(10) → Output(1)
   Hyperparameters: LR=4.11e-03, Batch=32, DropoutConv=0.0, DropoutDense=0.0
   Dataset: 160 train, 40 validation samples
   Device context: /GPU:0
   ✅ Model initialized on: /job:localhost/replica:0/task:0/device:GPU:0 (GPU)
   📊 Model parameters: 46,537
```

## Troubleshooting

### Data Errors
- **"All sequences must have the same length"**: Ensure all sequences in the CSV file have the same length. Use `pad_nucleotides.py` for padding.
- **"FileNotFoundError"**: Check that the TrainData folder exists and contains CSV files.
- **CSV encoding issues**: Save the file in UTF-8 or UTF-8-BOM.

### Model Errors
- **"Invalid units" or unsuitable parameters**: The model may prune unsuitable hyperparameters (TrialPruned). This is normal, Optuna will try others.
- **Model import errors**: Ensure all model files are in correct subfolders and contain the `@register_model` decorator.

### GPU/CUDA Errors
- **GPU not detected**: The model will automatically switch to CPU. Install CUDA Toolkit 12.x and cuDNN 9.x for GPU support.
- **Out of memory on GPU**: Reduce batch size or disable mixed precision.

### Other Issues
- **Slow performance**: Ensure GPU is being used (check initialization output).

If the problem persists, check console logs and files in `logs/`.

## P.S.

The model training used data from the repository:
https://github.com/BioinformaticsLabAtMUN/BacTermFinder

## License

MIT License
