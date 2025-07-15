# NEIROTERM

A project for training and testing neural network models on sequences (e.g. bioinformatics, DNA/RNA classification).

## Project structure

```
NeuralNetworks/
├── launcher.py
├── FullyConnectedPerceptron/
│ ├── Perceptron.py
│ ├── NN_Perceptron_50_*_BCE.h5 (many weight files)
│ └── NN_Perceptron_50_*_weightedBCE.h5
└── utils/
├── __init__.py
├── DataUtils.py
├── WeightedBCE.py
└── registry.py
TrainData/
├── balance_by_ratio.py
├── balance_by_species.py
├── pad_nucleotides.py
└── BTFTrainData/ (folder with training data)
```

- **launcher.py** — entry point, automatically imports all models and runs them on all CSV files from the `../TrainData` folder.
- **FullyConnectedPerceptron/** — example implementation of MLP model (Perceptron) for binary classification.
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

- The script will automatically find all models and all CSV files (including nested ones), start training with two loss function modes: regular BCE and weighted BCE (`pos_weight = neg / pos`).
- For each model and each file, training logs (TSV) and weights (H5) are saved in the corresponding model folder.
- In case of errors (e.g. missing data or model), an informative message is displayed.

## Dependencies

- Python 3.8+
- numpy
- pandas
- scikit-learn
- tensorflow (Keras API)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Example metrics

```
Found models: ['Perceptron']
=== Data file: TrainData/example.csv ===
--- Running Perceptron [BCE] ---
Final metrics:
accuracy : 0.9123
precision : 0.8765
recall : 0.8342
neg_recall : 0.9456
```

## PS

To train the model, data from the repository were used:
https://github.com/BioinformaticsLabAtMUN/BacTermFinder

## License

MIT License