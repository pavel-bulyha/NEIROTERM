import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.metrics import Precision, Recall, MeanMetricWrapper # type: ignore
import tensorflow.keras.backend as K # type: ignore

from utils import DataUtils, weighted_BCE, register_model

# Architectural constants
H1_UNITS   = 128
H2_UNITS   = 64

# Training parameters
LR         = 1e-3
EPOCHS     = 50
BATCH_SIZE = 64

def _negative_recall(y_true, y_pred):
    """
    Specificity = TN / (TN + FP)
    Proportion of correctly recognized negative examples.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tn = tf.reduce_sum((1 - y_true) * (1 - y_pred)) # type: ignore
    fp = tf.reduce_sum((1 - y_true) * y_pred) # type: ignore
    return tn / (tn + fp + K.epsilon())

@register_model("Perceptron")
def PerceptronRUN(data_csv: str, use_weighted_bce: bool):
    """
    Entry point for training Perceptron-MLP.
    Args:
        data_csv: path to CSV with columns 'sequence' and 'class'.
        use_weighted_bce: True — weighted BCE, False — regular.
    Returns:
        dict: final validation metrics.
    """
    data_file = Path(data_csv)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # 1) Data preparation
    proc = DataUtils(str(data_file))
    hp = proc.get_hyperparams()
    seq_len    = hp["seq_len"]
    vocab_size = hp["vocab_size"]
    seed       = hp["random_seed"]

    np.random.seed(seed)
    tf.random.set_seed(seed)
    X_train, X_val, y_train, y_val = proc.get_processed_data()

    # 2) Loss function
    stem = data_file.stem
    if use_weighted_bce:
        labels = y_train.numpy().flatten() # type: ignore
        neg, pos = np.bincount(labels, minlength=2)
        pos_weight = neg / pos
        loss_fn = weighted_BCE(pos_weight)
        mode = "weightedBCE"
    else:
        loss_fn = "binary_crossentropy"
        mode = "BCE"

    # 3) Model assembly and compilation
    model = tf.keras.Sequential([ # type: ignore
        tf.keras.layers.Input(shape=(seq_len, vocab_size), name="onehot_input"), # type: ignore
        tf.keras.layers.Flatten(name="flatten"), # type: ignore
        tf.keras.layers.Dense(H1_UNITS, activation="relu", name="dense_1"), # type: ignore
        tf.keras.layers.Dense(H2_UNITS, activation="relu", name="dense_2"), # type: ignore
        tf.keras.layers.Dense(1, activation="sigmoid", name="output") # type: ignore
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR), # type: ignore
        loss=loss_fn,
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            MeanMetricWrapper(_negative_recall, name="neg_recall")
        ]
    )

    # 4) Logging to TSV
    log_name = f"TrainLOG_Perceptron_{stem}_{mode}.tsv"
    log_path = Path(__file__).parent / log_name
    tsv_logger = tf.keras.callbacks.CSVLogger( # type: ignore
        str(log_path), append=False, separator="\t"
    )

    # 5) Training
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[tsv_logger]
    )

    # 5.1) Saving the trained model
    save_name = f"NN_Perceptron_{EPOCHS}_{stem}_{mode}.h5"
    save_path = Path(__file__).parent / save_name
    model.save(str(save_path))

    # 6) Final evaluation
    results = model.evaluate(X_val, y_val, verbose=0, return_dict=True)
    return results
