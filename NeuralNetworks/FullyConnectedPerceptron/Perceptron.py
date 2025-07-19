import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.metrics import Precision, Recall, MeanMetricWrapper  # type: ignore
import tensorflow.keras.backend as K  # type: ignore

from utils import DataUtils, weighted_BCE, register_model

def _negative_recall(y_true, y_pred):
    """
    Specificity = TN / (TN + FP)
    Proportion of correctly recognized negative examples.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    return tn / (tn + fp + K.epsilon())

# Number of epochs remains fixed
EPOCHS = 50

@register_model("Perceptron")
def PerceptronRUN(data_csv: str, use_weighted_bce: bool):
    """
    Entry point for Perceptron-MLP with hyperparameter sweep.
    Args:
        data_csv        : path to CSV with columns 'sequence' and 'class'
        use_weighted_bce: if True — weighted BCE, else regular BCE
    Returns:
        dict: empty (all results are printed & saved inside)
    """
    data_file = Path(data_csv)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # 1) Prepare data
    proc = DataUtils(str(data_file))
    hp = proc.get_hyperparams()
    seq_len, vocab_size = hp["seq_len"], hp["vocab_size"]
    seed = hp["random_seed"]

    np.random.seed(seed)
    tf.random.set_seed(seed)
    X_train, X_val, y_train, y_val = proc.get_processed_data()

    # 2) Define hyperparameter grids
    h1_list = [64, 128, 256, 512]
    h2_list = [32, 64, 128, 256]
    lr_list = np.logspace(-4, -2, num=5).tolist()
    bs_list = [16, 32, 64, 128]

    # 3) Prepare output directories
    base_dir   = Path(__file__).parent
    logs_dir   = base_dir / "logs"
    models_dir = base_dir / "saved_models"
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    stem = data_file.stem
    mode = "weightedBCE" if use_weighted_bce else "BCE"

    # 4) Sweep over all combinations
    for h1 in h1_list:
        for h2 in h2_list:
            for lr in lr_list:
                for bs in bs_list:
                    # 4.1) Build loss & model
                    if use_weighted_bce:
                        labels = y_train.numpy().flatten()  # type: ignore
                        neg, pos = np.bincount(labels, minlength=2)
                        pos_weight = neg / pos
                        loss_fn = weighted_BCE(pos_weight)
                    else:
                        loss_fn = "binary_crossentropy"

                    model = tf.keras.Sequential([                              # type: ignore
                        tf.keras.layers.Input(shape=(seq_len, vocab_size)),     # type: ignore
                        tf.keras.layers.Flatten(),                              # type: ignore
                        tf.keras.layers.Dense(h1, activation="relu"),           # type: ignore
                        tf.keras.layers.Dense(h2, activation="relu"),           # type: ignore
                        tf.keras.layers.Dense(1, activation="sigmoid")          # type: ignore
                    ])
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),   # type: ignore
                        loss=loss_fn,
                        metrics=[
                            "accuracy",
                            Precision(name="precision"),
                            Recall(name="recall"),
                            MeanMetricWrapper(_negative_recall, name="neg_recall")
                        ]
                    )

                    # 4.2) Prepare filenames
                    lr_str = f"{lr:.0e}"
                    log_file = (
                        f"TrainLOG_Perceptron_"
                        f"h1{h1}_h2{h2}_lr{lr_str}_bs{bs}_"
                        f"{stem}_{mode}.tsv"
                    )
                    model_file = (
                        f"NN_Perceptron_"
                        f"h1{h1}_h2{h2}_lr{lr_str}_bs{bs}_"
                        f"{EPOCHS}_{stem}_{mode}.h5"
                    )

                    # 4.3) Logging callback
                    log_path = logs_dir / log_file
                    tsv_logger = tf.keras.callbacks.CSVLogger(  # type: ignore
                        str(log_path), append=False, separator="\t"
                    )

                    # 4.4) Train
                    print(f"\n>>> Training: h1={h1}, h2={h2}, lr={lr_str}, bs={bs}, mode={mode}")
                    model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=EPOCHS,
                        batch_size=bs,
                        shuffle=True,
                        callbacks=[tsv_logger]
                    )

                    # 4.5) Save model
                    save_path = models_dir / model_file
                    model.save(str(save_path))

                    # 4.6) Evaluate
                    results = model.evaluate(X_val, y_val, verbose=0, return_dict=True)
                    print(" Results:", results)

    return {}
