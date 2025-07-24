import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.metrics import Precision, Recall, MeanMetricWrapper # type: ignore
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras import mixed_precision # type: ignore
from tensorflow.keras.callbacks import CSVLogger # type: ignore

from utils import DataUtils, weighted_BCE, register_model

# 0) GPU & mixed-precision setup
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Enable memory growth so TF doesn't grab all GPU memory at once
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Activate mixed-precision training for speed on modern GPUs
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision policy:", mixed_precision.global_policy())

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

EPOCHS = 50

@register_model("PerceptronCUDA")
def PerceptronRUN(data_csv: str, use_weighted_bce: bool):
    """
    Runs a 2-layer perceptron with optional weighted BCE loss,
    sweeping hyperparameters and logging to TSV + saving models.
    """
    data_file = Path(data_csv)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # 1) Load & preprocess data
    proc = DataUtils(str(data_file))
    hp = proc.get_hyperparams()
    seq_len, vocab_size, seed = hp["seq_len"], hp["vocab_size"], hp["random_seed"]

    np.random.seed(seed)
    tf.random.set_seed(seed)
    X_train, X_val, y_train, y_val = proc.get_processed_data()

    # 2) Hyperparameter grids
    h1_list = [64, 128, 256, 512]
    h2_list = [32, 64, 128, 256]
    lr_list = np.logspace(-4, -2, num=5).tolist()
    bs_list = [16, 32, 64, 128]

    # 3) Prepare output folders
    base_dir   = Path(__file__).parent
    logs_dir   = base_dir / "logs"
    models_dir = base_dir / "saved_models"
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    stem = data_file.stem
    mode = "weightedBCE" if use_weighted_bce else "BCE"

    # 4) Grid search
    for h1 in h1_list:
        for h2 in h2_list:
            for lr in lr_list:
                for bs in bs_list:
                    # 4.1) Loss function
                    if use_weighted_bce:
                        labels = y_train.numpy().flatten() # type: ignore
                        neg, pos = np.bincount(labels, minlength=2)
                        pos_weight = neg / pos
                        loss_fn = weighted_BCE(pos_weight)
                    else:
                        loss_fn = "binary_crossentropy"

                    # 4.2) Optimizer, wrapped for mixed precision if GPU is available
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                    if gpus:
                        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

                    # 4.3) Build model
                    model = tf.keras.Sequential([
                        tf.keras.layers.Input(shape=(seq_len, vocab_size)),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(h1, activation="relu"),
                        tf.keras.layers.Dense(h2, activation="relu"),
                        # Force output to float32 so metrics run in full precision
                        tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")
                    ])

                    model.compile(
                        optimizer=optimizer,
                        loss=loss_fn,
                        metrics=[
                            "accuracy",
                            Precision(name="precision"),
                            Recall(name="recall"),
                            MeanMetricWrapper(_negative_recall, name="neg_recall")
                        ]
                    )

                    # 4.4) File names for logs & model
                    lr_str = f"{lr:.0e}"
                    log_file = (
                        f"TrainLOG_Perceptron_h1{h1}_h2{h2}_lr{lr_str}"
                        f"_bs{bs}_{stem}_{mode}.tsv"
                    )
                    model_file = (
                        f"NN_Perceptron_h1{h1}_h2{h2}_lr{lr_str}"
                        f"_bs{bs}_{EPOCHS}_{stem}_{mode}.h5"
                    )

                    # 4.5) Set up callback
                    tsv_logger = CSVLogger(
                        str(logs_dir / log_file),
                        append=False, separator="\t"
                    )

                    # 4.6) Train (will run on GPU if available)
                    print(
                        f"\n>>> Training h1={h1}, h2={h2}, lr={lr_str}, bs={bs}, mode={mode}"
                    )
                    model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=EPOCHS,
                        batch_size=bs,
                        shuffle=True,
                        callbacks=[tsv_logger]
                    )

                    # 4.7) Save & evaluate
                    save_path = models_dir / model_file
                    model.save(str(save_path))

                    results = model.evaluate(
                        X_val, y_val,
                        verbose=0,             # type: ignore[reportArgumentType]
                        return_dict=True
                    )
                    print(" Evaluation results:", results)

    return {}
