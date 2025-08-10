import multiprocessing
import numpy as np
import tensorflow as tf
import optuna

from pathlib import Path
from tensorflow.keras.metrics import Precision, Recall, MeanMetricWrapper
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import mixed_precision, backend as K
from tensorflow.data import AUTOTUNE

from utils import DataUtils, weighted_BCE, register_model

from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned

# 0) GPU & mixed-precision setup
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    mixed_precision.set_global_policy("mixed_float16")

# 1) CPU threading optimization
num_cpu_cores = multiprocessing.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(num_cpu_cores)
tf.config.threading.set_intra_op_parallelism_threads(num_cpu_cores)

# 2) Global constants
EPOCHS = 50

# 2.1) Явный путь к SQLite-файлу рядом с Perceptron.py
BASE_DIR    = Path(__file__).parent.resolve()
DB_NAME     = "optuna_perceptron.db"
DB_PATH     = BASE_DIR / DB_NAME
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
STORAGE_URL = f"sqlite:///{DB_PATH}"

print(f"Optuna storage path: {DB_PATH}")

@register_model("PerceptronOptunaCPU")
def PerceptronOptunaCPU(data_csv: str, use_weighted_bce: bool):
    """
    Hyperparameter tuning + CPU/GPU optimization + tf.data + MedianPruner + explicit SQLite storage.
    """
    # 3) Load & preprocess
    data_file = Path(data_csv)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    proc = DataUtils(str(data_file))
    hp = proc.get_hyperparams()
    seq_len, vocab_size, seed = hp["seq_len"], hp["vocab_size"], hp["random_seed"]
    np.random.seed(seed)
    tf.random.set_seed(seed)
    X_train, X_val, y_train, y_val = proc.get_processed_data()

    # 4) Loss
    if use_weighted_bce:
        labels = y_train.numpy().flatten()
        neg, pos = np.bincount(labels, minlength=2)
        pos_weight = neg / (pos + K.epsilon())
        loss_fn = weighted_BCE(pos_weight)
    else:
        loss_fn = "binary_crossentropy"

    # 5) Search space
    h1_list = [64, 128, 256, 512]
    h2_list = [32, 64, 128, 256]
    lr_list = np.logspace(-4, -2, num=5).tolist()
    bs_list = [16, 32, 64, 128]

    # 6) Output dirs
    logs_dir   = BASE_DIR / "logs"
    models_dir = BASE_DIR / "saved_models"
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    stem = data_file.stem
    mode = "weightedBCE" if use_weighted_bce else "BCE"

    # 7) Objective
    def objective(trial: optuna.Trial) -> float:
        h1 = trial.suggest_categorical("h1", h1_list)
        h2 = trial.suggest_categorical("h2", h2_list)
        lr = trial.suggest_loguniform("lr", lr_list[0], lr_list[-1])
        bs = trial.suggest_categorical("bs", bs_list)

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(10_000)
            .batch(bs)
            .prefetch(AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(bs)
            .prefetch(AUTOTUNE)
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if gpus:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(seq_len, vocab_size)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(h1, activation="relu"),
            tf.keras.layers.Dense(h2, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32"),
        ])
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                "accuracy",
                Precision(name="precision"),
                Recall(name="recall"),
                MeanMetricWrapper(
                    lambda y_t, y_p: (
                        tf.reduce_sum((1 - tf.cast(y_t, tf.float32)) *
                                      (1 - tf.cast(y_p > 0.5, tf.float32)))
                        / (tf.reduce_sum(1 - tf.cast(y_t, tf.float32)) + K.epsilon())
                    ),
                    name="neg_recall"
                )
            ]
        )

        lr_str   = f"{lr:.0e}"
        log_file = logs_dir / f"TrainLOG_h1{h1}_h2{h2}_lr{lr_str}_bs{bs}_{stem}_{mode}.tsv"
        tsv_logger = CSVLogger(str(log_file), separator="\t", append=False)

        best_recall = 0.0
        for epoch in range(EPOCHS):
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=1,
                callbacks=[tsv_logger],
                verbose=0
            )
            val_recall = history.history["val_recall"][-1]
            best_recall = max(best_recall, val_recall)
            trial.report(val_recall, step=epoch)
            if trial.should_prune():
                raise TrialPruned()

        model_file = models_dir / (
            f"NN_h1{h1}_h2{h2}_lr{lr_str}_bs{bs}_"
            f"{EPOCHS}_{stem}_{mode}.h5"
        )
        model.save(str(model_file))
        return best_recall

    # 8) Study creation
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        storage=STORAGE_URL,
        load_if_exists=True
    )

    # 9) Optimize
    try:
        study.optimize(objective, n_trials=50)
    except KeyboardInterrupt:
        print("Optimization interrupted. State saved to DB.")

    # 10) Results
    print("=== Finished ===")
    print("Best trial #", study.best_trial.number)
    print("Best recall", study.best_value)
    print("Params", study.best_params)

    return {}
