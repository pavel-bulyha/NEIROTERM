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
print(f"Detected CPU cores: {num_cpu_cores}")
print("Inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())
print("Intra-op threads:", tf.config.threading.get_intra_op_parallelism_threads())

EPOCHS = 50

@register_model("PerceptronOptunaCPU")
def PerceptronOptunaCPU(data_csv: str, use_weighted_bce: bool):
    """
    Hyperparameter tuning with Optuna + CPU/GPU optimization + tf.data pipeline.
    """
    data_file = Path(data_csv)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # 2) Load & preprocess
    proc = DataUtils(str(data_file))
    hp = proc.get_hyperparams()
    seq_len, vocab_size, seed = hp["seq_len"], hp["vocab_size"], hp["random_seed"]

    np.random.seed(seed)
    tf.random.set_seed(seed)
    X_train, X_val, y_train, y_val = proc.get_processed_data()

    # 3) Loss function
    if use_weighted_bce:
        labels = y_train.numpy().flatten()
        neg, pos = np.bincount(labels, minlength=2)
        pos_weight = neg / (pos + K.epsilon())
        loss_fn = weighted_BCE(pos_weight)
    else:
        loss_fn = "binary_crossentropy"

    # 4) Hyperparameter choices
    h1_list = [64, 128, 256, 512]
    h2_list = [32, 64, 128, 256]
    lr_list = np.logspace(-4, -2, num=5).tolist()
    bs_list = [16, 32, 64, 128]

    # 5) Prepare output dirs
    base_dir   = Path(__file__).parent
    logs_dir   = base_dir / "logs"
    models_dir = base_dir / "saved_models"
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    stem = data_file.stem
    mode = "weightedBCE" if use_weighted_bce else "BCE"

    def objective(trial):
        # 5.1) Suggest hyperparameters
        h1 = trial.suggest_categorical("h1", h1_list)
        h2 = trial.suggest_categorical("h2", h2_list)
        lr = trial.suggest_loguniform("lr", lr_list[0], lr_list[-1])
        bs = trial.suggest_categorical("bs", bs_list)

        # 5.2) Build tf.data pipelines
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(buffer_size=10_000)
            .batch(bs)
            .prefetch(AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(bs)
            .prefetch(AUTOTUNE)
        )

        # 5.3) Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if gpus:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        # 5.4) Model definition
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

        # 5.5) Log & model filenames
        lr_str     = f"{lr:.0e}"
        log_file   = f"TrainLOG_Perceptron_h1{h1}_h2{h2}_lr{lr_str}_bs{bs}_{stem}_{mode}.tsv"
        model_file = (
            f"NN_Perceptron_h1{h1}_h2{h2}_lr{lr_str}_bs{bs}_"
            f"{EPOCHS}_{stem}_{mode}.h5"
        )
        tsv_logger = CSVLogger(str(logs_dir / log_file), separator="\t", append=False)

        # 5.6) Train
        print(f"Trial {trial.number}: h1={h1}, h2={h2}, lr={lr_str}, bs={bs}")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=[tsv_logger],
            verbose=0
        )

        # 5.7) Save & evaluate
        model.save(str(models_dir / model_file))
        results = model.evaluate(val_ds, verbose=0, return_dict=True)
        return results["recall"]

    # 6) Run Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("=== Optuna finished ===")
    print("Best trial:", study.best_trial.number)
    print("Best recall:", study.best_value)
    print("Best params:", study.best_params)

    return {}
