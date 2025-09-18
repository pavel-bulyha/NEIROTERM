#!/usr/bin/env python3
import multiprocessing
import numpy as np
import tensorflow as tf
import optuna
import os

from pathlib import Path
from tensorflow.keras.metrics import Precision, Recall, MeanMetricWrapper # type: ignore
from tensorflow.keras.callbacks import CSVLogger # type: ignore
from tensorflow.keras import mixed_precision, backend as K # type: ignore
from tensorflow.data import AUTOTUNE # type: ignore

from utils import DataUtils, weighted_BCE, register_model
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned

# Force CPU mode to avoid cuDNN issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

# 2) Global constants & explicit Optuna storage
EPOCHS     = 50
BASE_DIR   = Path(__file__).parent.resolve()
DB_PATH    = BASE_DIR / "optuna_cnn.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
STORAGE_URL = f"sqlite:///{DB_PATH}"

print(f"Optuna storage: {DB_PATH}")

# 2a) Prepare paired dropout lists
drop1_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
drop_conv_values = np.linspace(0.0, 0.3, num=len(drop1_values)).tolist()
drop_dense_values = np.linspace(0.0, 0.5, num=len(drop1_values)).tolist()

# 2b) Custom CSVLogger with global epoch counter
class EpochCSVLogger(CSVLogger):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        self.global_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.global_epoch += 1
        logs = {} if logs is None else dict(logs)
        logs['epoch'] = self.global_epoch
        super().on_epoch_end(epoch, logs)

# 2c) Custom metric function to avoid AutoGraph issues
@tf.autograph.experimental.do_not_convert
def neg_recall_metric(yt, yp):
    yt_float = tf.cast(yt, tf.float32)
    yp_binary = tf.cast(yp > 0.5, tf.float32)
    one = tf.constant(1.0, dtype=tf.float32)
    return (
        tf.reduce_sum(tf.multiply(tf.subtract(one, yt_float), tf.subtract(one, yp_binary))) /
        (tf.reduce_sum(tf.subtract(one, yt_float)) + K.epsilon())
    )

@register_model("CNNOptuna")
def CNNOptunaCPU(data_csv: str, use_weighted_bce: bool):
    # 3) Load & preprocess
    data_file = Path(data_csv)
    if not data_file.exists():
        raise FileNotFoundError(data_file)
    proc = DataUtils(str(data_file))
    hp = proc.get_hyperparams()
    seq_len, vocab_size, seed = hp["seq_len"], hp["vocab_size"], hp["random_seed"]

    np.random.seed(seed)
    tf.random.set_seed(seed)
    X_train, X_val, y_train, y_val = proc.get_processed_data()

    # 4) Loss fn
    if use_weighted_bce:
        labels = y_train.numpy().flatten() # type: ignore
        neg, pos   = np.bincount(labels, minlength=2)
        pos_weight = neg / (pos + K.epsilon())
        loss_fn    = weighted_BCE(pos_weight)
    else:
        loss_fn = "binary_crossentropy"

    # 5) Hyper-space & dirs
    bs_list = [16, 32, 64, 128]

    logs_dir   = BASE_DIR / "logs"
    models_dir = BASE_DIR / "saved_models"
    logs_dir.mkdir(exist_ok=True, parents=True)
    models_dir.mkdir(exist_ok=True, parents=True)

    stem = data_file.stem
    mode = "weightedBCE" if use_weighted_bce else "BCE"

    # 6) Objective with per-epoch logging & model save
    def objective(trial: optuna.Trial) -> float:
        # pick index to pair dropout_conv & dropout_dense
        idx       = trial.suggest_int("drop_idx", 0, len(drop1_values) - 1)
        dropout_conv  = drop_conv_values[idx]
        dropout_dense  = drop_dense_values[idx]

        kernel_size  = trial.suggest_categorical("kernel_size", [10,20,30,40,50])
        stride  = trial.suggest_categorical("stride", [1,2,4,6,8,10])
        lr  = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        bs  = trial.suggest_categorical("bs", bs_list)
        output_length = (seq_len - kernel_size) // stride + 1

        # build datasets
        train_ds = (
            tf.data.Dataset
              .from_tensor_slices((X_train, y_train))
              .shuffle(10_000)
              .batch(bs)
              .prefetch(AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset
              .from_tensor_slices((X_val, y_val))
              .batch(bs)
              .prefetch(AUTOTUNE)
        )

        # optimizer + model
        optimizer = tf.keras.optimizers.Adam(lr)
        if gpus:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        model = tf.keras.Sequential([
            tf.keras.layers.Input((seq_len, vocab_size)),
            tf.keras.layers.Conv1D(filters=1, kernel_size=kernel_size, strides=stride, activation="relu", padding="valid"),
            tf.keras.layers.Dropout(dropout_conv),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=output_length, activation="relu"),
            tf.keras.layers.Dropout(dropout_dense),
            tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32"),
        ])
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                "accuracy",
                Precision(name="precision"),
                Recall(name="recall"),
                MeanMetricWrapper(neg_recall_metric, name="neg_recall")
            ]
        )

        # dropout strings for filenames
        dropconv_str = f"dropconv{int(dropout_conv*100)}"
        dropdense_str = f"dropdense{int(dropout_dense*100)}"
        lr_str    = f"{lr:.0e}"

        # log file and model folder names include kernel_size, stride, dropouts
        log_file = logs_dir / f"{stem}_{mode}_kernel{kernel_size}_stride{stride}_{dropconv_str}_{dropdense_str}_lr{lr_str}_bs{bs}.tsv"
        trial_folder = models_dir / f"{stem}_{mode}_kernel{kernel_size}_stride{stride}_{dropconv_str}_{dropdense_str}_lr{lr_str}_bs{bs}"

        logger = EpochCSVLogger(str(log_file), separator="\t", append=True)
        trial_folder.mkdir(exist_ok=True)

        best_rec = 0.0
        for epoch in range(1, EPOCHS + 1):
            hist = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=1,
                callbacks=[logger],
                verbose=0 # type: ignore
            )
            val_rec = hist.history["val_recall"][-1] # type: ignore
            best_rec = max(best_rec, val_rec)

            trial.report(val_rec, step=epoch)
            if trial.should_prune():
                raise TrialPruned()

            epoch_path = trial_folder / f"epoch_{epoch:02d}.h5"
            model.save(str(epoch_path))

        return best_rec

    # 7) Study creation & optimize
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        storage=STORAGE_URL,
        load_if_exists=True
    )
    try:
        study.optimize(objective, n_trials=50)
    except KeyboardInterrupt:
        print("Interrupted, progress saved.")

    print("Best trial:", study.best_trial.number)
    print("Best recall:", study.best_value)
    print("Best params:", study.best_params)
    return {}
