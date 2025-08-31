import multiprocessing
import numpy as np
import tensorflow as tf
import optuna

from pathlib import Path
from tensorflow.keras.metrics import Precision, Recall, MeanMetricWrapper # type: ignore
from tensorflow.keras.callbacks import CSVLogger # type: ignore
from tensorflow.keras import mixed_precision, backend as K # type: ignore
from tensorflow.data import AUTOTUNE # type: ignore

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

# 2) Global constants & explicit Optuna storage
EPOCHS    = 50
BASE_DIR  = Path(__file__).parent.resolve()
DB_PATH   = BASE_DIR / "optuna_perceptron.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
STORAGE_URL = f"sqlite:///{DB_PATH}"

print(f"Optuna storage: {DB_PATH}")

@register_model("PerceptronOptuna")
def PerceptronOptunaCPU(data_csv: str, use_weighted_bce: bool):
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
        neg, pos = np.bincount(labels, minlength=2)
        pos_weight = neg / (pos + K.epsilon())
        loss_fn = weighted_BCE(pos_weight)
    else:
        loss_fn = "binary_crossentropy"

    # 5) Hyper-space & dirs
    h1_list = [64, 128, 256, 512]
    h2_list = [32, 64, 128, 256]
    lr_list = np.logspace(-4, -2, 5).tolist()
    bs_list = [16, 32, 64, 128]

    logs_dir   = BASE_DIR / "logs"
    models_dir = BASE_DIR / "saved_models"
    logs_dir.mkdir(exist_ok=True, parents=True)
    models_dir.mkdir(exist_ok=True, parents=True)

    stem = data_file.stem
    mode = "weightedBCE" if use_weighted_bce else "BCE"

    # 6) Objective with per-epoch logging & model save
    def objective(trial: optuna.Trial) -> float:
        h1 = trial.suggest_categorical("h1", h1_list)
        h2 = trial.suggest_categorical("h2", h2_list)
        lr = trial.suggest_loguniform("lr", lr_list[0], lr_list[-1])
        bs = trial.suggest_categorical("bs", bs_list)

        # build datasets
        train_ds = (
            tf.data.Dataset
              .from_tensor_slices((X_train, y_train))
              .shuffle(10_000).batch(bs).prefetch(AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset
              .from_tensor_slices((X_val, y_val))
              .batch(bs).prefetch(AUTOTUNE)
        )

        # optimizer + model
        optimizer = tf.keras.optimizers.Adam(lr)
        if gpus:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        model = tf.keras.Sequential([
            tf.keras.layers.Input((seq_len, vocab_size)),
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
                    lambda yt, yp: (
                        tf.reduce_sum((1 - tf.cast(yt, tf.float32)) * # type: ignore
                                      (1 - tf.cast(yp > 0.5, tf.float32))) # type: ignore
                        / (tf.reduce_sum(1 - tf.cast(yt, tf.float32)) + K.epsilon()) # type: ignore
                    ),
                    name="neg_recall"
                )
            ]
        )

        # CSVLogger: append=True to save all epochs in sequence
        lr_str = f"{lr:.0e}"
        log_file = logs_dir / f"{stem}_{mode}_h1{h1}_h2{h2}_lr{lr_str}_bs{bs}.tsv"
        logger   = CSVLogger(str(log_file), separator="\t", append=True)

        best_rec = 0.0
        # for each epoch: log, prune, and save the model
        trial_folder = models_dir / f"{stem}_{mode}_h1{h1}_h2{h2}_lr{lr_str}_bs{bs}"
        trial_folder.mkdir(exist_ok=True)
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

            # report + prune
            trial.report(val_rec, step=epoch)
            if trial.should_prune():
                raise TrialPruned()

            # save model this epoch
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
