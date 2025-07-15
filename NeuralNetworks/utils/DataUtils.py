# utils/DataUtils.py

"""
This module contains the DataUtils class, which:
  – reads CSV files with nucleotide sequences
  – calculates hyperparameters (seq_len, input_dim, etc.)
  – splits data into train/val
  – encodes one-hot and returns tf.Tensor for training
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict

class DataUtils:
    """
    Encapsulates the entire pipeline:
      – reading CSV
      – calculating dimensions
      – stratified split
      – one-hot encoding
      – packing into tf.Tensor
    
    Public methods:
      • get_hyperparams()      – returns a dictionary of hyperparameters
      • get_processed_data()   – returns X_train, X_val, y_train, y_val
    """

    def __init__(
        self,
        csv_path: str,
        val_split: float = 0.2,
        random_seed: int = 42
    ):
        # Run parameters
        self._csv_path = csv_path
        self._val_split = val_split
        self._random_seed = random_seed

        # Constant attributes
        self._char2idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        self._vocab_size = len(self._char2idx)

        # Will be filled in _init_dims()
        self._seq_len = 0
        self._input_dim = 0

        # Immediately calculate seq_len and input_dim
        self._init_dims()

    def _init_dims(self) -> None:
        """
        Reads the entire 'sequence' column, checks that
        all strings have the same length, and sets self._seq_len and self._input_dim.
        """
        df = pd.read_csv(self._csv_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        seqs = df['sequence'].astype(str).tolist()

        lengths = {len(s) for s in seqs}
        if len(lengths) != 1:
            raise ValueError(
                f"All sequences must have the same length, but found: {sorted(lengths)}"
            )

        self._seq_len = lengths.pop()
        self._input_dim = self._seq_len * self._vocab_size

    def get_hyperparams(self) -> Dict[str, float]:
        """
        Returns all key hyperparameters for building and training models:
          – seq_len       : sequence length
          – input_dim     : seq_len * vocab_size
          – vocab_size    : size of one-hot vector (number of channels)
          – val_split     : validation split ratio
          – random_seed   : seed for reproducible splitting
        """
        return {
            "seq_len":     self._seq_len,
            "input_dim":   self._input_dim,
            "vocab_size":  self._vocab_size,
            "val_split":   self._val_split,
            "random_seed": self._random_seed,
        }

    def get_processed_data(
        self
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Returns tensors:
          X_train (float32): (N_train, seq_len, vocab_size)
          X_val   (float32): (N_val,   seq_len, vocab_size)
          y_train (int32)  : (N_train, 1)
          y_val   (int32)  : (N_val,   1)
        """
        # 1) Read raw sequences and labels
        seqs, labels = self._load_raw()

        # 2) Stratified split
        seqs_tr, seqs_va, y_tr_np, y_va_np = train_test_split(
            seqs,
            labels,
            test_size=self._val_split,
            random_state=self._random_seed,
            shuffle=True,
            stratify=labels
        )

        # 3) One-hot encoding
        X_train = self._onehot_encode(seqs_tr)
        X_val   = self._onehot_encode(seqs_va)

        # 4) Labels to tf.Tensor
        y_train = tf.convert_to_tensor(y_tr_np.reshape(-1, 1), dtype=tf.int32)
        y_val   = tf.convert_to_tensor(y_va_np.reshape(-1, 1), dtype=tf.int32)

        return X_train, X_val, y_train, y_val

    def _load_raw(self) -> Tuple[List[str], np.ndarray]:
        """
        Reads CSV, returns:
          – list of sequences seqs
          – numpy array of labels
        """
        df = pd.read_csv(self._csv_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        seqs = df['sequence'].astype(str).tolist()
        labels = df['class'].to_numpy(dtype=np.int32)
        return seqs, labels

    def _onehot_encode(self, seqs: List[str]) -> tf.Tensor:
        """
        Converts a list of seqs to a one-hot tensor
        of shape (len(seqs), seq_len, vocab_size).
        """
        N = len(seqs)
        idx_arr = np.zeros((N, self._seq_len), dtype=np.int32)

        for i, s in enumerate(seqs):
            padded = s[:self._seq_len].ljust(self._seq_len, 'A')
            idx_arr[i] = [self._char2idx.get(ch, 0) for ch in padded]

        return tf.one_hot(idx_arr, depth=self._vocab_size, dtype=tf.float32)
