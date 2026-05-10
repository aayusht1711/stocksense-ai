"""
models/lstm_model.py
─────────────────────────────────────────────────────────────
Bidirectional LSTM with attention for stock price prediction.
Supports both regression (price) and classification (direction).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, classification_report
)
import joblib


MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models/saved"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Attention Layer ───────────────────────────────────────────────────────────

class AttentionLayer(layers.Layer):
    """Additive (Bahdanau-style) self-attention over the LSTM sequence."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attn_W", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="attn_b", shape=(1,),
                                 initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        # x: (batch, timesteps, features)
        e = tf.squeeze(tf.tanh(tf.matmul(x, self.W) + self.b), axis=-1)  # (batch, timesteps)
        alpha = tf.nn.softmax(e, axis=-1)                                   # (batch, timesteps)
        context = tf.reduce_sum(x * tf.expand_dims(alpha, -1), axis=1)     # (batch, features)
        return context, alpha


# ── Model Builder ─────────────────────────────────────────────────────────────

def build_lstm_model(
    seq_len: int,
    n_features: int,
    task: str = "regression",        # "regression" | "classification"
    lstm_units: list[int] = None,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build a Bidirectional LSTM with attention.

    Args:
        seq_len:       Number of timesteps (e.g. 60 days)
        n_features:    Number of features per timestep
        task:          'regression' → predict price/return; 'classification' → predict direction
        lstm_units:    Hidden units per LSTM layer (default [128, 64])
        dropout_rate:  Dropout between LSTM layers
        l2_reg:        L2 regularisation strength
        learning_rate: Adam learning rate

    Returns:
        Compiled Keras model
    """
    if lstm_units is None:
        lstm_units = [128, 64]

    reg = regularizers.l2(l2_reg)
    inp = keras.Input(shape=(seq_len, n_features), name="input")
    x   = inp

    # ── LSTM Layers ──────────────────────────────────────────
    for i, units in enumerate(lstm_units):
        return_seq = True   # always True until last LSTM (attention needs sequences)
        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=return_seq,
                        kernel_regularizer=reg, recurrent_regularizer=reg),
            name=f"bidir_lstm_{i+1}"
        )(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

    # ── Attention ────────────────────────────────────────────
    context, _ = AttentionLayer(name="attention")(x)

    # ── Dense Head ───────────────────────────────────────────
    x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(context)
    x = layers.Dropout(dropout_rate / 2)(x)
    x = layers.Dense(32, activation="relu")(x)

    if task == "regression":
        output = layers.Dense(1, name="price_output")(x)
        loss   = "huber"
        metric = ["mae"]
    else:
        output = layers.Dense(1, activation="sigmoid", name="direction_output")(x)
        loss   = "binary_crossentropy"
        metric = ["accuracy"]

    model = keras.Model(inputs=inp, outputs=output, name=f"StockSense_LSTM_{task}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=loss,
        metrics=metric,
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_lstm(
    X_train: np.ndarray,       # (samples, seq_len, features)
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    task:    str = "regression",
    ticker:  str = "AAPL",
    epochs:  int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    lstm_units: list[int] = None,
) -> tuple[keras.Model, dict]:
    """
    Train the LSTM and return (model, history_dict).
    Applies early stopping, ReduceLROnPlateau, and checkpoint saving.
    """
    seq_len   = X_train.shape[1]
    n_features= X_train.shape[2]

    model = build_lstm_model(
        seq_len=seq_len,
        n_features=n_features,
        task=task,
        lstm_units=lstm_units,
        learning_rate=learning_rate,
    )
    model.summary(print_fn=logger.info)

    ckpt_path = MODEL_DIR / f"lstm_{ticker}_{task}_best.keras"

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            str(ckpt_path), save_best_only=True, monitor="val_loss", verbose=0
        ),
        callbacks.TensorBoard(
            log_dir=str(MODEL_DIR / f"logs/{ticker}_{task}"), histogram_freq=0
        ),
    ]

    logger.info(f"Training LSTM {task} | {X_train.shape} → {y_train.shape}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1,
    )

    logger.info(f"Best val_loss: {min(history.history['val_loss']):.4f}")
    return model, history.history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_lstm(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task:   str = "regression",
    price_scaler=None,          # pass fitted scaler to invert-transform predictions
) -> dict:
    """Compute and return evaluation metrics."""
    y_pred = model.predict(X_test, verbose=0).flatten()

    if task == "regression":
        if price_scaler is not None:
            # Inverse-transform if prices were scaled
            y_test = price_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred = price_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2   = 1 - ss_res / (ss_tot + 1e-9)

        # Directional accuracy
        dir_actual = np.sign(np.diff(y_test))
        dir_pred   = np.sign(np.diff(y_pred))
        dir_acc    = np.mean(dir_actual == dir_pred)

        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2,
                   "Directional_Accuracy": dir_acc}

    else:
        y_bin = (y_pred > 0.5).astype(int)
        acc   = accuracy_score(y_test, y_bin)
        report= classification_report(y_test, y_bin, output_dict=True)
        metrics = {"Accuracy": acc, "Report": report,
                   "Precision": report["1"]["precision"],
                   "Recall":    report["1"]["recall"],
                   "F1":        report["1"]["f1-score"]}

    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
    return metrics


# ── Forecast ──────────────────────────────────────────────────────────────────

def forecast_next_n(
    model: keras.Model,
    last_sequence: np.ndarray,   # shape (seq_len, n_features)
    n_steps: int = 7,
    feature_idx_close: int = 0,  # column index for 'Close' in the sequence
) -> np.ndarray:
    """
    Auto-regressive multi-step forecast.
    Uses the model's own predictions as input for subsequent steps.
    Returns array of shape (n_steps,) with predicted price changes.
    """
    seq = last_sequence.copy()
    predictions = []

    for _ in range(n_steps):
        x_in = seq[np.newaxis, :, :]         # (1, seq_len, features)
        pred = model.predict(x_in, verbose=0)[0, 0]
        predictions.append(pred)
        # Roll sequence forward: shift by 1, insert new prediction in close column
        new_row = seq[-1].copy()
        new_row[feature_idx_close] = pred
        seq = np.vstack([seq[1:], new_row])

    return np.array(predictions)


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_lstm(model: keras.Model, ticker: str, task: str = "regression"):
    path = MODEL_DIR / f"lstm_{ticker}_{task}_final.keras"
    model.save(str(path))
    logger.info(f"LSTM saved → {path}")


def load_lstm(ticker: str, task: str = "regression") -> keras.Model:
    path = MODEL_DIR / f"lstm_{ticker}_{task}_final.keras"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Train first.")
    model = keras.models.load_model(str(path), custom_objects={"AttentionLayer": AttentionLayer})
    logger.info(f"LSTM loaded ← {path}")
    return model


# ── CLI quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Smoke-test with random data
    SEQ_LEN, N_FEAT, N_SAMPLES = 60, 40, 500

    X  = np.random.randn(N_SAMPLES, SEQ_LEN, N_FEAT).astype(np.float32)
    y_reg = np.random.randn(N_SAMPLES).astype(np.float32)
    y_cls = (np.random.rand(N_SAMPLES) > 0.5).astype(np.float32)

    split = int(0.8 * N_SAMPLES)

    print("\n── Regression LSTM ──")
    m, _ = train_lstm(X[:split], y_reg[:split], X[split:], y_reg[split:],
                      task="regression", ticker="TEST", epochs=3)
    metrics = evaluate_lstm(m, X[split:], y_reg[split:], task="regression")

    print("\n── Classification LSTM ──")
    m2, _ = train_lstm(X[:split], y_cls[:split], X[split:], y_cls[split:],
                       task="classification", ticker="TEST", epochs=3)
    metrics2 = evaluate_lstm(m2, X[split:], y_cls[split:], task="classification")
