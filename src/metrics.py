"""Evaluation metrics for probabilistic game forecasts."""
import numpy as np
import pandas as pd


def log_loss(y_true, p_pred, eps=1e-15):
    p = np.clip(np.asarray(p_pred, dtype=float), eps, 1 - eps)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y_true, p_pred):
    return float(np.mean((np.asarray(p_pred, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def accuracy(y_true, p_pred):
    return float(np.mean((np.asarray(p_pred) > 0.5).astype(int) == np.asarray(y_true)))


def summarize(y_true, p_pred, label=""):
    return {
        "model": label,
        "n": len(y_true),
        "log_loss": round(log_loss(y_true, p_pred), 5),
        "brier": round(brier(y_true, p_pred), 5),
        "accuracy": round(accuracy(y_true, p_pred), 4),
    }


def calibration_table(y_true, p_pred, bins=10):
    """Predicted vs actual rate by probability bucket."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p_pred, dtype=float)})
    df["bucket"] = pd.cut(df["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    out = df.groupby("bucket", observed=True).agg(
        n=("y", "size"),
        predicted=("p", "mean"),
        actual=("y", "mean"),
    ).round(4)
    out["gap"] = (out["actual"] - out["predicted"]).round(4)
    return out