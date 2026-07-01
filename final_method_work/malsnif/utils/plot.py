from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Iterable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_metric_keys(metric_keys: str | Iterable[str] | None) -> list[str]:
    if metric_keys is None:
        return []
    if isinstance(metric_keys, str):
        return [x.strip() for x in metric_keys.split(",") if x.strip()]
    return [str(x).strip() for x in metric_keys if str(x).strip()]


def plot_history(
    history: List[Dict],
    out_path: str | Path,
    mode: str = "essential",
    metric_keys: str | Iterable[str] | None = None,
) -> None:
    """Write training-history plots.

    mode="essential" writes a single compact history.png only.  This is the
    default because the raw history.json already contains every scalar and the
    per-metric pngs made analysis bundles unnecessarily large/noisy.  mode="all"
    reproduces the old behavior and additionally writes individual metric pngs.
    mode="none" disables plot generation.
    """
    if not history:
        return
    mode = str(mode or "essential").lower()
    if mode in {"none", "off", "0", "false"}:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
    default_keys = ["loss", "val_f1", "val_mcc", "val_average_precision", "val_threshold"]
    compact_keys = _parse_metric_keys(metric_keys) or default_keys

    if mode in {"all", "full", "legacy"}:
        keys = [
            "loss", "val_loss", "accuracy", "balanced_accuracy", "precision",
            "recall", "specificity", "f1", "mcc", "roc_auc",
            "average_precision", "val_accuracy", "val_balanced_accuracy",
            "val_precision", "val_recall", "val_specificity", "val_f1",
            "val_mcc", "val_roc_auc", "val_average_precision", "val_threshold",
        ]
        for key in keys:
            vals = [h.get(key) for h in history]
            if any(v is not None for v in vals):
                plt.figure()
                plt.plot(epochs, vals)
                plt.xlabel("Epoch")
                plt.ylabel(key)
                plt.title(key)
                plt.tight_layout()
                plt.savefig(out_path.parent / f"{key}.png", dpi=140)
                plt.close()

    # One combined compact plot for report.  It is the only history image kept
    # in essential mode; complete numeric values remain in history.json.
    plt.figure(figsize=(9, 5))
    plotted = False
    for key in compact_keys:
        vals = [h.get(key) for h in history]
        if any(v is not None for v in vals):
            plt.plot(epochs, vals, label=key)
            plotted = True
    if not plotted:
        vals = [h.get("loss") for h in history]
        if any(v is not None for v in vals):
            plt.plot(epochs, vals, label="loss")
    plt.xlabel("Epoch")
    plt.title("MalSnif training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_score_histogram(labels, scores, out_path: str | Path, threshold: float | None = None, mode: str = "essential") -> None:
    import numpy as np
    mode = str(mode or "essential").lower()
    if mode in {"none", "off", "0", "false"}:
        return
    labels = np.asarray(labels).astype(int) if len(labels) else np.asarray([])
    scores = np.asarray(scores).astype(float) if len(scores) else np.asarray([])
    if scores.size == 0:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    if (labels == 0).any():
        plt.hist(scores[labels == 0], bins=30, alpha=0.6, label="label=0")
    if (labels == 1).any():
        plt.hist(scores[labels == 1], bins=30, alpha=0.6, label="label=1")
    if threshold is not None:
        try:
            plt.axvline(float(threshold), linestyle="--", linewidth=1, label=f"threshold={float(threshold):.4f}")
        except Exception:
            pass
    plt.xlabel("Predicted malicious score")
    plt.ylabel("Count")
    plt.title("Score distribution by label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
