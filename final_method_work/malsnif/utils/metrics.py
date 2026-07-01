from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from tqdm.auto import tqdm


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    balanced_accuracy: Optional[float] = None
    mcc: Optional[float] = None
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    average_precision: Optional[float] = None
    prevalence: Optional[float] = None
    specificity: Optional[float] = None
    predicted_positive_rate: Optional[float] = None
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    score_mean: Optional[float] = None
    score_std: Optional[float] = None
    best_f1_threshold: Optional[float] = None
    best_f1: Optional[float] = None
    best_f1_precision: Optional[float] = None
    best_f1_recall: Optional[float] = None
    positive_count: Optional[int] = None
    negative_count: Optional[int] = None
    predicted_positive: Optional[int] = None
    predicted_negative: Optional[int] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None

    def to_dict(self):
        return asdict(self)


def _safe_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    sorted_scores = y_score[order]
    i = 0
    rank = 1.0
    while i < len(order):
        j = i + 1
        while j < len(order) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        ranks[order[i:j]] = avg_rank
        rank += (j - i)
        i = j
    sum_pos_ranks = float(ranks[pos].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    ranks = np.arange(1, len(y) + 1)
    precision_at_k = tp / ranks
    return float((precision_at_k * (y == 1)).sum() / n_pos)


def _confusion(y_true: np.ndarray, y_score: np.ndarray, threshold: float):
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return y_pred, tp, fp, tn, fn


def _threshold_candidates(y_score: np.ndarray) -> np.ndarray:
    """Stable threshold candidates for validation tuning."""
    if y_score.size == 0:
        return np.asarray([0.5], dtype=float)
    uniq = np.unique(y_score.astype(float))
    eps = 1e-6
    cand = [float(uniq[0] - eps), float(uniq[-1] + eps), 0.5]
    cand.extend(float(x) for x in uniq)
    if uniq.size > 1:
        cand.extend(float((a + b) / 2.0) for a, b in zip(uniq[:-1], uniq[1:]))
    cand = [min(1.0, max(0.0, x)) for x in cand]
    return np.unique(np.asarray(cand, dtype=float))


def _rate_metrics(tp: int, fp: int, tn: int, fn: int):
    total = max(tp + fp + tn + fn, 1)
    acc = (tp + tn) / total
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    tnr = tn / max(tn + fp, 1)
    bal = 0.5 * (rec + tnr)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return acc, prec, rec, f1, bal, mcc


def _confusions_for_thresholds(y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray):
    """Vectorized confusion counts for predictions y_score >= threshold."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    n = y_true.size
    if n == 0:
        z = np.zeros_like(thresholds, dtype=np.int64)
        return z, z, z, z
    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_labels = y_true[order]
    k = np.searchsorted(-sorted_scores, -thresholds, side="right").astype(np.int64)
    prefix_pos = np.concatenate([[0], np.cumsum(sorted_labels == 1).astype(np.int64)])
    total_pos = int((y_true == 1).sum())
    total_neg = int(n - total_pos)
    tp = prefix_pos[k]
    fp = k - tp
    fn = total_pos - tp
    tn = total_neg - fp
    return tp.astype(np.int64), fp.astype(np.int64), tn.astype(np.int64), fn.astype(np.int64)


def _rates_from_arrays(tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray) -> dict[str, np.ndarray]:
    tp = tp.astype(float)
    fp = fp.astype(float)
    tn = tn.astype(float)
    fn = fn.astype(float)
    total = np.maximum(tp + fp + tn + fn, 1.0)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    specificity = tn / np.maximum(tn + fp, 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    balanced = 0.5 * (recall + specificity)
    denom = np.sqrt(np.maximum((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
    mcc = np.divide(tp * tn - fp * fn, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    accuracy = (tp + tn) / total
    predicted_positive_rate = (tp + fp) / total
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced,
        "mcc": mcc,
        "predicted_positive_rate": predicted_positive_rate,
    }


def _argmax_threshold(score: np.ndarray, thresholds: np.ndarray, rates: dict[str, np.ndarray]) -> int:
    score = np.asarray(score, dtype=float).copy()
    pred_pos_rate = rates["predicted_positive_rate"]
    collapse = (pred_pos_rate <= 0.0) | (pred_pos_rate >= 1.0)
    score[collapse] -= 1e-6
    # deterministic tie break; prefer higher threshold, then precision, then recall
    score = score + 1e-12 * thresholds + 1e-14 * rates["precision"] + 1e-16 * rates["recall"]
    return int(np.nanargmax(score))


def _best_f1(y_true: np.ndarray, y_score: np.ndarray):
    if y_true.size == 0:
        return None, None, None, None
    thresholds = _threshold_candidates(y_score)
    tp, fp, tn, fn = _confusions_for_thresholds(y_true, y_score, thresholds)
    rates = _rates_from_arrays(tp, fp, tn, fn)
    idx = _argmax_threshold(rates["f1"], thresholds, rates)
    return float(thresholds[idx]), float(rates["f1"][idx]), float(rates["precision"][idx]), float(rates["recall"][idx])


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> BinaryMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0:
        return BinaryMetrics(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)
    y_pred, tp, fp, tn, fn = _confusion(y_true, y_score, threshold)
    acc, prec, rec, f1, bal, mcc = _rate_metrics(tp, fp, tn, fn)
    best_thr, best_f1, best_prec, best_rec = _best_f1(y_true, y_score)
    auc_roc = _safe_auc_roc(y_true, y_score)
    auc_pr = _safe_average_precision(y_true, y_score)
    return BinaryMetrics(
        accuracy=float(acc), precision=float(prec), recall=float(rec), f1=float(f1),
        tp=tp, fp=fp, tn=tn, fn=fn,
        balanced_accuracy=float(bal), mcc=float(mcc),
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        average_precision=auc_pr,
        prevalence=float((y_true == 1).mean()),
        specificity=float(tn / max(tn + fp, 1)),
        predicted_positive_rate=float((y_pred == 1).mean()),
        score_min=float(np.min(y_score)), score_max=float(np.max(y_score)),
        score_mean=float(np.mean(y_score)), score_std=float(np.std(y_score)),
        best_f1_threshold=None if best_thr is None else float(best_thr),
        best_f1=None if best_f1 is None else float(best_f1),
        best_f1_precision=None if best_prec is None else float(best_prec),
        best_f1_recall=None if best_rec is None else float(best_rec),
        positive_count=int((y_true == 1).sum()),
        negative_count=int((y_true == 0).sum()),
        predicted_positive=int((y_pred == 1).sum()),
        predicted_negative=int((y_pred == 0).sum()),
        roc_auc=auc_roc,
        pr_auc=auc_pr,
    )


def metric_warnings(metrics: dict) -> list[str]:
    warnings: list[str] = []
    ppr = metrics.get("predicted_positive_rate")
    if ppr is not None:
        if float(ppr) >= 0.999:
            warnings.append("模型在当前阈值下几乎预测全为恶意；Precision/F1 可能主要反映正样本占比。")
        elif float(ppr) <= 0.001:
            warnings.append("模型在当前阈值下几乎预测全为良性；Recall/F1 可能无意义。")
    auc = metrics.get("auc_roc")
    if auc is not None and float(auc) < 0.5:
        warnings.append("ROC-AUC 低于 0.5，说明分数排序可能反向或模型尚未学习到有效判别。")
    bf1 = metrics.get("best_f1")
    f1 = metrics.get("f1")
    if bf1 is not None and f1 is not None and float(bf1) - float(f1) > 0.15:
        warnings.append("当前阈值下 F1 明显低于该 split 上的最优 F1；请检查验证集阈值是否过拟合单个窗口，必要时使用 threshold_strategy=precision_at_recall。")
    prev = metrics.get("prevalence")
    if prev is not None and (float(prev) > 0.8 or float(prev) < 0.02):
        warnings.append(f"评估样本正例占比为 {float(prev):.3f}，类别分布极端；请同时关注 AUC-PR、AUC-ROC、MCC 和混淆矩阵。")
    return warnings


def score_distribution(y_true, y_score) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    out: dict[str, object] = {"all": _summary(y_score)}
    out["positive"] = _summary(y_score[y_true == 1])
    out["negative"] = _summary(y_score[y_true == 0])
    return out


def _summary(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
    }


def precision_at_k(y_true, y_score, ks=(10, 20, 50, 100)) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0:
        return {}
    order = np.argsort(-y_score)
    out = {}
    for k in ks:
        kk = min(int(k), y_true.size)
        if kk <= 0:
            continue
        out[f"precision_at_{k}"] = float(y_true[order[:kk]].mean())
        out[f"positives_at_{k}"] = int(y_true[order[:kk]].sum())
    return out


def choose_threshold(y_true, y_score, metric="f1", min_recall: float = 0.95, show_progress: bool = False):
    """Choose an operating threshold in O(N log N), not O(N^2)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0:
        return 0.5, {"reason": "empty"}
    thresholds = _threshold_candidates(y_score)
    metric = str(metric or "f1").lower()
    pbar = None
    if show_progress and thresholds.size >= 5000:
        pbar = tqdm(total=3, desc=f"threshold tuning ({metric})", unit="step", dynamic_ncols=True)
        pbar.set_postfix_str("candidates")
        pbar.update(1)
    tp, fp, tn, fn = _confusions_for_thresholds(y_true, y_score, thresholds)
    if pbar is not None:
        pbar.set_postfix_str("confusions")
        pbar.update(1)
    rates = _rates_from_arrays(tp, fp, tn, fn)
    if metric == "precision_at_recall":
        score = np.where(rates["recall"] >= float(min_recall), rates["precision"], -1e9 + rates["recall"])
    elif metric.endswith("_min_recall"):
        base_metric = metric[: -len("_min_recall")] or "f1"
        key = {"balanced": "balanced_accuracy", "bal": "balanced_accuracy", "roc_auc": "f1", "auc_roc": "f1", "pr_auc": "f1", "auc_pr": "f1"}.get(base_metric, base_metric)
        base_score = rates.get(key, rates["f1"])
        score = np.where(rates["recall"] >= float(min_recall), base_score, -1e9 + rates["recall"])
    else:
        key = {"balanced": "balanced_accuracy", "bal": "balanced_accuracy", "roc_auc": "f1", "auc_roc": "f1", "pr_auc": "f1", "auc_pr": "f1"}.get(metric, metric)
        score = rates.get(key, rates["f1"])
    idx = _argmax_threshold(score, thresholds, rates)
    if pbar is not None:
        pbar.set_postfix_str("select")
        pbar.update(1)
        pbar.close()
    best_t = float(thresholds[idx])
    bm = {
        "accuracy": float(rates["accuracy"][idx]),
        "precision": float(rates["precision"][idx]),
        "recall": float(rates["recall"][idx]),
        "specificity": float(rates["specificity"][idx]),
        "f1": float(rates["f1"][idx]),
        "balanced_accuracy": float(rates["balanced_accuracy"][idx]),
        "mcc": float(rates["mcc"][idx]),
        "predicted_positive_rate": float(rates["predicted_positive_rate"][idx]),
        "tp": int(tp[idx]), "fp": int(fp[idx]), "tn": int(tn[idx]), "fn": int(fn[idx]),
        "positive_count": int((y_true == 1).sum()),
        "negative_count": int((y_true == 0).sum()),
        "predicted_positive": int(tp[idx] + fp[idx]),
        "predicted_negative": int(tn[idx] + fn[idx]),
    }
    return best_t, {"metric": metric, "threshold": best_t, "num_thresholds": int(thresholds.size), "num_threshold_candidates": int(thresholds.size), "metrics_at_threshold": bm}
