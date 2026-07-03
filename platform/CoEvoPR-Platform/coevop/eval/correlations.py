"""Rank-correlation metrics without a SciPy dependency."""

from __future__ import annotations

import math

import numpy as np


def finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = finite_pair(x, y)
    if len(x) < 2:
        return math.nan
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom == 0:
        return math.nan
    return float(np.dot(x_centered, y_centered) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = finite_pair(x, y)
    if len(x) < 2:
        return math.nan
    return pearson(rankdata(x), rankdata(y))


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    x, y = finite_pair(x, y)
    n = len(x)
    if n < 2:
        return math.nan

    concordant = discordant = ties_x = ties_y = 0
    for i in range(n - 1):
        dx = x[i] - x[i + 1 :]
        dy = y[i] - y[i + 1 :]
        sign_x = np.sign(dx)
        sign_y = np.sign(dy)
        ties_x += int(np.sum(sign_x == 0))
        ties_y += int(np.sum(sign_y == 0))
        products = sign_x * sign_y
        concordant += int(np.sum(products > 0))
        discordant += int(np.sum(products < 0))

    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return math.nan
    return float((concordant - discordant) / denom)
