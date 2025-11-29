from __future__ import annotations
from typing import Iterable, List
import numpy as np


def precision_recall_at_k(
    recommended: Iterable[int],
    relevant: Iterable[int],
    k: int,
) -> tuple[float, float, float]:
    rec = list(recommended)[:k]
    rel = set(relevant)

    if len(rec) == 0:
        return 0.0, 0.0, 0.0

    hits = sum(1 for r in rec if r in rel)

    precision = hits / len(rec)
    recall = hits / max(len(rel), 1)
    hr = 1.0 if hits > 0 else 0.0  

    return precision, recall, hr


def ndcg_at_k(
    recommended: Iterable[int],
    relevant: Iterable[int],
    k: int,
) -> float:

    rec = list(recommended)[:k]
    rel = set(relevant)

    gains = []
    for rank, item in enumerate(rec, start=1):
        if item in rel:
            gains.append(1.0 / np.log2(rank + 1))
        else:
            gains.append(0.0)

    dcg = float(np.sum(gains))

    ideal_hits = min(len(rel), k)
    ideal_gains = [1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1)]
    idcg = float(np.sum(ideal_gains))

    if idcg == 0:
        return 0.0
    return dcg / idcg