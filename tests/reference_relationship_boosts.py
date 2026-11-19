"""Verbatim copy of the pre-Batch-2 `_relationship_boosts` (pandas version), kept only so tests can prove the faster numpy
version gives identical roles, scores and evidence. Do not edit it; do not import it from application code."""
from itertools import combinations

import pandas as pd
from pandas.api import types as pdt

from cleaning.type_inference import to_numeric_clean
from semantic.column_classifier import NUMERIC_ROLES, _Feat


def reference_relationship_boosts(df: pd.DataFrame, feats: dict[str, _Feat],
                         scores: dict[str, dict[str, tuple[float, list[str]]]]) -> None:
    """Detect a ~ b x c (revenue = quantity x price) and a ~ b - c (profit = revenue - cost)."""
    cand = [n for n, f in feats.items() if f.kind == "numeric" and f.vals is not None and f.vals.std() > 0]
    cand.sort(key=lambda n: -max(scores[n][r][0] for r in NUMERIC_ROLES))
    cand = cand[:16]
    if len(cand) < 3:
        return
    idx = df.sample(min(len(df), 2000), random_state=0).index if len(df) > 2000 else df.index
    cols = {}
    for n in cand:
        s = df.loc[idx, n]
        cols[n] = to_numeric_clean(s) if not pdt.is_numeric_dtype(s) else s.astype(float)
    num = pd.DataFrame(cols)

    def boost(col: str, role: str, amount: float, text: str) -> None:
        sc, ev = scores[col][role]
        scores[col][role] = (min(1.0, sc + amount), ev + [text])

    def close(a: pd.Series, p: pd.Series) -> float | None:
        ok = a.notna() & p.notna()
        if ok.sum() < 20:
            return None
        diff = (a[ok] - p[ok]).abs() <= 0.01 * a[ok].abs() + 1e-9
        return float(diff.mean())

    done: set[str] = set()
    for b, c in combinations(cand, 2):
        prod = num[b] * num[c]
        for a in cand:
            if a in (b, c) or a in done:
                continue
            r = close(num[a], prod)
            if r is not None and r >= 0.97:
                qty, pr = (b, c) if (feats[b].is_int, -num[b].mean()) >= (feats[c].is_int, -num[c].mean()) else (c, b)
                txt = f"Relationship: '{a}' = '{b}' x '{c}' in {r:.0%} of sampled rows"
                boost(a, "revenue", 0.3, txt)
                boost(qty, "quantity", 0.25, txt)
                boost(pr, "price", 0.25, txt)
                done.add(a)
    for b, c in combinations(cand, 2):
        for first, second in ((b, c), (c, b)):
            if num[first].mean() <= num[second].mean() or num[second].mean() <= 0:
                continue
            diff = num[first] - num[second]
            plausible = scores[first]["revenue"][0] >= 0.3 and scores[second]["cost"][0] >= 0.3
            for a in cand:
                if a in (first, second) or a in done:
                    continue
                if not (plausible or scores[a]["profit"][0] >= 0.3):   # a-b-c alone is ambiguous
                    continue
                r = close(num[a], diff)
                if r is not None and r >= 0.97:
                    txt = f"Relationship: '{a}' = '{first}' - '{second}' in {r:.0%} of sampled rows"
                    boost(a, "profit", 0.3, txt)
                    boost(first, "revenue", 0.2, txt)
                    boost(second, "cost", 0.2, txt)
                    done.add(a)
