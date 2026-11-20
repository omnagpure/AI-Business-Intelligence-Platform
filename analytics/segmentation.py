"""Segmentation (Phase 6C): KMeans on entities such as suppliers, customers or products.

Rows are first aggregated to one row per entity (identifiers are never used as features), missing values
are handled explicitly and reported, features are scaled, K is chosen with silhouette (elbow inertia is
reported too) and every step uses a fixed seed. Clusters are a convenient grouping, not objective truth.
Pure pandas/NumPy/scikit-learn; no Streamlit. Inputs are never modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

from analytics.data_checks import parse_flag
from analytics.kpis import REORDER_TOKENS, STOCK_TOKENS, UNIT_COST_TOKENS, DataFilter, _Ctx, _order_id, _revenue, _units
from analytics.supply_chain_kpis import (DEFECT_WORDS, _closed_rows, _cols, _on_time, _qty_pair, _status, lead_days,
                                          lead_label)
from config.analytics_config import DEFAULT_ANALYTICS, AnalyticsConfig
from core.exceptions import InsufficientDataError, SegmentationError
from core.progress import ProgressFn, notify
from semantic.semantic_model import SemanticModel

ENTITY_ROLES = ("supplier", "customer", "product", "warehouse")
NOTE = ("Clusters are a statistical grouping of similar entities in the chosen features. They are not objectively "
        "correct, and a different feature set or K can give a different grouping.")
SCALERS = {"robust": "Robust (median and IQR; less affected by outliers)", "standard": "Standard (mean and standard deviation)"}
TRAIT_Z = 0.5          # a cluster differs from the overall average by at least this many standard deviations
TRAIT_MIN_REL = 0.05   # ... and by at least 5% of the overall value (2 points for rates), so tiny gaps are not traits
TRAIT_MIN_PTS = 2.0


@dataclass(frozen=True)
class FeatureTable:
    entity: str
    frame: pd.DataFrame                 # one row per entity, one column per feature (raw values, may hold NaN)
    labels: dict[str, str]              # feature -> readable name
    units: dict[str, str]               # feature -> percent | number | amount
    excluded_features: dict[str, str]
    excluded_entities: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SegmentationResult:
    assignments: pd.DataFrame           # entity, cluster, then the raw feature values
    profile: pd.DataFrame               # one row per cluster: entities, share, mean of each feature
    overall: pd.Series                  # overall mean of each feature
    traits: dict[int, str]              # cluster -> plain-words description
    scores: pd.DataFrame                # k, inertia, silhouette, min_share
    k: int
    k_mode: str                         # "automatic" or "chosen"
    silhouette: float
    quality: str
    features: tuple[str, ...]
    labels: dict[str, str]
    units: dict[str, str]
    scaler: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    note: str = NOTE
    names: dict[int, str] = field(default_factory=dict)          # cluster -> short name ("Lower on-time rate")
    separation: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))   # feature -> share of its variance
                                                                                     # explained by the clusters (0-1)


# ---- which columns can be entities ------------------------------------------------------------
def entity_candidates(df: pd.DataFrame, model: SemanticModel, config: AnalyticsConfig = DEFAULT_ANALYTICS) -> list[dict]:
    """Supplier, customer, product or warehouse columns with enough entities and enough rows per entity."""
    out = []
    for role in ENTITY_ROLES:
        for col in model.columns_with_role(role):
            name = col if isinstance(col, str) else getattr(col, "name", str(col))
            if name not in df.columns:
                continue
            key = df[name].dropna()
            n = int(key.nunique())
            if n < config.seg_min_entities or len(key) == 0:
                continue
            avg = len(key) / n
            if avg >= config.seg_min_rows:
                out.append({"column": name, "role": role, "entities": n, "avg_rows": round(avg, 1)})
    out = sorted(out, key=lambda d: (ENTITY_ROLES.index(d["role"]), -d["entities"]))
    return _distinct_groupings(df, out)


def _distinct_groupings(df: pd.DataFrame, cands: list[dict]) -> list[dict]:
    """Keep one column per grouping: 'Customer Id' and 'Order Customer Id', or 'Product Card Id' and 'Product Name',
    group the rows the same way (one-to-one values), so offering both would only repeat the result. A name (text)
    is kept over a numeric id because it reads better."""
    kept: list[dict] = []
    for c in cands:
        twin = None
        for k in kept:
            if k["entities"] != c["entities"]:
                continue
            pair = df[[k["column"], c["column"]]].dropna()
            if len(pair.drop_duplicates()) == k["entities"]:
                twin = k
                break
        if twin is None:
            kept.append(dict(c))
            continue
        twin.setdefault("same_as", []).append(c["column"])
        if pd.api.types.is_numeric_dtype(df[twin["column"]]) and not pd.api.types.is_numeric_dtype(df[c["column"]]):
            twin["same_as"] = [twin["column"]] + [x for x in twin["same_as"] if x != c["column"]]
            twin["column"] = c["column"]
    return kept


# ---- entity-level features --------------------------------------------------------------------
def _grouped(key: pd.Series, values: pd.Series, how: str = "mean") -> pd.Series:
    f = pd.DataFrame({"k": key, "v": values}).dropna()
    return getattr(f.groupby("k")["v"], how)() if not f.empty else pd.Series(dtype=float)


def _stockout_flag(ctx: _Ctx):
    for name in ctx.model.columns:
        t = ctx.tokens(name)
        if name in ctx.df.columns and ("stockout" in t or "oos" in t or {"stock", "out"} <= t):
            f, _ = parse_flag(ctx.df[name])
            if f is not None:
                return name, f
    return None, None


def build_entity_features(df: pd.DataFrame, model: SemanticModel, entity: str,
                          config: AnalyticsConfig = DEFAULT_ANALYTICS) -> FeatureTable:
    """One row per entity. Supply-chain measures come first (on-time rate, delay, lead time, fill rate, defect rate,
    stockout rate), then value and volume, then other measure columns by role. Never uses identifiers as features."""
    if entity not in df.columns:
        raise SegmentationError("The chosen entity column is not in the data.")
    ctx = _Ctx(df, model, config, DataFilter())
    key = df[entity].astype("string")
    feats: dict[str, pd.Series] = {}
    labels: dict[str, str] = {}
    units: dict[str, str] = {}

    def add(k: str, label: str, unit: str, s: pd.Series) -> None:
        if len(s):
            feats[k], labels[k], units[k] = s, label, unit

    row_n = key.dropna().value_counts().astype(float)
    oid = _order_id(ctx)
    if oid is not None and oid != entity and df[oid].nunique() < len(df):      # several rows per order: count orders
        add("rows", "Orders", "count", _grouped(key, df[oid], "nunique").astype(float))
    else:
        add("rows", "Rows", "count", row_n)
    on, delay, _, _, _, _ = _on_time(ctx)
    if on is not None:
        add("on_time_rate", "On-time rate", "percent", 100 * _grouped(key, on.astype("Float64").astype(float)))
        if delay is not None:
            add("avg_delay", "Average delay vs expected", "days", _grouped(key, delay))
    d, dcols_lead, _ = lead_days(ctx)
    if d is not None and len(d):
        add("lead_time", f"Average {lead_label(dcols_lead)}", "days", _grouped(key.reindex(d.index), d))
    ordered, received = _qty_pair(ctx)
    if ordered is not None:
        mask, _ = _closed_rows(ctx, ordered, received)
        o, r = ctx.num(ordered)[mask], ctx.num(received)[mask]
        f = pd.DataFrame({"k": key[mask], "o": o, "c": pd.concat([o, r], axis=1).min(axis=1)}).dropna().groupby("k").sum()
        f = f[f["o"] > 0]
        add("fill_rate", "Fill rate", "percent", 100 * f["c"] / f["o"])
    dcols = _cols(ctx, ("quantity",), lambda t: bool(t & DEFECT_WORDS), True)
    u, _ = _units(ctx)
    if dcols and u is not None:
        f = pd.DataFrame({"k": key, "d": ctx.num(dcols[0]), "u": ctx.num(u)}).dropna().groupby("k").sum()
        f = f[f["u"] > 0]
        add("defect_rate", "Defect rate", "percent", 100 * f["d"] / f["u"])
    sname, sflag = _stockout_flag(ctx)
    if sflag is not None:
        add("stockout_rate", "Stockout rate", "percent", 100 * _grouped(key, sflag.astype("Float64").astype(float)))
    used = set(dcols[:1] + ([u] if u else []) + ([ordered, received] if ordered else []))
    rev, _ = _revenue(ctx)
    if rev is not None:
        cancelled, _, _ = _status(ctx)
        keep = ~cancelled if cancelled is not None else pd.Series(True, index=df.index)
        add("total_value", f"Total {rev}", "amount", _grouped(key[keep], ctx.num(rev)[keep], "sum"))
        add("avg_value", f"Average {rev} per row", "amount", _grouped(key[keep], ctx.num(rev)[keep]))
        used.add(rev)
    if u is not None:
        add("units", f"Total {u}", "number", _grouped(key, ctx.num(u), "sum"))
    for role in ("quantity", "cost", "profit", "price", "duration", "percentage"):
        for c in ctx.rank(role):
            if c in used or c == entity:
                continue
            used.add(c)
            t = ctx.tokens(c)
            # levels (stock, reorder point), unit costs and prices are averaged; amounts and counts are summed
            how = "mean" if (role in ("price", "duration", "percentage") or t & (STOCK_TOKENS | REORDER_TOKENS)
                             or (role == "cost" and t & UNIT_COST_TOKENS)) else "sum"
            unit = "days" if role == "duration" else "amount" if role in ("cost", "profit", "price") else "number"
            add(f"{how}_{c}", f"{'Total' if how == 'sum' else 'Average'} {c}", unit, _grouped(key, ctx.num(c), how))
            break
    frame = pd.DataFrame(feats)
    frame.index.name = entity
    rows = row_n.reindex(frame.index).fillna(0)
    excluded_entities = int((rows < config.seg_min_rows).sum())
    frame = frame[rows >= config.seg_min_rows]
    warnings: list[str] = []
    excluded: dict[str, str] = {}
    if excluded_entities:
        warnings.append(f"{excluded_entities:,} {entity} value(s) with fewer than {config.seg_min_rows} rows were left out "
                        "because their averages would be unreliable.")
    for c in list(frame.columns):                                     # missing, constant, duplicate features
        miss = float(frame[c].isna().mean())
        if len(frame) and miss > config.seg_max_missing:
            excluded[c] = f"missing for {miss:.0%} of entities"
        elif frame[c].nunique(dropna=True) < 2:
            excluded[c] = "same value for every entity"
    frame = frame.drop(columns=list(excluded))
    cols = list(frame.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a in frame and b in frame and abs(frame[a].corr(frame[b])) >= config.seg_dup_corr:
                excluded[b] = f"duplicates '{labels[a]}' (correlation {frame[a].corr(frame[b]):.2f})"
                frame = frame.drop(columns=[b])
    if len(frame.columns) > config.seg_max_features:
        for c in list(frame.columns)[config.seg_max_features:]:
            excluded[c] = f"more than {config.seg_max_features} features; later ones were left out"
        frame = frame.iloc[:, :config.seg_max_features]
    return FeatureTable(entity, frame, {k: labels[k] for k in frame.columns}, {k: units[k] for k in frame.columns},
                        excluded, excluded_entities, tuple(warnings))


# ---- clustering -------------------------------------------------------------------------------
def _fmt(v: float, unit: str) -> str:
    if unit == "percent":
        return f"{v:.1f}%"
    if unit == "days":
        return f"{v:,.1f} days"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}".rstrip("0").rstrip(".")


def _meaningful(diff: float, overall: float, unit: str) -> bool:
    """A gap worth calling a trait: 2+ points for a rate, else 5%+ of the overall value (or of 1 day for days)."""
    if unit == "percent":
        return abs(diff) >= TRAIT_MIN_PTS
    base = max(abs(overall), 1.0) if unit == "days" else abs(overall)
    return base == 0 or abs(diff) >= TRAIT_MIN_REL * base


def quality_label(sil: float) -> str:
    if sil >= 0.5:
        return "Clear structure: clusters are well separated."
    if sil >= 0.25:
        return "Moderate structure: clusters overlap somewhat."
    return "Weak structure: the groups overlap a lot, so treat this as a convenient grouping only."


def _trait_z(prof: pd.DataFrame, overall: pd.Series, spread: pd.Series, units) -> dict[int, pd.Series]:
    """Per cluster: the features where it differs clearly and meaningfully from the overall average, strongest first."""
    out = {}
    for c, row in prof.iterrows():
        z = ((row[overall.index] - overall) / spread.replace(0, np.nan)).dropna()
        keep = [f for f in z.index if abs(z[f]) >= TRAIT_Z and _meaningful(row[f] - overall[f], overall[f], units[f])]
        z = z[keep]
        out[c] = z.reindex(z.abs().sort_values(ascending=False).index)
    return out


def _lower_first(label: str) -> str:
    """'On-time rate' -> 'on-time rate' but 'Total Sales' -> 'total Sales' (column names keep their case)."""
    return label[:1].lower() + label[1:] if label[:1].isupper() and not label[1:2].isupper() else label


def _direction(f: str, up: bool, units, capital: bool = False) -> str:
    """'more'/'fewer' for counts (rows, orders), else 'higher'/'lower'."""
    w = ("more" if up else "fewer") if units.get(f) == "count" else ("higher" if up else "lower")
    return w.capitalize() if capital else w


def _traits(prof: pd.DataFrame, overall: pd.Series, spread: pd.Series, labels, units) -> dict[int, str]:
    out = {}
    for c, z in _trait_z(prof, overall, spread, units).items():
        z = z.head(3)
        if z.empty:
            out[c] = "Close to the overall average on every feature."
            continue
        row = prof.loc[c]
        out[c] = "; ".join(f"{_direction(f, v > 0, units)} {_lower_first(labels[f])} ({_fmt(row[f], units[f])} vs "
                           f"{_fmt(overall[f], units[f])} overall)" for f, v in z.items())
    return out


def _names(prof: pd.DataFrame, overall: pd.Series, spread: pd.Series, labels, units) -> dict[int, str]:
    """A short name per cluster from its strongest trait; clusters with the same name get the next trait added."""
    zs = _trait_z(prof, overall, spread, units)
    names = {}
    order = list(prof.columns)
    for c, z in zs.items():
        if z.empty:
            names[c] = "Close to average"
            continue
        near = [f for f in z.index if abs(z[f]) >= 0.85 * abs(z.iloc[0])]
        lead = min(near, key=order.index)
        zs[c] = pd.concat([z[[lead]], z.drop(lead)])
        names[c] = f"{_direction(lead, z[lead] > 0, units, True)} {_lower_first(labels[lead])}"
    seen: dict[str, list[int]] = {}
    for c, n in names.items():
        seen.setdefault(n, []).append(c)
    for n, cs in seen.items():
        if len(cs) > 1:
            for c in cs:
                z = zs[c]
                if len(z) > 1:
                    names[c] = f"{n}, {_direction(z.index[1], z.iloc[1] > 0, units)} {_lower_first(labels[z.index[1]])}"
    return names


def separation(X: pd.DataFrame, clusters: pd.Series) -> pd.Series:
    """Share of each feature's variance that lies between clusters (eta squared, 0-1): which features the clusters
    actually differ on. Computed on the scaled values the clustering used."""
    total = ((X - X.mean()) ** 2).sum()
    means = X.groupby(clusters).transform("mean")
    between = ((means - X.mean()) ** 2).sum()
    return (between / total.replace(0, np.nan)).fillna(0.0).clip(0, 1).sort_values(ascending=False)


def run_segmentation(table: FeatureTable, features: list[str] | None = None, k: int | None = None,
                     scaler: str = "robust", config: AnalyticsConfig = DEFAULT_ANALYTICS,
                     progress: ProgressFn | None = None) -> SegmentationResult:
    """Cluster the entities. `k=None` chooses K by silhouette (avoiding tiny clusters); otherwise K is used as given."""
    feats = list(features) if features else list(table.frame.columns)
    bad = [f for f in feats if f not in table.frame.columns]
    if bad:
        raise SegmentationError("Unknown feature(s): " + ", ".join(bad))
    if len(feats) < 2:
        raise SegmentationError("Choose at least two features to segment on.")
    if scaler not in SCALERS:
        raise SegmentationError("Scaler must be 'robust' or 'standard'.")
    raw = table.frame[feats].replace([np.inf, -np.inf], np.nan)
    n = len(raw)
    if n < config.seg_min_entities:
        raise InsufficientDataError(f"Only {n} {table.entity} value(s) are usable; segmentation needs at least {config.seg_min_entities}.")
    warnings = list(table.warnings)
    gaps = raw.isna().sum()
    if gaps.sum():
        warnings.append("Missing feature values were filled with the median of that feature: "
                        + ", ".join(f"{table.labels[c]} ({int(v)} entities)" for c, v in gaps[gaps > 0].items()) + ".")
    X = raw.fillna(raw.median())
    Z = (RobustScaler() if scaler == "robust" else StandardScaler()).fit_transform(X)
    k_max = min(config.seg_k_max, n // 3)
    if k is not None and not 2 <= k <= max(k_max, 2):
        raise SegmentationError(f"K must be between 2 and {max(k_max, 2)} for {n} entities.")
    if k_max < 2:
        raise InsufficientDataError("Too few entities to form at least two clusters of three.")
    rows, models = [], {}
    for kk in range(2, k_max + 1):
        notify(progress, kk - 2, k_max - 1, f"Trying {kk} segments, {kk - 1} of {k_max - 1}")
        km = KMeans(n_clusters=kk, n_init=10, random_state=config.seg_random_seed).fit(Z)
        labs = km.labels_
        sil = float(silhouette_score(Z, labs, sample_size=5000 if n > 5000 else None, random_state=config.seg_random_seed))
        rows.append({"k": kk, "inertia": float(km.inertia_), "silhouette": sil, "min_share": float(np.bincount(labs).min() / n)})
        models[kk] = labs
    scores = pd.DataFrame(rows)
    if k is None:
        ok = scores[scores["min_share"] >= config.seg_min_cluster_share]
        pool = ok if not ok.empty else scores
        if ok.empty:
            warnings.append("Every K gave a very small cluster, so K was chosen on silhouette alone.")
        k, mode = int(pool.loc[pool["silhouette"].idxmax(), "k"]), "automatic"      # first maximum = smallest K on ties
    else:
        mode = "chosen"
    labs = models[k]
    order = pd.Series(labs).value_counts().sort_values(ascending=False, kind="stable").index        # cluster 1 = largest
    remap = {old: new + 1 for new, old in enumerate(order)}
    cl = pd.Series(labs, index=raw.index).map(remap)
    sil = float(scores.loc[scores["k"] == k, "silhouette"].iloc[0])
    if scores.loc[scores["k"] == k, "min_share"].iloc[0] < config.seg_min_cluster_share:
        warnings.append("The smallest cluster holds very few entities; check whether it is a real group or a set of outliers.")
    prof = X.groupby(cl).mean()
    prof.insert(0, "share", cl.value_counts(normalize=True).sort_index())
    prof.insert(0, "entities", cl.value_counts().sort_index())
    overall = X.mean()
    traits = _traits(prof, overall, X.std(ddof=0), table.labels, table.units)
    names = _names(prof, overall, X.std(ddof=0), table.labels, table.units)
    sep = separation(pd.DataFrame(Z, index=raw.index, columns=feats), cl)
    assign = pd.concat([cl.rename("cluster"), raw], axis=1).reset_index()
    assign = assign.rename(columns={assign.columns[0]: table.entity}).sort_values(["cluster", table.entity]).reset_index(drop=True)
    return SegmentationResult(assign, prof, overall, traits, scores, k, mode, sil, quality_label(sil), tuple(feats),
                              {f: table.labels[f] for f in feats}, {f: table.units[f] for f in feats}, scaler, tuple(warnings),
                              names=names, separation=sep)
