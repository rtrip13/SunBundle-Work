"""
Scoring engine for SunBundle Expansion Decision Tool.

Two-part model:
- Need score: poverty_rate, inverse median_household_income, school_count
- Feasibility score: density, distance_to_ann_arbor_miles

total_score = need_weight * need_score + feas_weight * feasibility_score
"""

from __future__ import annotations

import pandas as pd

# Direction of desirability for each metric:
# True  => higher raw value = higher score
# False => lower raw value = higher score
SCORING_DIRECTION = {
    "poverty_rate": True,
    "median_household_income": False,  # lower income = more need
    "school_count": True,
    "density": True,
    "distance_to_ann_arbor_miles": False,  # closer = better
}


def normalize_series(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """
    Min-max normalize to [0, 1].
    If higher_is_better is False, invert so lower raw values get higher scores.
    """
    s = series.astype(float)
    min_val, max_val = s.min(), s.max()
    if pd.isna(min_val) or max_val == min_val:
        return pd.Series(0.5, index=series.index)
    norm = (s - min_val) / (max_val - min_val)
    if not higher_is_better:
        norm = 1 - norm
    return norm


def _ensure_normalized_metrics(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Ensure score_<metric> columns exist for all requested metrics."""
    out = df.copy()
    for col in metrics:
        if col not in out.columns:
            continue
        score_col = f"score_{col}"
        if score_col in out.columns:
            continue
        higher = SCORING_DIRECTION.get(col, True)
        out[score_col] = normalize_series(out[col], higher_is_better=higher)
    return out


def _component_score(
    df: pd.DataFrame,
    metric_weights: dict[str, float],
    name: str,
) -> pd.Series:
    """
    Compute a component score (need or feasibility) as a weighted combo of normalized metrics.
    metric_weights: keys are metric names on df (e.g. 'poverty_rate').
    """
    used_metrics = [m for m, w in metric_weights.items() if w and m in df.columns]
    if not used_metrics:
        return pd.Series(0.0, index=df.index, name=f"{name}_score")

    out = _ensure_normalized_metrics(df, used_metrics)
    total_w = sum(metric_weights[m] for m in used_metrics)
    if total_w == 0:
        return pd.Series(0.0, index=df.index, name=f"{name}_score")

    weights_norm = {m: metric_weights[m] / total_w for m in used_metrics}
    score = pd.Series(0.0, index=df.index)
    for m in used_metrics:
        score_col = f"score_{m}"
        if score_col in out.columns:
            score += weights_norm[m] * out[score_col].fillna(0)
    score.name = f"{name}_score"
    return score


def run_scoring(
    df: pd.DataFrame,
    need_weights: dict[str, float],
    feasibility_weights: dict[str, float],
    overall_need_weight: float = 0.7,
    overall_feasibility_weight: float = 0.3,
) -> pd.DataFrame:
    """
    Compute need_score, feasibility_score, and total_score.

    - need_weights: e.g. {'poverty_rate': 45, 'median_household_income': 35, 'school_count': 20}
    - feasibility_weights: e.g. {'density': 50, 'distance_to_ann_arbor_miles': 50}
    - overall_*_weight are relative (will be normalized to sum to 1)
    """
    if df.empty:
        return df.assign(need_score=0.0, feasibility_score=0.0, total_score=0.0)

    need_score = _component_score(df, need_weights, "need")
    feas_score = _component_score(df, feasibility_weights, "feasibility")

    total_overall = overall_need_weight + overall_feasibility_weight
    if total_overall <= 0:
        need_w = feas_w = 0.5
    else:
        need_w = overall_need_weight / total_overall
        feas_w = overall_feasibility_weight / total_overall

    total_score = need_w * need_score + feas_w * feas_score
    result = df.copy()
    result["need_score"] = need_score
    result["feasibility_score"] = feas_score
    result["total_score"] = total_score
    return result

