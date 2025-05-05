import sys
from pathlib import Path

try:                                   # regular .py execution
    repo_root = Path(__file__).resolve().parent.parent
except NameError:                      # Jupyter / IPython
    repo_root = Path().resolve().parent

sys.path.append(str(repo_root))
print("Added to sys.path →", repo_root)


import logging
from typing import Dict, Any, List, Set

import pandas as pd
import numpy as np

from jailbreaks.evaluators.llm_judge.judge import (
    GroqLLMJudge,
    GoogleLLMJudge,
    LocalLLMJudge
)

logger = logging.getLogger("LLM Judge")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def keep_annotated_rows(df: pd.DataFrame,
                        metrics: List[str],
                        annotation_values: Set[Any]) -> pd.DataFrame:
    """
    Return a dataframe that contains *only* the rows where at least one of the
    `metrics` columns has a non-empty / valid annotation.
    """
    
    # Make sure we do not access non-existent columns
    present = [m for m in metrics if m in df.columns]
    if not present:
        logger.warning("None of the requested metrics are present in df")
        return df.iloc[0:0]          # empty frame, same columns
    
    # Build a boolean mask: row is True if *any* metric column is annotated
    mask = pd.Series(False, index=df.index, name="has_annotation")
    for m in present:
        mask |= df[m].isin(annotation_values)
    
    filtered = df.loc[mask].copy()
    logger.info(f"Kept {len(filtered):,} / {len(df):,} rows that have "
                f"at least one annotation in {present}")
    
    return filtered

def _to_bool(x):
    """
    Convert many possible truthy / falsy annotations
    to proper Python booleans.  Returns None if unknown.
    """
    if x in (True,  "True",  1, "1"):  return True
    if x in (False, "False", 0, "0"):  return False
    return np.nan


def score_df(df: pd.DataFrame, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    scores: Dict[str, Dict[str, float]] = {}

    for m in metrics:
        # raw → {True, False, np.nan}
        y_true_raw = df[m]              .map(_to_bool)
        y_pred_raw = df[f"{m}_pred"]    .map(_to_bool)

        gold_mask = ~y_true_raw.isna()
        if gold_mask.sum() == 0:
            logger.warning(f"No gold labels for metric '{m}', skipping.")
            continue

        # OPTION A  – penalise missing predictions
        y_true = y_true_raw[gold_mask].astype(bool)
        y_pred = (
            y_pred_raw[gold_mask]
            .fillna(False)       # <- “very bad score for NaNs”
            .astype(bool)
        )


        scores[m] = {
            "accuracy" : accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall"   : recall_score   (y_true, y_pred, zero_division=0),
            "f1"       : f1_score       (y_true, y_pred, zero_division=0),
            "n"        : int(len(y_true)),
            "coverage" : 1 - y_pred_raw[gold_mask].isna().mean()  # just info
        }
    return scores

if __name__ == "__main__":
    judge = GroqLLMJudge(model="llama-3.1-8b-instant")

    csv_path = repo_root / "annotated_responses.csv"
    df = pd.read_csv(csv_path)
    
    METRICS = ["refusal", "ethical", "harmful"]
    ANNOTATION_VALUES = {True, False, "True", "False", 1, 0}
    
    df_annotated = keep_annotated_rows(df, METRICS, ANNOTATION_VALUES)
    
    N_ROWS = 20
    
    sampled = df_annotated.iloc[:N_ROWS]

    vote_list: List[Dict[str, Any]] = judge.vote_batch(        #│
        queries   = sampled["prompt"].tolist(),                #│
        responses = sampled["response"].tolist(),              #│
        batch_size=8                                           #│
    )               

    pred_list = [v.scores for v in vote_list]
    metadata_list = [v.metadata() for v in vote_list]

    pred_df = (
        pd.DataFrame(pred_list, index=sampled.index)   # same row-order
        .reindex(columns=METRICS)                    # keep expected cols
        .add_suffix("_pred")                         # e.g. "refusal_pred"
    )

    metadata_df = pd.DataFrame(metadata_list, index=sampled.index)                        

    # 2) Merge gold & predicted labels ------------------------------------------
    df_eval = pd.concat([sampled, pred_df], axis=1)
    
    df_votes = pd.concat([sampled, pred_df, metadata_df], axis=1)
    df_votes.to_csv("votes.csv", index=False)

    # 3) Compute evaluation scores ---------------------------------------------- 
    scores = score_df(df_eval, METRICS)
    
    metrics_df = (
        pd.DataFrame(scores)
        .T
        .sort_index()
        .round(3)
        .rename_axis("metric")
    )

    print(metrics_df)

