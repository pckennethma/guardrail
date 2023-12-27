from __future__ import annotations

from typing import Dict, Optional, cast

import pandas as pd

from nsyn.app.q2_util.q2_grammar import _T_MODEL, Query2
from nsyn.util.base_model import BaseModel
from nsyn.util.flags import (
    SAN_RELEVANCE_ANALYSIS_FLAG,
    SAN_RELEVANCE_ANALYSIS_JSONL_PATH,
)
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.app.ml_backend.relevance_analysis")


class RelevanceAnalysisContext(BaseModel):
    query: str
    model: _T_MODEL
    dataset: str
    call_index: int = 0

    @classmethod
    def create_contexts(cls, query: Query2) -> Dict[str, RelevanceAnalysisContext]:
        """
        The method to create relevance analysis contexts for a query.

        Args:
            query (Query2): The query to create relevance analysis contexts for.

        Returns:
            Dict[str, Optional[RelevanceAnalysisContext]]: A dictionary of relevance analysis contexts.
        """
        main_query = query.main_query
        dataset = (
            query.dataset_name
            if query.dataset_version is None
            else f"{query.dataset_name}.{query.dataset_version}"
        )
        contexts: Dict[str, RelevanceAnalysisContext] = {}

        if query.projection_model is not None:
            contexts[query.projection_model[0]] = cls(
                query=main_query, model=query.projection_model, dataset=dataset
            )
        if query.projection_model_with_case_when is not None:
            contexts[query.projection_model_with_case_when.model[0]] = cls(
                query=main_query,
                model=query.projection_model_with_case_when.model,
                dataset=dataset,
            )
        if query.group_by_model is not None:
            contexts[query.group_by_model[0]] = cls(
                query=main_query, model=query.group_by_model, dataset=dataset
            )
        if query.selection_models is not None:
            for i in range(len(query.selection_models)):
                if query.selection_models[i] is not None:
                    contexts[query.selection_models[i][0]] = cls(
                        query=main_query,
                        model=query.selection_models[i],
                        dataset=dataset,
                    )
        return contexts


class RelevanceAnalysisDumpEntry(BaseModel):
    prediction_error_num: int
    detected_error_num: int
    noise_injected_num: int
    detected_noise_num: int
    noise_induced_error_num: int
    corr_prediction_error_sanitizer_alert: float
    corr_prediction_error_noise_injection: float
    corr_sanitizer_alert_noise_injection: float
    ctx: Optional[RelevanceAnalysisContext]


def relevance_analysis(
    df: pd.DataFrame,
    pred: pd.Series,
    sanitizer_alert: pd.Series,
    label_column: str,
    ra_ctx: Optional[RelevanceAnalysisContext] = None,
) -> None:
    if not SAN_RELEVANCE_ANALYSIS_FLAG:
        return

    if label_column not in df.columns:
        logger.error(
            f"Label column {label_column} not found in input data. Cannot perform relevance analysis."
        )
        return

    prediction_errors = pred != df[label_column]
    prediction_error_num = prediction_errors.sum()

    logger.info(
        f"Model mis-predicts {prediction_error_num} out of {len(df)} rows of data."
    )

    if "_nsyn_noisy_injected" not in df.columns:
        noise_injected = pd.Series([False] * len(df))
    else:
        noise_injected = df["_nsyn_noisy_injected"]
        logger.info(f"Found {noise_injected.sum()} rows of data with injected noise.")
        detected_noise = sanitizer_alert & noise_injected
        logger.info(
            f"Out of {noise_injected.sum()} rows of data with injected noise, {detected_noise.sum()} are detected by sanitizer."
        )
        noise_induced_error = prediction_errors & noise_injected
        noise_induced_error_num = noise_induced_error.sum()
        logger.info(
            f"Out of {prediction_error_num} prediction errors, {noise_induced_error_num} are induced by noise."
        )

    relevance_df = pd.DataFrame(
        {
            "prediction_error": prediction_errors,
            "sanitizer_alert": sanitizer_alert,
            "noise_injection": noise_injected,
        }
    )

    detected_error_num = relevance_df[
        (relevance_df["sanitizer_alert"] == True)  # noqa: E712
        & (relevance_df["prediction_error"] == True)  # noqa: E712
    ].shape[0]

    logger.info(f"{detected_error_num} prediction errors are detected by sanitizer.")

    logger.info(f"SANITIZER_RELEVANCE_ANALYSIS:\n{relevance_df.corr()}")

    dump_item = RelevanceAnalysisDumpEntry(
        prediction_error_num=prediction_error_num,
        detected_error_num=detected_error_num,
        noise_injected_num=noise_injected.sum(),
        detected_noise_num=detected_noise.sum(),
        noise_induced_error_num=noise_induced_error_num,
        corr_prediction_error_sanitizer_alert=cast(
            float, relevance_df.corr().loc["prediction_error", "sanitizer_alert"]
        ),
        corr_prediction_error_noise_injection=cast(
            float, relevance_df.corr().loc["prediction_error", "noise_injection"]
        ),
        corr_sanitizer_alert_noise_injection=cast(
            float, relevance_df.corr().loc["sanitizer_alert", "noise_injection"]
        ),
        ctx=ra_ctx,
    )

    # append the row to the JSONL file
    with open(SAN_RELEVANCE_ANALYSIS_JSONL_PATH, "a") as f:
        f.write(dump_item.model_dump_json() + "\n")

    if ra_ctx is not None:
        ra_ctx.call_index += 1
