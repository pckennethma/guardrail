from __future__ import annotations

from typing import Dict, Optional, cast

import pandas as pd

from nsyn.app.q2_util.q2_grammar import _T_MODEL, Query2
from nsyn.util.base_model import BaseModel
from nsyn.util.flags import (
    ERROR_HANDLING_FLAG,
    SAN_ANALYSIS_OUTPUT_JSONL_PATH,
    SAN_COMPARATIVE_ANALYSIS_FLAG,
    SAN_RELEVANCE_ANALYSIS_FLAG,
)
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.app.ml_backend.analysis")


class AnalysisContext(BaseModel):
    query: str
    model: _T_MODEL
    dataset: str
    call_index: int = 0

    @classmethod
    def create_contexts(cls, query: Query2) -> Dict[str, AnalysisContext]:
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
        contexts: Dict[str, AnalysisContext] = {}

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


class RelevanceAnalysisDumpItem(BaseModel):
    total_pred_error_num: int
    detected_pred_error_num: int
    actual_data_error_num: int
    detected_data_error_num: int
    falsely_detected_data_error_num: int
    data_error_caused_pred_error: int
    total_input_num: int
    corr_prediction_error_sanitizer_alert: Optional[float]
    corr_prediction_error_noise_injection: Optional[float]
    corr_sanitizer_alert_noise_injection: Optional[float]
    ctx: Optional[AnalysisContext]


class ComparativeAnalysisDumpItem(BaseModel):
    original_prediction_error_num: int
    rectified_prediction_error_num: int

    noise_injected_num: int
    rectified_noise_injected_num: int

    total_input_num: int = 0

    ctx: Optional[AnalysisContext]


def relevance_analysis(
    df: pd.DataFrame,
    pred: pd.Series,
    sanitizer_alert: pd.Series,
    label_column: str,
    ctx: Optional[AnalysisContext] = None,
) -> None:
    if not SAN_RELEVANCE_ANALYSIS_FLAG:
        return

    if ERROR_HANDLING_FLAG == "rectify":
        logger.info(
            "No relevance analysis performed because error handling flag is rectify."
        )
        return

    if label_column not in df.columns:
        logger.error(
            f"Label column {label_column} not found in input data. Cannot perform relevance analysis."
        )
        return

    prediction_errors = pred != df[label_column]
    total_pred_error_num = prediction_errors.sum()

    logger.info(
        f"Model mis-predicts {total_pred_error_num} out of {len(df)} rows of data."
    )

    if "_nsyn_noisy_injected" not in df.columns:
        error_injected = pd.Series([False] * len(df))
    else:
        error_injected = df["_nsyn_noisy_injected"]
        logger.info(
            f"Ground truth: {error_injected.sum()} rows of data with actual data errors."
        )
        error_detected = sanitizer_alert & error_injected
        logger.info(
            f"Out of {error_injected.sum()} rows of data with actual data errors, {error_detected.sum()} are detected by sanitizer."
        )
        noise_induced_error = prediction_errors & error_injected
        data_error_caused_pred_error = noise_induced_error.sum()
        logger.info(
            f"Out of {total_pred_error_num} prediction errors, {data_error_caused_pred_error} are induced by data errors."
        )
        falsely_detected_data_error_num = (sanitizer_alert & ~error_injected).sum()

    relevance_df = pd.DataFrame(
        {
            "prediction_error": prediction_errors,
            "sanitizer_alert": sanitizer_alert,
            "noise_injection": error_injected,
        }
    )

    detected_pred_error_num = relevance_df[
        (relevance_df["sanitizer_alert"] == True)  # noqa: E712
        & (relevance_df["prediction_error"] == True)  # noqa: E712
    ].shape[0]

    logger.info(f"{detected_pred_error_num} prediction errors are detected by sanitizer.")

    logger.info(f"SANITIZER_RELEVANCE_ANALYSIS:\n{relevance_df.corr()}")

    logger.info(
        """\# Detected~Mis-pred. / \#Total~Detected~Data~Error: {:.2f}""".format(
            detected_pred_error_num / error_detected.sum()
        )
    )

    # $\frac{\text{\# Missed~Mis-pred.}}{\text{\#Total~Missed~Data~Error}}$
    logger.info(
        """\# Missed~Mis-pred. / \#Total~Missed~Data~Error: {:.2f}""".format(
            (total_pred_error_num - detected_pred_error_num)
            / (error_injected.sum() - error_detected.sum())
        )
    )

    if SAN_ANALYSIS_OUTPUT_JSONL_PATH is not None:
        dump_item = RelevanceAnalysisDumpItem(
            total_pred_error_num=total_pred_error_num,
            detected_pred_error_num=detected_pred_error_num,
            actual_data_error_num=error_injected.sum(),
            detected_data_error_num=error_detected.sum(),
            data_error_caused_pred_error=data_error_caused_pred_error,
            total_input_num=len(df),
            falsely_detected_data_error_num=falsely_detected_data_error_num,
            corr_prediction_error_sanitizer_alert=cast(
                float, relevance_df.corr().loc["prediction_error", "sanitizer_alert"]
            ),
            corr_prediction_error_noise_injection=cast(
                float, relevance_df.corr().loc["prediction_error", "noise_injection"]
            ),
            corr_sanitizer_alert_noise_injection=cast(
                float, relevance_df.corr().loc["sanitizer_alert", "noise_injection"]
            ),
            ctx=ctx,
        )

        # append the row to the JSONL file
        logger.info(f"Dumping relevance analysis result to {SAN_ANALYSIS_OUTPUT_JSONL_PATH}")
        with open(SAN_ANALYSIS_OUTPUT_JSONL_PATH, "a") as f:
            f.write(dump_item.model_dump_json() + "\n")

    if ctx is not None:
        ctx.call_index += 1


def comparative_analysis(
    df: pd.DataFrame,
    rectified_noise_injected_num: int,
    pred: pd.Series,
    original_pred: pd.Series,
    label_column: str,
    ra_ctx: Optional[AnalysisContext] = None,
) -> None:
    if not SAN_COMPARATIVE_ANALYSIS_FLAG:
        raise RuntimeError(
            "Should not reach here if SAN_COMPARATIVE_ANALYSIS_FLAG is False."
        )
    if ERROR_HANDLING_FLAG != "rectify":
        raise RuntimeError(
            "Should not reach here if ERROR_HANDLING_FLAG is not rectify."
        )

    original_prediction_error_num = (original_pred != df[label_column]).sum()
    rectified_prediction_error_num = (pred != df[label_column]).sum()

    noise_injected = df["_nsyn_noisy_injected"]
    original_noise_injected_num = noise_injected.sum()
    total_input_num = len(df)

    dump_item = ComparativeAnalysisDumpItem(
        original_prediction_error_num=original_prediction_error_num,
        rectified_prediction_error_num=rectified_prediction_error_num,
        noise_injected_num=original_noise_injected_num,
        rectified_noise_injected_num=rectified_noise_injected_num,
        total_input_num=total_input_num,
        ctx=ra_ctx,
    )

    # append the row to the JSONL file
    logger.info(f"Dumping comparative analysis result to {SAN_ANALYSIS_OUTPUT_JSONL_PATH}")
    with open(SAN_ANALYSIS_OUTPUT_JSONL_PATH, "a") as f:
        f.write(dump_item.model_dump_json() + "\n")

    if ra_ctx is not None:
        ra_ctx.call_index += 1
