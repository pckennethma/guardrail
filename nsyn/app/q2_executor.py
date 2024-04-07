import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from nsyn.app.ml_backend.analysis import AnalysisContext
from nsyn.app.ml_backend.auto import get_inference_model
from nsyn.app.q2_util.q2_grammar import _T_MODEL, Query2
from nsyn.dataset.loader import load_data_by_name, load_data_by_name_and_vers
from nsyn.util.color import get_keyword_text, get_output_df_text
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.app.q2_util.executor")

CONDITION_MATCH_PATTERN = re.compile(r"(==|!=|>|<|>=|<=)\s*('[^']*'|\"[^\"]*\"|\d+)")


class Q2Executor:
    @staticmethod
    def execute_query(
        query: Query2,
    ) -> pd.DataFrame:
        logger.info(f"Executing query: {query}")
        if query.dataset_version:
            df = load_data_by_name_and_vers(query.dataset_name, query.dataset_version)
            logger.info(
                f"Loaded dataset {query.dataset_name} with version {query.dataset_version}."
            )
        else:
            df = load_data_by_name(query.dataset_name)
            logger.info(f"Loaded dataset {query.dataset_name}")
        assert isinstance(df, pd.DataFrame), "Expected DataFrame"
        ra_ctxs = AnalysisContext.create_contexts(query)
        # perform selection filtering first
        df = Q2Executor._legacy_selection_filter(query.legacy_selection_filters, df)
        df = Q2Executor._model_selection_filter(
            query.selection_models, query.selection_model_conditions, df, ra_ctxs
        )
        # perform group by & projection
        if query.legacy_group_by_column is not None or query.group_by_model is not None:
            df_grouped: DataFrameGroupBy
            if query.legacy_group_by_column is not None:
                logger.info(
                    f"Applying legacy group by column {query.legacy_group_by_column}"
                )
                df_grouped = df.groupby(query.legacy_group_by_column)
            if query.group_by_model is not None:
                logger.info(
                    f"Applying model group by {query.group_by_model} with threshold {query.group_by_partition_threshold}"
                )
                df_grouped = Q2Executor._model_group_by(
                    query.group_by_model,
                    query.group_by_partition_threshold,
                    df,
                    ra_ctxs,
                )
            assert isinstance(df_grouped, DataFrameGroupBy)
            group_names: List[Any] = []
            aggregated_values: list[float] = []
            if (
                query.legacy_projection_column is not None
                and query.legacy_projection_agg_func is not None
                and query.legacy_projection_with_case_when is None
            ):
                logger.info(
                    f"debug Applying legacy projection column {query.legacy_projection_column} with agg func {query.legacy_projection_agg_func}"
                )
                if query.legacy_projection_agg_func == "count":
                    df = df_grouped.size().reset_index(name="count(*)")
                else:
                    df = (
                        df_grouped[query.legacy_projection_column]
                        .agg(query.legacy_projection_agg_func)
                        .reset_index(
                            name=f"{query.legacy_projection_agg_func}({query.legacy_projection_column})"
                        )
                    )
            elif (
                query.legacy_projection_with_case_when is not None
                and query.legacy_projection_agg_func is not None
                and query.legacy_projection_column is None
            ):
                logger.info(
                    f"Applying legacy projection with case when clause {query.legacy_projection_with_case_when} with agg func {query.legacy_projection_agg_func}"
                )
                projection_with_case_when = query.legacy_projection_with_case_when
                for group_name, group_df in df_grouped:
                    group_names.append(group_name)
                    aggregated_values.append(
                        group_df[projection_with_case_when.column]
                        .apply(
                            Q2Executor._get_safe_eval(
                                projection_with_case_when.condition,
                                group_df[projection_with_case_when.column].dtype
                                == bool,
                            )
                        )
                        .dropna()
                        .agg(query.legacy_projection_agg_func)
                    )
                new_column_name = f"{query.legacy_projection_agg_func}({projection_with_case_when.get_name()})"
                df = pd.DataFrame(
                    {
                        query.legacy_group_by_column: group_names,
                        new_column_name: aggregated_values,
                    }
                )
            elif (
                query.legacy_projection_with_case_when is not None
                and query.legacy_projection_agg_func is None
                and query.legacy_projection_column is None
            ):
                logger.info(
                    f"Applying legacy projection with case when clause {query.legacy_projection_with_case_when} without agg func."
                )
                projection_with_case_when = query.legacy_projection_with_case_when
                df = (
                    df_grouped[projection_with_case_when.column]
                    .apply(
                        Q2Executor._get_safe_eval(
                            projection_with_case_when.condition,
                            df[projection_with_case_when.column].dtype == bool,
                        )
                    )
                    .reset_index(
                        name=projection_with_case_when.get_name(),
                    )
                )
            elif (
                query.projection_model is not None
                and query.projection_model_agg_func is not None
                and query.projection_model_with_case_when is None
            ):
                logger.info(
                    f"Applying model projection {query.projection_model} with agg func {query.projection_model_agg_func}"
                )
                inference_engine = get_inference_model(
                    query.projection_model[0], query.projection_model[1]
                )
                model_label = inference_engine.inference_model_config.label_column
                for group_name, group_df in df_grouped:
                    group_names.append(group_name)
                    aggregated_values.append(
                        inference_engine.predict(
                            group_df, ra_ctxs[query.projection_model[0]]
                        )
                        .dropna()
                        .agg(query.projection_model_agg_func)
                    )
                df = pd.DataFrame(
                    {
                        query.legacy_group_by_column: group_names,
                        f"{query.projection_model_agg_func}({model_label} [predicted])": aggregated_values,
                    }
                )
            elif (
                query.projection_model is None
                and query.projection_model_agg_func is not None
                and query.projection_model_with_case_when is not None
            ):
                logger.info(
                    f"Applying model projection with case when clause {query.projection_model_with_case_when} with agg func {query.projection_model_agg_func}"
                )
                projection_model_with_case_when = query.projection_model_with_case_when
                inference_engine = get_inference_model(
                    projection_model_with_case_when.model[0],
                    projection_model_with_case_when.model[1],
                )
                model_label = inference_engine.inference_model_config.label_column

                for group_name, group_df in df_grouped:
                    group_names.append(group_name)
                    pred = inference_engine.predict(
                        group_df, ra_ctxs[projection_model_with_case_when.model[0]]
                    )
                    aggregated_values.append(
                        pred.apply(
                            Q2Executor._get_safe_eval(
                                projection_model_with_case_when.condition,
                                pred.dtype == bool,
                            )
                        )
                        .dropna()
                        .agg(query.projection_model_agg_func)
                    )
                new_column_name = projection_model_with_case_when.get_name(model_label)
                df = pd.DataFrame(
                    {
                        query.legacy_group_by_column: group_names,
                        new_column_name: aggregated_values,
                    }
                )
            else:
                raise ValueError("Invalid query")
        # no group by
        else:
            if (
                query.legacy_projection_column == "*"
                and query.legacy_projection_agg_func is None
                and query.legacy_projection_with_case_when is None
            ):
                logger.info("Applying legacy projection column *")
                df = df
            elif (
                query.legacy_projection_column is not None
                and query.legacy_projection_agg_func is not None
                and query.legacy_projection_with_case_when is None
            ):
                logger.info(
                    f"Applying legacy projection column {query.legacy_projection_column} with agg func {query.legacy_projection_agg_func}"
                )
                if query.legacy_projection_agg_func == "count":
                    df = pd.DataFrame(
                        {
                            f"{query.legacy_projection_agg_func}({query.legacy_projection_column})": [
                                len(df)
                            ]
                        }
                    )
                else:
                    df = df[[query.legacy_projection_column]]
                    df = df.agg(query.legacy_projection_agg_func).reset_index(
                        name=f"{query.legacy_projection_agg_func}({query.legacy_projection_column})"
                    )
            elif (
                query.legacy_projection_column is not None
                and query.legacy_projection_agg_func is None
                and query.legacy_projection_with_case_when is None
            ):
                logger.info(
                    f"Applying legacy projection column {query.legacy_projection_column}"
                )
                df = df[[query.legacy_projection_column]]
            elif (
                query.legacy_projection_with_case_when is not None
                and query.legacy_projection_agg_func is not None
                and query.legacy_projection_column is None
            ):
                logger.info(
                    f"Applying legacy projection with case when clause {query.legacy_projection_with_case_when} with agg func {query.legacy_projection_agg_func}"
                )
                projection_with_case_when = query.legacy_projection_with_case_when
                agg_val = (
                    df[projection_with_case_when.column]
                    .apply(
                        Q2Executor._get_safe_eval(
                            projection_with_case_when.condition,
                            df[projection_with_case_when.column].dtype == bool,
                        )
                    )
                    .agg(query.legacy_projection_agg_func)
                )
                new_column_name = f"{query.legacy_projection_agg_func}({projection_with_case_when.get_name()})"
                df = pd.DataFrame({new_column_name: [agg_val]})
            elif (
                query.legacy_projection_with_case_when is not None
                and query.legacy_projection_agg_func is None
                and query.legacy_projection_column is None
            ):
                logger.info(
                    f"Applying legacy projection with case when clause {query.legacy_projection_with_case_when} without agg func."
                )
                projection_with_case_when = query.legacy_projection_with_case_when
                new_column_name = projection_with_case_when.get_name()
                df[new_column_name] = df[projection_with_case_when.column].apply(
                    Q2Executor._get_safe_eval(
                        projection_with_case_when.condition,
                        df[projection_with_case_when.column].dtype == bool,
                    )
                )
                # move the case when column to the front
                df = df[
                    [
                        new_column_name,
                        *[c for c in df.columns if c != new_column_name],
                    ]
                ]

            elif (
                query.projection_model is not None
                and query.projection_model_agg_func is not None
                and query.projection_model_with_case_when is None
            ):
                logger.info(
                    f"Applying model projection {query.projection_model} with agg func {query.projection_model_agg_func}"
                )
                inference_engine = get_inference_model(
                    query.projection_model[0], query.projection_model[1]
                )
                agg_val = inference_engine.predict(
                    df, ra_ctxs[query.projection_model[0]]
                ).agg(query.projection_model_agg_func)
                model_label = inference_engine.inference_model_config.label_column
                df = pd.DataFrame(
                    {
                        f"{query.projection_model_agg_func}({model_label} [predicted])": [
                            agg_val
                        ]
                    }
                )
            elif (
                query.projection_model is not None
                and query.projection_model_agg_func is None
                and query.projection_model_with_case_when is None
            ):
                logger.info(
                    f"Applying model projection {query.projection_model} with agg func {query.projection_model_agg_func}"
                )
                inference_engine = get_inference_model(
                    query.projection_model[0], query.projection_model[1]
                )
                model_label = inference_engine.inference_model_config.label_column
                df[f"{model_label} [predicted]"] = inference_engine.predict(
                    df, ra_ctxs[query.projection_model[0]]
                )
                # move the predicted column to the front
                df = df[
                    [
                        f"{model_label} [predicted]",
                        *[c for c in df.columns if c != f"{model_label} [predicted]"],
                    ]
                ]
            elif (
                query.projection_model is None
                and query.projection_model_agg_func is not None
                and query.projection_model_with_case_when is not None
            ):
                logger.info(
                    f"Applying model projection with case when clause {query.projection_model_with_case_when} with agg func {query.projection_model_agg_func}"
                )
                projection_model_with_case_when = query.projection_model_with_case_when
                inference_engine = get_inference_model(
                    projection_model_with_case_when.model[0],
                    projection_model_with_case_when.model[1],
                )
                pred = inference_engine.predict(
                    df, ra_ctxs[projection_model_with_case_when.model[0]]
                )
                agg_val = pred.apply(
                    Q2Executor._get_safe_eval(
                        projection_model_with_case_when.condition,
                        pred.dtype == bool,
                    )
                ).agg(query.projection_model_agg_func)
                model_label = inference_engine.inference_model_config.label_column
                new_column_name = f"{query.projection_model_agg_func}({projection_model_with_case_when.get_name(model_label)})"
                df = pd.DataFrame({new_column_name: [agg_val]})
            elif (
                query.projection_model is None
                and query.projection_model_agg_func is None
                and query.projection_model_with_case_when is not None
            ):
                logger.info(
                    f"Applying model projection with case when clause {query.projection_model_with_case_when} without agg func."
                )
                projection_model_with_case_when = query.projection_model_with_case_when
                inference_engine = get_inference_model(
                    projection_model_with_case_when.model[0],
                    projection_model_with_case_when.model[1],
                )
                model_label = inference_engine.inference_model_config.label_column
                new_column_name = projection_model_with_case_when.get_name(model_label)
                pred = inference_engine.predict(
                    df, ra_ctxs[projection_model_with_case_when.model[0]]
                )
                df[new_column_name] = pred.apply(
                    Q2Executor._get_safe_eval(
                        projection_model_with_case_when.condition,
                        pred.dtype == bool,
                    )
                )
                # move the predicted column to the front
                df = df[
                    [
                        new_column_name,
                        *[c for c in df.columns if c != new_column_name],
                    ]
                ]
            else:
                raise ValueError("Invalid query")
        return df

    @staticmethod
    def _legacy_selection_filter(
        filters: Optional[List[Tuple[str, Any]]], df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply selection filters to a DataFrame.
        """
        if filters is None:
            return df
        logger.info(
            f"Applying legacy selection filters: {filters} on df with shape {df.shape}"
        )
        for column, condition in filters:
            safe_eval_func = Q2Executor._get_safe_eval(
                condition, df[column].dtype == bool
            )
            df = df[df[column].apply(safe_eval_func)]
        logger.info(f"Result df: {df.shape}")
        return df

    @staticmethod
    def _model_selection_filter(
        selection_models: Optional[List[_T_MODEL]],
        selection_model_conditions: Optional[List[str]],
        df: pd.DataFrame,
        ra_ctxs: Dict[str, AnalysisContext],
    ) -> pd.DataFrame:
        """
        Apply model selection filters to a DataFrame.
        """
        if selection_models is None:
            return df
        assert (
            selection_model_conditions is not None
        ), "Expected model conditions to be provided if models are provided"
        logger.info(
            f"Applying model selection filters: {selection_models} with conditions {selection_model_conditions} on df with shape {df.shape}"
        )
        for model, condition in zip(selection_models, selection_model_conditions):
            inference_engine = get_inference_model(model[0], model[1])
            pred = inference_engine.predict(df, ra_ctxs[model[0]])
            safe_eval_func = Q2Executor._get_safe_eval(condition, pred.dtype == bool)
            eval_result = pred.apply(safe_eval_func)
            df = df[eval_result]
        logger.info(f"Result df: {df.shape}")
        return df

    @staticmethod
    def _model_group_by(
        group_by_model: _T_MODEL,
        group_by_partition_threshold: Optional[float],
        df: pd.DataFrame,
        ra_ctxs: Dict[str, AnalysisContext],
    ) -> DataFrameGroupBy:
        """
        Group a DataFrame by a model.

        Args:
            group_by_model (_T_MODEL): The model to group by.
            group_by_partition_threshold (Optional[float]): The threshold to partition the data by.
            df (pd.DataFrame): The DataFrame to group.

        Returns:
            DataFrameGroupBy: The grouped DataFrame.
        """
        inference_engine = get_inference_model(group_by_model[0], group_by_model[1])
        pred_result = inference_engine.predict(df, ra_ctxs[group_by_model[0]])
        if group_by_partition_threshold is not None:
            # Create formatted strings
            greater_than_str = f"> {group_by_partition_threshold}"
            less_than_or_equal_str = f"<= {group_by_partition_threshold}"

            # Create masks for conditions
            greater_than_threshold = pred_result > group_by_partition_threshold
            less_than_or_equal_to_threshold = (
                pred_result <= group_by_partition_threshold
            )

            # Apply vectorized operations
            pred_result = pd.Series(
                np.where(
                    greater_than_threshold,
                    greater_than_str,
                    np.where(
                        less_than_or_equal_to_threshold, less_than_or_equal_str, "NaN"
                    ),
                ),
                index=pred_result.index,
            )

        model_label = inference_engine.inference_model_config.label_column
        df[f"{model_label} [predicted]"] = pred_result
        grouped_df = df.groupby(f"{model_label} [predicted]", dropna=False)
        return grouped_df

    @staticmethod
    def _get_safe_eval(
        condition: str,
        is_bool: bool,
    ) -> Callable[[Any], bool]:
        """
        Get a safe eval function for a condition.
        """

        match = CONDITION_MATCH_PATTERN.match(condition)
        assert match is not None, f"Invalid condition: {condition}"
        op, val2 = match.groups()
        assert isinstance(val2, str)
        true_val2 = np.safe_eval(val2)
        if isinstance(true_val2, str) and is_bool:
            boolean_flag = true_val2.lower() in [
                "true",
                "false",
            ]
        else:
            boolean_flag = False
        boolean_true_val2 = bool(true_val2)
        logger.info(
            f"Creating safe eval function for condition {condition} with op {op}, true_val2 {true_val2}, boolean_flag {boolean_flag}, boolean_true_val2 {boolean_true_val2}"
        )
        if op == "==" and boolean_flag:
            return lambda val: val == boolean_true_val2
        elif op == "!=" and boolean_flag:
            return lambda val: val != boolean_true_val2
        elif op == "==" and not boolean_flag:
            return lambda val: val == true_val2
        elif op == "!=" and not boolean_flag:
            return lambda val: val != true_val2
        elif op == ">":
            return lambda val: val > true_val2
        elif op == "<":
            return lambda val: val < true_val2
        elif op == ">=":
            return lambda val: val >= true_val2
        elif op == "<=":
            return lambda val: val <= true_val2
        else:
            raise ValueError(f"Invalid operator: {op} in condition: {condition}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query_path",
        "-p",
        type=str,
        help="The path to the query file (.sql). It can be either a single query file or a folder containing multiple query files (with .sql extension). If specified, the following arguments will be ignored.",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="The query to execute.",
    )

    args = parser.parse_args()

    if args.query_path is not None and os.path.isdir(args.query_path):
        # list all txt files in the folder
        logger.info(f"Executing all queries in {args.query_path} ...")
        query_list = [
            os.path.join(args.query_path, f)
            for f in os.listdir(args.query_path)
            if os.path.isfile(os.path.join(args.query_path, f)) and f.endswith(".sql")
        ]
        query_list.sort()
        logger.info(f"Found {len(query_list)} queries.")
        for query_path in query_list:
            with open(query_path, "r") as f:
                logger.info(f"Executing query in {query_path} ...")
                query = Query2.parse_query(f.read())
                df = Q2Executor.execute_query(query)
                logger.info(f"Query: {get_keyword_text(query.main_query)}")
                logger.info(f"Result df:\n{get_output_df_text(df.to_markdown())}")
    elif (
        args.query_path is not None and os.path.isfile(args.query_path)
    ) or args.query is not None:
        if args.query_path is not None:
            with open(args.query_path, "r") as f:
                query = Query2.parse_query(f.read())
        else:
            query = Query2.parse_query(args.query)
        df = Q2Executor.execute_query(query)
        logger.info(f"Query: {get_keyword_text(query.main_query)}")
        logger.info(f"Result df:\n{get_output_df_text(df.to_markdown())}")
    else:
        parser.print_help()
