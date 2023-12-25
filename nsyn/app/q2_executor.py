from typing import Any, List, Optional, Tuple

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from nsyn.app.ml_backend.auto import get_inference_model
from nsyn.app.q2_util.q2 import _T_MODEL, Query2
from nsyn.dataset.loader import load_data_by_name, load_data_by_name_and_vers
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.app.q2_util.executor")


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
        # perform selection filtering first
        df = Q2Executor._legacy_selection_filter(query.legacy_selection_filters, df)
        df = Q2Executor._model_selection_filter(
            query.selection_models, query.selection_model_conditions, df
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
                )
            assert isinstance(df_grouped, DataFrameGroupBy)
            if (
                query.legacy_projection_column is not None
                and query.legacy_projection_agg_func is not None
            ):
                logger.info(
                    f"Applying legacy projection column {query.legacy_projection_column} with agg func {query.legacy_projection_agg_func}"
                )
                df = (
                    df_grouped[query.legacy_projection_column]
                    .agg(query.legacy_projection_agg_func)
                    .reset_index(
                        name=f"{query.legacy_projection_agg_func}({query.legacy_projection_column})"
                    )
                )
            elif query.projection_model is not None and query.projection_model_agg_func:
                logger.info(
                    f"Applying model projection {query.projection_model} with agg func {query.projection_model_agg_func}"
                )
                inference_engine = get_inference_model(
                    query.projection_model[0], query.projection_model[1]
                )
                model_label = inference_engine.inference_model_config.label_column
                df = (
                    df_grouped.apply(inference_engine.predict)
                    .agg(query.projection_model_agg_func)
                    .reset_index(
                        name=f"{query.projection_model_agg_func}({model_label} (predicted))"
                    )
                )
            else:
                raise ValueError("Invalid query")
        else:
            if query.legacy_projection_column == "*":
                logger.info("Applying legacy projection column *")
                df = df
            elif (
                query.legacy_projection_column is not None
                and query.legacy_projection_agg_func is not None
            ):
                logger.info(
                    f"Applying legacy projection column {query.legacy_projection_column} with agg func {query.legacy_projection_agg_func}"
                )
                df = df[[query.legacy_projection_column]]
                df = df.agg(query.legacy_projection_agg_func).reset_index(
                    name=f"{query.legacy_projection_agg_func}({query.legacy_projection_column})"
                )
            elif (
                query.legacy_projection_column is not None
                and query.legacy_projection_agg_func is None
            ):
                logger.info(
                    f"Applying legacy projection column {query.legacy_projection_column}"
                )
                df = df[[query.legacy_projection_column]]
            elif (
                query.projection_model is not None
                and query.projection_model_agg_func is not None
            ):
                logger.info(
                    f"Applying model projection {query.projection_model} with agg func {query.projection_model_agg_func}"
                )
                inference_engine = get_inference_model(
                    query.projection_model[0], query.projection_model[1]
                )
                agg_val = inference_engine.predict(df).agg(
                    query.projection_model_agg_func
                )
                model_label = inference_engine.inference_model_config.label_column
                df = pd.DataFrame(
                    {
                        f"{query.projection_model_agg_func}({model_label} (predicted))": [
                            agg_val
                        ]
                    }
                )
            elif (
                query.projection_model is not None
                and query.projection_model_agg_func is None
            ):
                logger.info(
                    f"Applying model projection {query.projection_model} with agg func {query.projection_model_agg_func}"
                )
                inference_engine = get_inference_model(
                    query.projection_model[0], query.projection_model[1]
                )
                model_label = inference_engine.inference_model_config.label_column
                df = inference_engine.predict(df).reset_index(
                    name=f"{model_label} (predicted)"
                )
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
            # TODO: this is a hacky way to handle string conditions a method
            # similar to `np.safe_eval` should be used
            df = df[df[column].apply(lambda x: eval(f'"""{x}""" {condition}'))]
        logger.info(f"Result df: {df.shape}")
        return df

    @staticmethod
    def _model_selection_filter(
        selection_models: Optional[List[_T_MODEL]],
        selection_model_conditions: Optional[List[str]],
        df: pd.DataFrame,
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
            # TODO: this is a hacky way to handle string conditions a method
            # similar to `np.safe_eval` should be used
            eval_result = inference_engine.predict(df).apply(
                lambda x: eval(f"{x} {condition}")
            )
            df = df[eval_result]
        logger.info(f"Result df: {df.shape}")
        return df

    @staticmethod
    def _model_group_by(
        group_by_model: _T_MODEL,
        group_by_partition_threshold: Optional[float],
        df: pd.DataFrame,
    ) -> DataFrameGroupBy:
        """
        Group a DataFrame by a model.
        """
        inference_engine = get_inference_model(group_by_model[0], group_by_model[1])
        pred_result = inference_engine.predict(df)
        if group_by_partition_threshold is not None:
            pred_result = pred_result.apply(
                lambda x: (
                    f"> {group_by_partition_threshold}"
                    if x > group_by_partition_threshold
                    else f"<= {group_by_partition_threshold}"
                )
            )
        model_label = inference_engine.inference_model_config.label_column
        df[f"{model_label} (predicted)"] = pred_result
        return df.groupby(f"{model_label} (predicted)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query_path",
        "-p",
        type=str,
        help="The path to the query file. If specified, the following arguments will be ignored.",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="The query to execute.",
    )

    args = parser.parse_args()

    if args.query_path is not None:
        with open(args.query_path, "r") as f:
            query = Query2.parse_query(f.read())
    elif args.query is not None:
        query = Query2.parse_query(args.query)
    else:
        raise ValueError("Either query_path or query must be provided.")

    df = Q2Executor.execute_query(query)

    logger.info(f"Result df:\n{df}")
