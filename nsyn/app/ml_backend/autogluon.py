from __future__ import annotations

import inspect
import os
import pickle
from typing import Literal, Optional

import pandas as pd
from autogluon.tabular import TabularPredictor

from nsyn.app.ml_backend.analysis import (
    AnalysisContext,
    comparative_analysis,
    relevance_analysis,
)
from nsyn.app.ml_backend.base import BaseModelConfig, InferenceModel
from nsyn.app.ml_backend.error_handling import postprocessing, preprocessing
from nsyn.dsl.prog import DSLProg
from nsyn.util.flags import DISABLE_SANITIZER_FLAG, SAN_COMPARATIVE_ANALYSIS_FLAG
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.app.ml_backend.autogluon")

_DEFAULT_MODEL_ROOT = "models"

_QUERY_EXECUTION_FUNC_NAMES = [
    "execute_query",
    "_model_selection_filter",
    "_model_group_by",
]


class AutoGluonConfig(BaseModelConfig):
    """
    A class to represent the configuration for AutoGluon model.
    """


class AutoGluonModel(InferenceModel):
    """
    A class to represent the AutoGluon model.
    """

    # Use `_model_config` to avoid name collision with `model_config` in `BaseModel`
    inference_model_config: AutoGluonConfig
    model: TabularPredictor
    sanitizer: Optional[DSLProg] = None

    @classmethod
    def create(cls, model_path: str) -> AutoGluonModel:
        """
        Initializes the AutoGluon model.

        Args:
            model_path (str): The path to the AutoGluon model.

        Returns:
            AutoGluonModel: The initialized AutoGluon model.
        """
        # try default model root first
        if not os.path.isdir(os.path.join(_DEFAULT_MODEL_ROOT, model_path)):
            # try absolute path
            if not os.path.isdir(model_path):
                raise ValueError(
                    f"AutoGluon model path {model_path} is not a directory."
                )
            else:
                model_path = os.path.abspath(model_path)
        else:
            model_path = os.path.join(_DEFAULT_MODEL_ROOT, model_path)
            logger.info(f"Use {model_path} as model path.")
        model = TabularPredictor.load(model_path)
        label = model.label
        assert isinstance(label, str)
        feature_columns = model.feature_metadata_in.get_features()

        model_output_type: Literal["discrete", "continuous"]
        if model.problem_type == "binary" or model.problem_type == "multiclass":
            model_output_type = "discrete"
        elif model.problem_type == "regression":
            model_output_type = "continuous"
        else:
            raise ValueError(f"Problem type {model.problem_type} is not supported.")

        model_config = AutoGluonConfig(
            model_path=model_path,
            label_column=label,
            feature_columns=feature_columns,
            model_output_type=model_output_type,
        )

        if not isinstance(model, TabularPredictor):
            raise TypeError(f"Model is not an instance of TabularPredictor: {model}")

        if os.path.isfile(os.path.join(model_path, "nsyn_prog.pkl")):
            logger.info("Loading sanitizer...")
            with open(os.path.join(model_path, "nsyn_prog.pkl"), "rb") as f:
                sanitizer = pickle.load(f)
                assert isinstance(sanitizer, DSLProg)
            logger.info("Sanitizer loaded.")
        else:
            logger.info("No sanitizer found.")
            sanitizer = None
        return cls(
            inference_model_config=model_config,
            model=model,
            sanitizer=sanitizer,
        )

    def predict(
        self, df: pd.DataFrame, ra_ctx: Optional[AnalysisContext] = None
    ) -> pd.Series:
        """
        Predicts the label of the given feature.

        Args:
            feature (pd.DataFrame): The feature to predict.
            disable_sanitizer (bool, optional): Whether to disable sanitizer. Defaults to False.

        Returns:
            pd.Series: The predicted label.
        """
        model_feature_cols = set(self.inference_model_config.feature_columns)
        input_df_cols = set(df.columns)

        if not model_feature_cols.issubset(input_df_cols):
            raise ValueError(
                f"Model feature columns are {model_feature_cols} which is not a subset of input feature columns: {input_df_cols}."
            )

        model_input_df = preprocessing(
            df, self.sanitizer, self.inference_model_config.feature_columns
        )[self.inference_model_config.feature_columns]
        pred = self.model.predict(model_input_df)
        assert isinstance(pred, pd.Series)

        if DISABLE_SANITIZER_FLAG:
            return pred

        # Below is the code for comparative analysis and relevance analysis for
        # getting the experimental results in the paper.
        caller_func = inspect.stack()[1].function
        if caller_func not in _QUERY_EXECUTION_FUNC_NAMES:
            logger.info(
                f"Caller function: {caller_func} not in execute_query, skip analysis."
            )
            return pred

        # when sanitizer is enabled but not loaded, raise error
        if self.sanitizer is None:
            raise ValueError("Sanitizer is not loaded.")

        if SAN_COMPARATIVE_ANALYSIS_FLAG:
            original_model_input_df = df[self.inference_model_config.feature_columns]
            original_pred = self.model.predict(original_model_input_df)
            rectified_noise_injected_num = (
                (model_input_df != original_model_input_df).any(axis=1).sum()
            )
            comparative_analysis(
                df=df,
                rectified_noise_injected_num=rectified_noise_injected_num,
                pred=pred,
                original_pred=original_pred,
                label_column=self.inference_model_config.label_column,
                ra_ctx=ra_ctx,
            )
            sanitizer_alert = None
        else:
            sanitizer_alert = self.sanitizer.evaluate_df(df)

            logger.info(
                f"Sanitizer found {sanitizer_alert.sum()} errors out of {len(sanitizer_alert)} rows of data."
            )

            relevance_analysis(
                df=df,
                pred=pred,
                sanitizer_alert=sanitizer_alert,
                label_column=self.inference_model_config.label_column,
                ctx=ra_ctx,
            )

        return postprocessing(pred, sanitizer_alert)
