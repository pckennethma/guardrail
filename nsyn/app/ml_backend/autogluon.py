from __future__ import annotations

import os
import pickle
from typing import Literal, Optional

import pandas as pd
from autogluon.tabular import TabularPredictor

from nsyn.app.ml_backend.base import BaseModelConfig, InferenceModel
from nsyn.dsl.prog import DSLProg
from nsyn.util.flags import DISABLE_SANITIZER_FLAG, SAN_RELEVANCE_ANALYSIS_FLAG
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.app.ml_backend.autogluon")

_DEFAULT_MODEL_ROOT = "models"


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

    def predict(self, df: pd.DataFrame) -> pd.Series:
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

        model_input_df = df[self.inference_model_config.feature_columns]
        pred = self.model.predict(model_input_df)

        assert isinstance(pred, pd.Series)

        if DISABLE_SANITIZER_FLAG:
            return pred

        # when sanitizer is enabled but not loaded, raise error
        if self.sanitizer is None:
            raise ValueError("Sanitizer is not loaded.")

        sanitizer_alert = self.sanitizer.evaluate_df(df)

        logger.info(
            f"Sanitizer found {sanitizer_alert.sum()} errors out of {len(sanitizer_alert)} rows of data."
        )

        if not SAN_RELEVANCE_ANALYSIS_FLAG:
            return pred

        label = self.inference_model_config.label_column
        if label not in df.columns:
            logger.error(
                f"Label column {label} not found in input data. Cannot perform relevance analysis."
            )
            return pred

        prediction_errors = pred != df[label]
        prediction_error_num = prediction_errors.sum()

        logger.info(
            f"Model mis-predicts {prediction_error_num} out of {len(df)} rows of data."
        )

        if "_nsyn_noisy_injected" not in df.columns:
            noise_injected = pd.Series([False] * len(df))
        else:
            noise_injected = df["_nsyn_noisy_injected"]
            logger.info(
                f"Found {noise_injected.sum()} rows of data with injected noise."
            )
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
                "noise_injection": noise_injected
            }
        )

        detected_error_num = relevance_df[
            (relevance_df["sanitizer_alert"] == True)  # noqa: E712
            & (relevance_df["prediction_error"] == True)  # noqa: E712
        ].shape[0]

        logger.info(
            f"{detected_error_num} prediction errors are detected by sanitizer."
        )

        logger.info(f"SANITIZER_RELEVANCE_ANALYSIS:\n{relevance_df.corr()}")

        return pred
