from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Literal, Optional

import pandas as pd

from nsyn.app.ml_backend.relevance_analysis import RelevanceAnalysisContext
from nsyn.dsl.prog import DSLProg
from nsyn.util.base_model import BaseModel


class BaseModelConfig(BaseModel):
    """
    A class to represent the configuration for a model.
    """

    model_path: str
    label_column: str
    feature_columns: List[str]
    model_output_type: Literal["discrete", "continuous"]


class InferenceModel(ABC, BaseModel):
    inference_model_config: BaseModelConfig
    sanitizer: Optional[DSLProg]

    @classmethod
    @abstractmethod
    def create(cls, model_path: str) -> InferenceModel:
        ...

    @abstractmethod
    def predict(
        self, df: pd.DataFrame, ra_ctx: Optional[RelevanceAnalysisContext] = None
    ) -> pd.Series:
        ...
