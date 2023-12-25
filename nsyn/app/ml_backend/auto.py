from typing import Literal

from nsyn.app.ml_backend.autogluon import AutoGluonModel
from nsyn.app.ml_backend.base import InferenceModel


def get_inference_model(
    model_path: str,
    model_type: Literal["autogluon", "llm"],
) -> InferenceModel:
    if model_type == "autogluon":
        return AutoGluonModel.create(model_path)
    elif model_type == "llm":
        raise NotImplementedError("LLM is not implemented yet.")
    else:
        raise ValueError(f"Model type {model_type} is not supported.")
