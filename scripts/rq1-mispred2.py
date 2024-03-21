import numpy as np
from pydantic.type_adapter import TypeAdapter

from nsyn.app.ml_backend.analysis import RelevanceAnalysisDumpItem
from nsyn.util.logger import get_logger

logger = get_logger(name="scripts.error_vs_misprediction")

dataset_order = [
    "adult",
    "lung_cancer",
    "insurance",
    "bird_strikes",
]
dataset_alias = {
    "adult": "ADULT",
    "lung_cancer": "LC",
    "insurance": "INS",
    "bird_strikes": "BS",
}


def process_file(stat_path: str) -> tuple[dict[str, float], str]:
    logger.info(f"Processing file: {stat_path}")
    adapter = TypeAdapter(RelevanceAnalysisDumpItem)
    with open(stat_path) as reader:
        data_samples = list(map(adapter.validate_json, reader.readlines()))
        logger.info(f"Loaded {len(data_samples)} data samples.")
        logger.info(f"Sample: {data_samples[0]}")

    statistics = {
        "total_pred_error_num": np.mean(
            [item.total_pred_error_num for item in data_samples]
        ),
        "detected_data_error_num": np.mean(
            [item.detected_data_error_num for item in data_samples]
        ),
        "actual_data_error_num": np.mean(
            [item.actual_data_error_num for item in data_samples]
        ),
        "detected_pred_error_num": np.mean(
            [item.detected_pred_error_num for item in data_samples]
        ),
        "total_input_num": np.mean([item.total_input_num for item in data_samples]),
        "falsely_detected_data_error_num": np.mean(
            [item.falsely_detected_data_error_num for item in data_samples]
        ),
    }

    return (
        statistics,
        "na"
        if data_samples[0].ctx is None
        else data_samples[0].ctx.dataset.split(".")[0],
    )
