from typing import List, Optional

import pandas as pd

from nsyn.dsl.prog import DSLProg
from nsyn.util.flags import ERROR_HANDLING_FLAG
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.app.ml_backend.error_handling")


def preprocessing(
    df: pd.DataFrame,
    sanitizer: Optional[DSLProg],
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    A helper function to preprocess the input data before feeding it to the model.

    Args:
        model_input_df (pd.DataFrame): The input data.
        sanitizer (Optional[DSLProg]): The sanitizer.

    Returns:
        pd.DataFrame: The preprocessed input data.
    """

    if ERROR_HANDLING_FLAG == "rectify":
        if sanitizer is not None:
            logger.info("Rectifying data...")
            rectified_df = df.copy()
            rectified_df[feature_columns] = sanitizer.get_expected_df(df)[
                feature_columns
            ]
            logger.info("Rectification done.")
            return rectified_df
        else:
            raise ValueError("Sanitizer is None but error handling flag is rectify.")

    return df


def postprocessing(
    pred: pd.Series,
    sanitizer_alert: Optional[pd.Series],
) -> pd.Series:
    """
    A helper function to postprocess the prediction under different error handling settings.

    Args:
        pred (pd.Series): The prediction.
        sanitizer_alert (pd.Series): The sanitizer alert.

    Returns:
        pd.Series: The postprocessed prediction.
    """
    logger.info(f"Error handling flag: {ERROR_HANDLING_FLAG}")
    total_errors = sanitizer_alert.sum() if sanitizer_alert is not None else -1
    if ERROR_HANDLING_FLAG == "ignore":
        assert (
            sanitizer_alert is not None
        ), "Sanitizer alert is None but error handling flag is ignore."
        if total_errors > 0:
            logger.warning(
                f"Sanitizer alert found {total_errors} data errors out of {len(sanitizer_alert)} rows of data."
            )
        return pred
    elif ERROR_HANDLING_FLAG == "raise":
        assert (
            sanitizer_alert is not None
        ), "Sanitizer alert is None but error handling flag is raise."
        if total_errors > 0:
            raise ValueError(
                f"Sanitizer alert found {total_errors} data errors out of {len(sanitizer_alert)} rows of data."
            )
        return pred
    elif ERROR_HANDLING_FLAG == "coerce":
        assert (
            sanitizer_alert is not None
        ), "Sanitizer alert is None but error handling flag is coerce."
        pred.loc[sanitizer_alert] = None
        logger.warning(
            f"Sanitizer alert found {total_errors} data errors out of {len(sanitizer_alert)} rows of data. Coercing them to None. This may cause downstream errors. Resulting distribution: {pred.value_counts(dropna=False).to_dict()}"
        )
        return pred
    elif ERROR_HANDLING_FLAG == "rectify":
        logger.info("No postprocessing needed for rectify error handling.")
        return pred
    else:
        raise ValueError(
            f"Unknown error handling flag {ERROR_HANDLING_FLAG}. Supported flags are 'ignore', 'raise', and 'coerce'."
        )
