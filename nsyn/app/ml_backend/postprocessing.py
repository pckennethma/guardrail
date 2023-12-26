import pandas as pd

from nsyn.util.flags import ERROR_HANDLING_FLAG
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.app.ml_backend.postprocessing")


def postprocessing(
    pred: pd.Series,
    sanitizer_alert: pd.Series,
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
    total_errors = sanitizer_alert.sum()
    if ERROR_HANDLING_FLAG == "ignore":
        if total_errors > 0:
            logger.warning(
                f"Sanitizer alert found {total_errors} data errors out of {len(sanitizer_alert)} rows of data."
            )
        return pred
    elif ERROR_HANDLING_FLAG == "raise":
        if total_errors > 0:
            raise ValueError(
                f"Sanitizer alert found {total_errors} data errors out of {len(sanitizer_alert)} rows of data."
            )
        return pred
    elif ERROR_HANDLING_FLAG == "coerce":
        pred.loc[sanitizer_alert] = None
        logger.warning(
            f"Sanitizer alert found {total_errors} data errors out of {len(sanitizer_alert)} rows of data. Coercing them to None. This may cause downstream errors. Resulting distribution: {pred.value_counts(dropna=False).to_dict()}"
        )
        return pred
    else:
        raise ValueError(
            f"Unknown error handling flag {ERROR_HANDLING_FLAG}. Supported flags are 'ignore', 'raise', and 'coerce'."
        )
