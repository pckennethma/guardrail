import pandas as pd

from nsyn.dataset.loader import _DATASET_PATH, load_data_by_name
from nsyn.util.logger import get_logger

logger = get_logger("nsyn.tane.non_mec_enum")


def count_non_trivial_single_attribute_fds(df: pd.DataFrame) -> int:
    """
    Given a DataFrame, compute the number of possible non-trivial functional
    dependencies of the form X -> A, where A is a single attribute.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.

    Returns
    -------
    int
        The number of non-trivial single-attribute FDs.
    """
    n = len(df.columns)  # Number of attributes
    if n < 1:
        return 0  # No attributes means no FDs
    # Apply the formula: n * (2^(n-1) - 1)
    return n * ((2 ** (n - 1)) - 1)


def scientific_expr(num: int) -> str:
    """
    Given a number, return the scientific notation of the number.

    Parameters
    ----------
    num : int
        The input number.

    Returns
    -------
    str
        The scientific notation of the number (e.g., 1.23×10^4).
    """
    return "{:.2e}".format(num)


if __name__ == "__main__":
    for name in _DATASET_PATH:
        df = load_data_by_name(name)
        num_fds = count_non_trivial_single_attribute_fds(df)
        logger.info(f"Dataset: {name}")
        logger.info(
            f"Number of non-trivial single-attribute FDs: {scientific_expr(num_fds)}"
        )
