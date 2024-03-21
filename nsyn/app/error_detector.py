import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

import numpy as np
import pandas as pd

from nsyn.app.error_detector_util.error import RowError
from nsyn.dsl.prog import DSLProg
from nsyn.util.base_model import BaseModel
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.app.error_detector")


def partition_dataframe(df: pd.DataFrame, num_partitions: int) -> List[pd.DataFrame]:
    """
    Partitions a DataFrame into a list of DataFrames.

    Args:
        df (pd.DataFrame): The DataFrame to partition.
        num_partitions (int): The number of partitions to create.

    Returns:
        List[pd.DataFrame]: A list of DataFrames.
    """
    partition_size = len(df) // num_partitions
    return [df.iloc[i : i + partition_size] for i in range(0, len(df), partition_size)]


class ErrorDetector(BaseModel):
    """
    A class to detect errors in a DataFrame using a DSL program.

    This class encapsulates the logic for detecting errors in a DataFrame using a DSL program. It uses the DSL program to evaluate each row of the DataFrame and returns a list of errors.

    Attributes:
        data (pd.DataFrame): The DataFrame to check for errors.
        program (DSLProg): The DSL program to use for error detection.
        worker_num (int): The number of workers to use for parallel processing.

    Methods:
        process_row: Evaluates a row of data using the DSL program.
        run_parallel: Runs the error detection process on a chunk of data.
        run: Runs the error detection process on the entire DataFrame.
    """

    data: pd.DataFrame
    program: DSLProg
    worker_num: int = int(os.getenv("NSYN_NUM_WORKERS", 4))

    def process_row(self, row: pd.Series) -> Optional[RowError]:
        """
        Evaluates a row of data using the DSL program.

        Args:
            row (pd.Series): A row of data from the DataFrame.

        Returns:
            Optional[RowError]: A RowError instance if the row contains an error, None otherwise.
        """
        original_row = row.to_dict()
        expected_row, has_error = self.program.evaluate(original_row)
        if has_error:
            return RowError(
                row_index=row.name,
                original_row=original_row,
                expected_row=expected_row,
            )
        return None

    def run_parallel(self, data_chunk: pd.DataFrame) -> List[RowError]:
        """
        Runs the error detection process on a chunk of data.

        Args:
            data_chunk (pd.DataFrame): A chunk of data from the DataFrame.

        Returns:
            List[RowError]: A list of RowError instances.
        """
        errors = []
        for _, row in data_chunk.iterrows():
            error = self.process_row(row)
            if error is not None:
                errors.append(error)
        return errors

    def run(self) -> List[RowError]:
        """
        Runs the error detection process on the entire DataFrame.

        Returns:
            List[RowError]: A list of RowError instances.
        """
        num_cores = np.min([len(self.data), self.worker_num])
        data_partitions = partition_dataframe(self.data, num_cores)
        errors = []
        with ProcessPoolExecutor() as executor:
            futures = executor.map(self.run_parallel, data_partitions)
            for future in futures:
                errors.extend(future)
        return errors


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", "-d", type=str, required=True)
    parser.add_argument("--prog-path", "-p", type=str, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    with open(args.prog_path, "rb") as f:
        prog: DSLProg = pickle.load(f)

    alerts = prog.evaluate_df(data)
    actual_error = data["_nsyn_noisy_injected"]
    # compute metrics
    tp = np.sum((alerts == 1) & (actual_error == 1))
    tn = np.sum((alerts == 0) & (actual_error == 0))
    fp = np.sum((alerts == 1) & (actual_error == 0))
    fn = np.sum((alerts == 0) & (actual_error == 1))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    logger.info(f"Precision: {precision:.2f}")
    logger.info(f"Recall: {recall:.2f}")
    logger.info(f"F1: {f1:.2f}")
    logger.info(f"Accuracy: {accuracy:.2f}")
    logger.info(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    logger.info(f"Total error: {np.sum(actual_error)}")
    logger.info(f"Detected error: {np.sum(alerts)}")
