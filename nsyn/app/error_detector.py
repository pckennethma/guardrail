from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Hashable, List, Optional

import numpy as np
import pandas as pd

from nsyn.dsl.prog import DSLProg
from nsyn.util.base_model import BaseModel
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.app.error_detector")

# ANSI escape sequences for printing in color
RED = "\033[91m"
GREEN = "\033[92m"
END = "\033[0m"


class RowError(BaseModel):
    """
    A class to represent an error in a row of data.

    Attributes:
        row_index (Hashable): The index of the row in the original DataFrame.
        original_row (Dict[str, Any]): The original row of data.
        expected_row (Dict[str, Any]): The expected row of data.

    Methods:
        __str__: Provides a string representation of the error.
    """

    row_index: Hashable
    original_row: Dict[str, Any]
    expected_row: Dict[str, Any]

    def __str__(self) -> str:
        """
        Use ANSI escape sequences to highlight the differences between the original and expected rows.
        """
        output = f"Index {self.row_index}:\n"
        for key in self.original_row:
            if self.original_row[key] != self.expected_row[key]:
                output += f"{key}: {RED}{self.original_row[key]}{END} -> {GREEN}{self.expected_row[key]}{END}\n"
        return output


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
    worker_num: int = 4

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
