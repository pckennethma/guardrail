from typing import Any, Dict, Hashable

from nsyn.util.base_model import BaseModel
from nsyn.util.logger import END, GREEN, RED


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


class FeatureError(BaseModel):
    """
    A class to represent an error in the feature to be predicted.

    Attributes:
        row_index (Hashable): The index of the row in the original DataFrame.
        original_feature (Dict[str, Any]): The original feature.
        expected_row (Dict[str, Any]): The expected feature.

    Methods:
        __str__: Provides a string representation of the error.
    """

    row_index: Hashable
    original_feature: Dict[str, Any]
    expected_feature: Dict[str, Any]

    def __str__(self) -> str:
        """
        Use ANSI escape sequences to highlight the differences between the original and expected features.
        """
        output = f"Index {self.row_index}:\n"
        for key in self.original_feature:
            if self.original_feature[key] != self.expected_feature[key]:
                output += f"{key}: {RED}{self.original_feature[key]}{END} -> {GREEN}{self.expected_feature[key]}{END}\n"
        return output
