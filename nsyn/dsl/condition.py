from typing import Any, Dict, List, Tuple

from nsyn.dsl.util import get_keyword_text
from nsyn.util.base_model import BaseModel

AND_str = get_keyword_text("AND")
EQ_str = get_keyword_text("=")


class DSLCondition(BaseModel):
    """
    A class representing a condition in a Domain-Specific Language (DSL) within the NSYN framework.

    This class encapsulates a list of predicates, each representing a condition that must be met. It provides functionality to evaluate these conditions against a given row of data and to represent the condition as a string.

    Attributes:
        predicates (List[Tuple[str, Any]]): A list of predicates forming the condition. Each predicate is a tuple with a column name and a value.

    Methods:
        evaluate: Evaluates the condition against a given row of data.
        __str__: Provides a string representation of the condition.
    """

    predicates: List[Tuple[str, Any]]

    def evaluate(self, row: Dict[str, Any]) -> bool:
        """
        Evaluates the condition against a given row of data.

        Args:
            row (Dict[str, Any]): A dictionary representing a row of data, with keys as column names and values as row values.

        Returns:
            bool: True if the row satisfies all the predicates of the condition, False otherwise.
        """
        return all(row[predicate[0]] == predicate[1] for predicate in self.predicates)

    def __str__(self) -> str:
        """
        Provides a string representation of the condition.

        Returns:
            str: A string representation of the condition, composed of its predicates joined by the AND keyword.
        """
        return f" {AND_str} ".join(
            f"{predicate[0]} {EQ_str} {predicate[1]}" for predicate in self.predicates
        )
