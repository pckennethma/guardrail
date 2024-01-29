from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

import pandas as pd

from nsyn.dsl.assign import DSLAssign
from nsyn.dsl.branch import DSLBranch
from nsyn.dsl.condition import DSLCondition
from nsyn.util.base_model import BaseModel
from nsyn.util.color import get_keyword_text
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.dsl.stmt")

# Predefine formatted strings for keywords
GIVEN_str = get_keyword_text("GIVEN")
ON_str = get_keyword_text("ON")
HAVING_str = get_keyword_text("HAVING")


class DSLStmt(BaseModel):
    """
    A class representing a Domain-Specific Language (DSL) statement in the NSYN framework.

    This class encapsulates the logic for creating, fitting, and representing a statement in a DSL program. It handles the relationship between determinants and a dependent variable, creating branches of logic based on the input data.

    Attributes:
        determinants (List[str]): The list of determinant variables.
        dependent (str): The dependent variable.
        branches (List[DSLBranch]): A list of branches (rules) derived from the data.
        loss (Optional[int]): The loss metric for the statement, if calculated.
        coverage (int): The coverage metric for the statement, representing how much of the data the statement accounts for.

    Methods:
        create: A class method for creating a DSLStmt instance.
        fit: Fits the DSL statement to the provided data.
        _make_branch: Helper method to create a DSLBranch based on a subset of data.
        evaluate: Evaluates the DSL statement on the provided data. (Not implemented)
        cardinality: Returns the number of branches in the statement.
        dimensions: Returns the number of determinant variables.
        __str__: Provides a string representation of the DSL statement.
    """

    determinants: List[str]
    dependent: str
    branches: List[DSLBranch] = []
    loss: Optional[int] = None
    coverage: int = 0

    @classmethod
    def create(cls, dependent: str, determinants: List[str]) -> DSLStmt:
        """
        Creates an instance of DSLStmt.

        Args:
            dependent (str): The dependent variable in the statement.
            determinants (List[str]): A list of determinant variables.

        Returns:
            DSLStmt: An instance of DSLStmt with the specified dependent and determinants.
        """
        return cls(dependent=dependent, determinants=determinants)

    def fit(self, input_data: pd.DataFrame, epsilon: float, min_support: int) -> None:
        """
        Fits the DSL statement to the provided DataFrame.

        Args:
            input_data (pd.DataFrame): The DataFrame containing the relevant data.
            epsilon (float): A threshold parameter for determining significant branches.

        Raises:
            ValueError: If the determinants and dependent are not in the input data.
            RuntimeError: If the statement is already fit.
        """
        cols = self.determinants + [self.dependent]
        if not set(cols).issubset(input_data.columns):
            raise ValueError("Determinants and dependent must be in input data")

        if self.branches:
            raise RuntimeError("Already fit")

        grouped = input_data[cols].groupby(self.determinants)
        self.branches = [
            branch
            for key, sub_df in grouped
            if (
                (len(sub_df) >= min_support)
                and (
                    branch := self._make_branch(
                        cast(tuple[str, ...], key), sub_df, epsilon, min_support
                    )
                )
                is not None
            )
        ]

    def _make_branch(
        self,
        key: tuple[str, ...],
        sub_df: pd.DataFrame,
        epsilon: float,
        min_support: int,
    ) -> Optional[DSLBranch]:
        """
        Creates a DSLBranch based on a subset of the data.

        Args:
            key (tuple[str, ...]): A tuple representing a specific combination of determinant values.
            sub_df (pd.DataFrame): The subset of the DataFrame corresponding to the key.
            epsilon (float): A threshold parameter for determining significant branches.

        Returns:
            Optional[DSLBranch]: A DSLBranch if it meets the criteria based on epsilon; otherwise, None.
        """
        cond = DSLCondition(
            predicates=[
                (determinant, value)
                for determinant, value in zip(self.determinants, key)
            ]
        )
        counts = sub_df[self.dependent].value_counts().to_dict()
        if len(counts) == 0:
            logger.warning(
                f"Empty counts for key {key} with {len(sub_df)} rows of data"
            )
            return None
        most_common = max(counts, key=counts.get)  # type: ignore
        max_count = counts[most_common]
        if 1 - epsilon < max_count / len(sub_df) and len(sub_df) > min_support:
            self.coverage += len(sub_df)
            return DSLBranch(
                condition=cond,
                assign=DSLAssign(variable=self.dependent, value=most_common),
            )
        else:
            return None

    def evaluate(self, input_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates the DSL statement on the provided row.

        Args:
            input_row (Dict[str, Any]): A dictionary representing a row of data.

        Returns:
            Dict[str, Any]: A dictionary representing the output of the DSL statement.
        """
        for branch in self.branches:
            if branch.condition.evaluate(input_row):
                return {branch.assign.variable: branch.assign.value}
        return {}

    @property
    def cardinality(self) -> int:
        """
        The number of branches in the statement.

        Returns:
            int: The number of branches.

        Raises:
            RuntimeError: If the statement has not been fit yet.
        """
        if self.branches is None:
            raise RuntimeError("Must fit before computing cardinality")
        return len(self.branches)

    @property
    def dimensions(self) -> int:
        """
        The number of determinant variables.

        Returns:
            int: The number of determinant variables.
        """
        return len(self.determinants)

    def compute_loss(self, input_data: pd.DataFrame) -> int:
        """
        Computes the loss metric for the statement.

        Args:
            input_data (pd.DataFrame): The DataFrame containing the relevant data.

        Returns:
            int: The loss metric for the statement.
        """

        if self.loss is None:

            def _evaluate_conditions(row: pd.Series) -> int:
                row_dict = row.to_dict()
                
                matched_branch: Optional[DSLBranch] = None
                for branch in self.branches:
                    if branch.condition.evaluate(row_dict):
                        matched_branch = branch
                        break
                if matched_branch is None:
                    return 0
                else:
                    if row[matched_branch.assign.variable] == matched_branch.assign.value:
                        return 0
                    else:
                        return 1
            self.loss = input_data.apply(_evaluate_conditions, axis=1).sum()
        return self.loss

    def __str__(self) -> str:
        """
        Provides a string representation of the DSL statement.

        Returns:
            str: A string representing the DSL statement.
        """
        determinant_str = ", ".join(self.determinants)
        trim = False
        if len(self.branches) >= 20:
            branches_to_display = self.branches[:20]
            trim = True
        else:
            branches_to_display = self.branches
        branch_str = "\t" + ";\n\t".join(
            [str(branch) for branch in branches_to_display]
        )
        if trim:
            branch_str += f"\n\t... ({len(self.branches) - 20} more branches)"
        return f"{GIVEN_str} {determinant_str} {ON_str} {self.dependent} {HAVING_str}\n{branch_str}"
