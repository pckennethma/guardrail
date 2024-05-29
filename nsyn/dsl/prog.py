import copy
from typing import Any, Dict, List, Tuple

import dask.dataframe as dd
import pandas as pd

from nsyn.dsl.stmt import DSLStmt
from nsyn.util.base_model import BaseModel


class DSLProg(BaseModel):
    """
    A class representing a Domain-Specific Language (DSL) program in the NSYN framework.

    This class encapsulates a collection of DSL statements (DSLStmt) and provides functionalities to add statements to the program and to compute overall metrics like coverage.

    Attributes:
        stmts (List[DSLStmt]): A list of DSLStmt instances that make up the program.

    Methods:
        add_stmt: Adds a DSLStmt instance to the program.
        coverage: A property that calculates and returns the average coverage of all statements in the program.
        __str__: Provides a string representation of the DSL program.
    """

    stmts: List[DSLStmt] = []

    def add_stmt(self, stmt: DSLStmt) -> None:
        """
        Adds a DSLStmt instance to the DSL program.

        Args:
            stmt (DSLStmt): The DSL statement to be added to the program.
        """
        self.stmts.append(stmt)

    @property
    def coverage(self) -> float:
        """
        Calculates the average coverage of all the statements in the program.

        Coverage for a statement is a measure of how much of the data the statement accounts for. This method computes the average coverage across all statements in the program.

        Returns:
            float: The average coverage of the statements in the program. Returns 0 if there are no statements.
        """
        return (
            sum(stmt.coverage for stmt in self.stmts) / len(self.stmts)
            if self.stmts
            else 0
        )

    @property
    def statistics(self) -> str:
        """
        Provides a string representation of the statistics of the DSL program.

        Returns:
            str: A string representation of the DSL program statistics, composed of the string representations of its statements, each on a new line.
        """

        stmt_num = len(self.stmts)
        branch_num = sum(stmt.cardinality for stmt in self.stmts)
        coverage = self.coverage

        return (
            f"# Statements: {stmt_num}\n# Branches: {branch_num}\nCoverage: {coverage}"
        )

    def evaluate(self, input_row: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Evaluates the DSL program on a given row of data.

        Args:
            input_row (Dict[str, Any]): A dictionary representing a row of data.

        Returns:
            Tuple[Dict[str, Any], bool]: A tuple containing the expected row and a boolean indicating whether the row is an error.
        """
        expected_row = input_row.copy()
        for stmt in self.stmts:
            expected_row.update(stmt.evaluate(input_row))
        return expected_row, expected_row != input_row

    def get_used_stmts(self, input_row: Dict[str, Any]) -> List[DSLStmt]:
        """
        Returns the list of DSL statements that are used in evaluating a given row of data.

        Args:
            input_row (Dict[str, Any]): A dictionary representing a row of data.

        Returns:
            List[DSLStmt]: A list of DSLStmt instances that are used in evaluating the given row of data.
        """
        used_stmts = []
        for stmt in self.stmts:
            result = stmt.evaluate(input_row)
            if result:
                tmp_dict = copy.deepcopy(input_row)
                tmp_dict.update(result)
                if tmp_dict != input_row:
                    used_stmts.append(stmt)
        return used_stmts

    def evaluate_df(self, df: pd.DataFrame, worker_num: int = 4) -> pd.Series:
        """
        Evaluates the DSL program on a given DataFrame.

        Args:
            df (pd.DataFrame): A DataFrame to evaluate the DSL program on.

        Returns:
            pd.Series: A Series containing whether each row in the DataFrame mismatches the expected row.
        """
        ddf = dd.from_pandas(df, npartitions=worker_num)

        def apply_row(row: pd.Series) -> bool:
            return self.evaluate(row.to_dict())[1]

        def apply_df(df: pd.DataFrame) -> pd.Series:
            return df.apply(apply_row, axis=1)

        res = ddf.map_partitions(apply_df).compute(schedule="processes")
        assert isinstance(res, pd.Series)
        return res

    def get_expected_df(
        self,
        df: pd.DataFrame,
        worker_num: int = 4,
    ) -> pd.DataFrame:
        """
        Computes the expected DataFrame from the given DataFrame.

        Args:
            df (pd.DataFrame): A DataFrame to compute the expected DataFrame from.
            worker_num (int, optional): The number of workers to use. Defaults to 4.

        Returns:
            pd.DataFrame: The expected DataFrame.
        """

        ddf = dd.from_pandas(df, npartitions=worker_num)

        def apply_row(row: pd.Series) -> pd.Series:
            return pd.Series(self.evaluate(row.to_dict())[0])

        def apply_df(df: pd.DataFrame) -> pd.DataFrame:
            return df.apply(apply_row, axis=1)

        res = ddf.map_partitions(apply_df).compute(schedule="processes")
        assert isinstance(res, pd.DataFrame)
        return res

    def __str__(self) -> str:
        """
        Provides a string representation of the DSL program.

        Returns:
            str: A string representation of the DSL program, composed of the string representations of its statements, each on a new line.
        """
        if not self.stmts:
            return "Empty program"
        return "\n" + "\n".join(str(stmt) for stmt in self.stmts)
