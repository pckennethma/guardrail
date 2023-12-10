from typing import Any, Dict, List, Tuple

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

    def __str__(self) -> str:
        """
        Provides a string representation of the DSL program.

        Returns:
            str: A string representation of the DSL program, composed of the string representations of its statements, each on a new line.
        """
        if not self.stmts:
            return "Empty program"
        return "\n" + "\n".join(str(stmt) for stmt in self.stmts)
