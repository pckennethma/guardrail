from __future__ import annotations

from typing import Any, Dict, Hashable, Iterable, List, Optional

import pandas as pd
from z3 import (
    And,
    ArithRef,
    Bool,
    BoolRef,
    If,
    Implies,
    Int,
    IntVal,
    ModelRef,
    Not,
    Optimize,
    Or,
    Sum,
    sat,
)

from nsyn.dataset.loader import load_data_by_name
from nsyn.dsl.assign import DSLAssign
from nsyn.dsl.branch import DSLBranch
from nsyn.dsl.condition import DSLCondition
from nsyn.dsl.prog import DSLProg
from nsyn.dsl.stmt import DSLStmt
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.maxsmt.solver_z3")

_MAX_BRANCHES = 4
_MAX_FILTERS = 2


class SymbolicRow:
    def __init__(self, record: dict[Hashable, Any], columns: Iterable[str]):
        self.record = record
        self.symbolic_records = {k: IntVal(i) for k, i in record.items()}
        self.symbolic_values = [self.symbolic_records[column] for column in columns]

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> List[SymbolicRow]:
        return [
            SymbolicRow(record, df.columns) for record in df.to_dict(orient="records")
        ]


class SymbolicCondition:
    def __init__(self, cid: int) -> None:
        self.filters = [
            {
                "Column_Index": Int(f"cond_{cid}_filter_{i}_column"),
                "Value_Index": Int(f"cond_{cid}_filter_{i}_value"),
                "Enabled": Bool(f"cond_{cid}_filter_{i}_enabled"),
            }
            for i in range(_MAX_FILTERS)
        ]

    def validity_constraint(self) -> BoolRef:
        return And(
            [
                Implies(
                    And(self.filters[i]["Enabled"], self.filters[j]["Enabled"]),
                    self.filters[i]["Column_Index"] != self.filters[j]["Column_Index"],
                )
                for i in range(_MAX_FILTERS)
                for j in range(i + 1, _MAX_FILTERS)
            ]
        )

    def unequal_constraint(self, other: SymbolicCondition) -> BoolRef:
        equal = And(
            And(
                [
                    Implies(
                        self.filters[i]["Enabled"],
                        Or(
                            [
                                And(
                                    other.filters[j]["Enabled"],
                                    self.filters[i]["Column_Index"]
                                    == other.filters[j]["Column_Index"],
                                    self.filters[i]["Value_Index"]
                                    == other.filters[j]["Value_Index"],
                                )
                                for j in range(_MAX_FILTERS)
                            ]
                        ),
                    )
                    for i in range(_MAX_FILTERS)
                ]
            ),
            And(
                [
                    Implies(
                        other.filters[i]["Enabled"],
                        Or(
                            [
                                And(
                                    self.filters[j]["Enabled"],
                                    other.filters[i]["Column_Index"]
                                    == self.filters[j]["Column_Index"],
                                    other.filters[i]["Value_Index"]
                                    == self.filters[j]["Value_Index"],
                                )
                                for j in range(_MAX_FILTERS)
                            ]
                        ),
                    )
                    for i in range(_MAX_FILTERS)
                ]
            ),
        )
        return Not(equal)

    def value_range_constraint(
        self, value_mappings: Dict[str, Dict[int, str]], columns: list[str]
    ) -> BoolRef:
        return And(
            [
                Implies(
                    And(
                        f["Enabled"],
                        f["Column_Index"] == idx,
                    ),
                    And(
                        f["Value_Index"] >= 0,
                        f["Value_Index"] <= max(value_mappings[col].keys()),
                    ),
                )
                for idx, col in enumerate(columns)
                for f in self.filters
            ]
            + [
                And(
                    f["Column_Index"] >= 0,
                    f["Column_Index"] < len(columns),
                )
                for f in self.filters
            ]
        )

    def evaluate(self, row: SymbolicRow) -> BoolRef:
        conditions = []
        for f in self.filters:
            enabled = f["Enabled"]
            col_idx = f["Column_Index"]
            val_idx = f["Value_Index"]
            col_conditions = []
            for idx, column_value in enumerate(row.symbolic_values):
                col_match = col_idx == idx
                val_match = column_value == val_idx
                col_conditions.append(Implies(col_match, val_match))
            filter_condition = Implies(enabled, And(col_conditions))
            conditions.append(filter_condition)
        return And(conditions)

    def filter_count(self) -> ArithRef:
        return Sum([If(f["Enabled"], 1, 0) for f in self.filters])


class SymbolicBranch:
    def __init__(self, bid: int, target_column: str):
        self.condition = SymbolicCondition(bid)
        self.assignment = Int(f"branch_{bid}_assignment")
        self.enabled = Bool(f"branch_{bid}_enabled")
        self.target_column = target_column

    def evaluate(self, row: SymbolicRow) -> BoolRef:
        cond_match = self.condition.evaluate(row)
        value_match = self.assignment == row.symbolic_records[self.target_column]
        return Implies(And(self.enabled, cond_match), value_match)

    def unequal_constraint(self, other: SymbolicBranch) -> BoolRef:
        return self.condition.unequal_constraint(other.condition)

    def assignment_value_range_constraint(
        self, value_mappings: Dict[str, Dict[int, str]]
    ) -> BoolRef:
        return Implies(
            self.enabled,
            And(
                self.assignment >= 0,
                self.assignment <= max(value_mappings[self.target_column].keys()),
            ),
        )


class SymbolicStatement:
    def __init__(self, target_column: str, sid: int):
        self.enabled = Bool(f"statement_{sid}_enabled")
        self.target_column = target_column
        self.branches = [
            SymbolicBranch(sid * _MAX_BRANCHES + i, target_column)
            for i in range(_MAX_BRANCHES)
        ]

    def condition_size_alignment(self) -> BoolRef:
        return And(
            [
                self.branches[i].condition.filter_count()
                == self.branches[i + 1].condition.filter_count()
                for i in range(_MAX_BRANCHES - 1)
            ]
        )

    def non_overlapping_branches(self) -> BoolRef:
        return And(
            [
                # If both branches are enabled, they should not have the same condition
                Implies(
                    And(self.branches[i].enabled, self.branches[j].enabled),
                    self.branches[i].unequal_constraint(self.branches[j]),
                )
                for i in range(_MAX_BRANCHES)
                for j in range(i + 1, _MAX_BRANCHES)
            ]
        )

    def branch_validity(
        self, value_mappings: Dict[str, Dict[int, str]], columns: list[str]
    ) -> BoolRef:
        return And(
            [branch.condition.validity_constraint() for branch in self.branches]
            + [
                branch.condition.value_range_constraint(value_mappings, columns)
                for branch in self.branches
            ]
            + [
                branch.assignment_value_range_constraint(value_mappings)
                for branch in self.branches
            ]
        )

    def coverage(self, rows: List[SymbolicRow]) -> ArithRef:
        """
        Compute the number of rows covered by the statement.
        """

        return If(
            self.enabled,
            Sum(
                [
                    If(And(branch.enabled, branch.condition.evaluate(row)), 1, 0)
                    for branch in self.branches
                    for row in rows
                ]
            ),
            0,
        )

    def loss(self, rows: List[SymbolicRow]) -> List[ArithRef]:
        """
        Compute the loss of each branch in the statement.
        """

        return [
            If(
                self.enabled,
                Sum(
                    [
                        If(
                            And(
                                branch.enabled,
                                branch.condition.evaluate(row),
                                branch.assignment
                                == row.symbolic_records[self.target_column],
                            ),
                            0,
                            1,
                        )
                        for row in rows
                    ]
                ),
                0,
            )
            for branch in self.branches
        ]


class SymbolicProgram:
    @staticmethod
    def preprocess(
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, Dict[str, Dict[int, str]]]:
        """
        Converts a pandas DataFrame by dropping numeric columns and converting
        categorical values to integers.

        Args:
            data (pd.DataFrame): The input DataFrame to be converted.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
            Dict[str, Dict[int, str]]: A dictionary containing the mapping of column names to the original values.
        """

        if data.shape[1] > 5 and data.shape[0] > 400:
            threshold = max(15, data.shape[0] // 1000)
            data = data.drop(
                columns=[
                    col for col in data.columns if len(data[col].unique()) > threshold
                ]
            )

        data = data.astype("category")

        value_mappings: Dict[str, Dict[int, str]] = {}
        for col in data.columns:
            value_mappings[col] = dict(enumerate(data[col].cat.categories))

        data = data.apply(lambda x: x.cat.codes)

        return data, value_mappings

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.raw_df = load_data_by_name(dataset_name)
        self.df, self.value_mappings = SymbolicProgram.preprocess(self.raw_df)
        self.statements = [
            SymbolicStatement(column, i) for i, column in enumerate(self.df.columns)
        ]
        self.rows = SymbolicRow.from_df(self.df)
        self.enabled_statement_num = Sum(
            [If(statement.enabled, 1, 0) for statement in self.statements]
        )

    def coverage(self) -> ArithRef:
        # average coverage
        return If(
            self.enabled_statement_num == 0,
            0,
            Sum([statement.coverage(self.rows) for statement in self.statements])
            / self.enabled_statement_num,
        )

    def optimize(self) -> Optional[DSLProg]:
        opt = Optimize()
        opt.set("timeout", 3600_000 * 24)
        loss_threshold = int(0.01 * len(self.rows))
        logger.info(f"Loss threshold: {loss_threshold}")
        for idx, stmt in enumerate(self.statements):
            opt.assert_and_track(
                Implies(stmt.enabled, stmt.condition_size_alignment()),
                f"condition_size_alignment_{idx}",
            )
            opt.assert_and_track(
                Implies(stmt.enabled, stmt.non_overlapping_branches()),
                f"non_overlapping_branches_{idx}",
            )
            opt.assert_and_track(
                Implies(
                    stmt.enabled,
                    stmt.branch_validity(
                        self.value_mappings, self.df.columns.to_list()
                    ),
                ),
                f"branch_validity_{idx}",
            )
            for loss in stmt.loss(self.rows):
                opt.add_soft(Implies(stmt.enabled, loss <= loss_threshold))
        opt.assert_and_track(self.enabled_statement_num > 0, "enabled_statement_num")
        opt.maximize(self.coverage())

        rlt = opt.check()
        logger.info(rlt)
        logger.info(opt.statistics())

        if rlt == sat:
            model = opt.model()
            try:
                return self.model_to_dsl(model)
            except Exception as e:
                logger.error(model)
                raise e
        else:
            model = {}
            logger.info("No solution found")
            # unsat core
            logger.info(opt.unsat_core())
            return None

    def model_to_dsl(self, model: ModelRef) -> DSLProg:
        prog = DSLProg()
        for idx, stmt in enumerate(self.statements):
            if model[stmt.enabled]:
                dependent = self.df.columns[idx]
                determinants = set()
                branches = []
                for branch in stmt.branches:
                    if model[branch.enabled]:
                        predicates = []
                        for f in branch.condition.filters:
                            if model[f["Enabled"]]:
                                determinant = self.df.columns[
                                    model[f["Column_Index"]].as_long()
                                ]
                                predicates.append(
                                    (
                                        determinant,
                                        self.value_mappings[determinant][
                                            model[f["Value_Index"]].as_long()
                                        ],
                                    )
                                )
                        if predicates:
                            dsl_branch = DSLBranch(
                                condition=DSLCondition(predicates=predicates),
                                assign=DSLAssign(
                                    variable=dependent,
                                    value=self.value_mappings[dependent][
                                        model[branch.assignment].as_long()
                                    ],
                                ),
                            )
                            branches.append(dsl_branch)
                            determinants.update([p[0] for p in predicates])
                if len(branches) > 0:
                    prog.add_stmt(
                        DSLStmt(
                            determinants=list(determinants),
                            dependent=dependent,
                            branches=branches,
                        )
                    )
        return prog


if __name__ == "__main__":
    program = SymbolicProgram("blood-transfusion-service-center")
    prog = program.optimize()
    logger.info(prog)
