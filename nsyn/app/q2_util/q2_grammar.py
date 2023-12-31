from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

from nsyn.util.base_model import BaseModel
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.app.q2_util.q2")

# model type
_T_MODEL_TYPE = Literal["autogluon", "llm"]

# agg func in pandas
_T_AGG_FUNC = Literal["mean", "sum", "count", "min", "max", "std", "var", "median"]

# model
_T_MODEL = Tuple[str, _T_MODEL_TYPE]

CLAUSE_MATCH_PATTERN = re.compile(
    r"""
    SELECT\s+(.*?)\s+FROM\s+(.*?)  # SELECT and FROM clauses
    (?:\s+WHERE\s+(.*?))?         # Optional WHERE clause
    (?:\s+GROUP\s+BY\s+(.*?))?    # Optional GROUP BY clause
    (?:\s+WHERE\s+(.*?))?         # Optional WHERE clause, if it comes after GROUP BY
    $""",
    re.IGNORECASE | re.VERBOSE,
)
MODEL_MATCH_PATTERN = re.compile(r"M(\d+)\s*:\s*([\w-]+),\s*(\w+)")
AGG_PROJECTION_MODEL_MATCH_PATTERN = re.compile(r"(\w+)\(M(\d+)\)")
RAW_PROJECTION_MODEL_MATCH_PATTERN = re.compile(r"(?<!\w\()M(\d+)")
WHERE_MODEL_MATCH_PATTERN = re.compile(
    r"(\bM\d+\b)\s*(==|!=|>|<|>=|<=)\s*('[^']*'|\"[^\"]*\"|\d+)"
)
GROUP_BY_MODEL_MATCH_PATTERN = re.compile(r"M(\d+)+")
# example case when pattern: "CASE WHEN M1 == 'yes' THEN 1 ELSE 0 END"
RAW_CASE_WHEN_MODEL_MATCH_PATTERN = re.compile(
    r"CASE WHEN (.+) THEN (\d+) ELSE (\d+) END"
)
AGG_CASE_WHEN_MODEL_MATCH_PATTERN = re.compile(
    r"(\w+)\(CASE WHEN (.+) THEN (\d+) ELSE (\d+) END\)"
)


def _fix_agg_func(agg_func: str) -> _T_AGG_FUNC:
    """
    Fix the aggregation function.

    Args:
        agg_func (str): The aggregation function.

    Returns:
        _T_AGG_FUNC: The fixed aggregation function.
    """
    agg_func = agg_func.lower()
    if agg_func == "avg":
        logger.warning(
            "The aggregation function `avg` is not a valid aggregation function in pandas. Use `mean` instead."
        )
        return "mean"
    else:
        assert agg_func in [
            "mean",
            "sum",
            "count",
            "min",
            "max",
            "std",
            "var",
            "median",
        ]
        return cast(_T_AGG_FUNC, agg_func)


class CaseWhen(BaseModel):
    model: _T_MODEL
    condition: str

    true_outcome: int = 1
    false_outcome: int = 0

    @classmethod
    def create(
        cls,
        raw_model_and_condition: str,
        raw_true_outcome: str,
        raw_false_outcome: str,
        model_info: Dict[int, _T_MODEL],
    ) -> CaseWhen:
        case_when_condition_match = cast(
            List[List[str]], WHERE_MODEL_MATCH_PATTERN.findall(raw_model_and_condition)
        )
        assert (
            len(case_when_condition_match) == 1
        ), "Must have one condition in case when clause."
        raw_model_num, op, val = case_when_condition_match[0]
        model_num = int(raw_model_num[1:])
        condition = f"{op} {val}"
        true_outcome, false_outcome = int(raw_true_outcome), int(raw_false_outcome)
        return cls(
            model=model_info[model_num],
            condition=condition,
            true_outcome=true_outcome,
            false_outcome=false_outcome,
        )

    # def __str__(self) -> str:
    #     return f"CASE WHEN {self.model[0]} [predicted] {self.condition} THEN {self.true_outcome} ELSE {self.false_outcome} END"

    def get_name(self, model_label: str) -> str:
        return f"CASE WHEN {model_label} [predicted] {self.condition} THEN {self.true_outcome} ELSE {self.false_outcome} END"


class LegacyCaseWhen(BaseModel):
    column: str
    condition: str

    true_outcome: int = 1
    false_outcome: int = 0

    @classmethod
    def create(
        cls,
        dataset_name: str,
        raw_column_and_condition: str,
        raw_true_outcome: str,
        raw_false_outcome: str,
    ) -> LegacyCaseWhen:
        legacy_selection_filter_matches = cast(
            List[List[str]],
            re.findall(
                rf"{dataset_name}\.(\w+)\s*(==|!=|>|<|>=|<=)\s*('[^']*'|\"[^\"]*\"|\d+)",
                raw_column_and_condition,
            ),
        )
        assert (
            len(legacy_selection_filter_matches) == 1
        ), "Must have one condition in case when clause."
        col, op, val = legacy_selection_filter_matches[0]

        condition = f"{op} {val}"
        true_outcome, false_outcome = int(raw_true_outcome), int(raw_false_outcome)
        return cls(
            column=col,
            condition=condition,
            true_outcome=true_outcome,
            false_outcome=false_outcome,
        )

    # def __str__(self) -> str:
    #     return f"CASE WHEN {self.column} {self.condition} THEN {self.true_outcome} ELSE {self.false_outcome} END"

    def get_name(self) -> str:
        return f"CASE WHEN {self.column} {self.condition} THEN {self.true_outcome} ELSE {self.false_outcome} END"


class Query2(BaseModel):
    """
    This class represents the query 2.0 of input sanitizer.

    Three types of Query 2.0 are supported:
    Q1: SELECT AVG(M1) FROM R
    Q2: SELECT COUNT(R.*) FROM R WHERE M1 > 0.5
    Q3: SELECT COUNT(*) FROM R GROUP BY M.predict(R)

    Composite form of above queries are also supported:
    Q4: SELECT AVG(M_1.predict(R)) FROM R WHERE M_2.predict(R) GROUP BY M_3.predict(R)

    Attributes:
        dataset_name (str): The name of the dataset.
        projection_model_path (str): The path to the model for projection.
        projection_model_type (str): The type of the model for projection.
        projection_model_agg_func (str): The aggregation function for projection.
        selection_model_path (str): The path to the model for selection.
        selection_model_type (str): The type of the model for selection.
        selection_model_condition (str): The condition for selection.
        group_by_model_path (str): The path to the model for group by.
        group_by_model_type (str): The type of the model for group by.
        legacy_projection (Tuple[str, str]): The projection in standard SQL query.
        legacy_selection_filters (List[Tuple[str, Any]]): The selection filters in standard SQL query.
        legacy_group_by_column (str): The group by column in standard SQL query.
    """

    dataset_name: str
    dataset_version: Optional[str]

    # if the model is not specified, the corresponding model path and model type should be None
    # and, the `legacy_projection` should be specified
    projection_model: Optional[_T_MODEL]
    projection_model_with_case_when: Optional[CaseWhen]
    projection_model_agg_func: Optional[_T_AGG_FUNC]

    selection_models: Optional[List[_T_MODEL]]
    # the condition for selection model, for example, `WHERE M.predict(R) > 0.5`
    # here, we will use np.safe_eval to evaluate the condition as f"{model_output} {condition}"
    selection_model_conditions: Optional[List[str]]

    group_by_model: Optional[_T_MODEL]
    # for some models that output a continuous value, we need to discretize the output to do group by
    # for example, `GROUP BY M.predict(R) > 0.5`
    group_by_partition_threshold: Optional[float]

    legacy_projection_column: Optional[str]
    legacy_projection_agg_func: Optional[_T_AGG_FUNC]
    legacy_projection_with_case_when: Optional[LegacyCaseWhen]
    legacy_selection_filters: Optional[List[Tuple[str, Any]]]
    legacy_group_by_column: Optional[str]

    main_query: str
    raw_model_info: str
    user_friendly_interpretation: Optional[str] = None

    @classmethod
    def parse_query(cls, query: str) -> "Query2":
        """
        The method to parse the query.

        Args:
            query (str): The query to parse.

        Returns:
            Query2: The parsed query.
        """

        # initialize all attributes to None
        dataset_name: str
        projection_model: Optional[_T_MODEL] = None
        projection_model_agg_func: Optional[_T_AGG_FUNC] = None
        projection_model_with_case_when: Optional[CaseWhen] = None
        selection_models: Optional[List[_T_MODEL]] = None
        selection_model_conditions: Optional[List[str]] = None
        group_by_model: Optional[_T_MODEL] = None
        group_by_partition_threshold: Optional[float] = None
        legacy_projection_column: Optional[str] = None
        legacy_projection_agg_func: Optional[_T_AGG_FUNC] = None
        legacy_selection_filters: Optional[List[Tuple[str, Any]]] = None
        legacy_group_by_column: Optional[str] = None
        legacy_projection_with_case_when: Optional[LegacyCaseWhen] = None

        # Extract and process model information
        query = query.strip()
        main_query = query.split("\n")[0]
        raw_model_info = query[len(main_query) :].strip()
        model_matches = MODEL_MATCH_PATTERN.findall(raw_model_info)
        model_info: Dict[int, _T_MODEL] = {
            int(model_num): (model_path, model_type)
            for model_num, model_path, model_type in model_matches
        }

        logger.info(f"main query: {main_query}")
        logger.info(f"model info:\n{model_info}")

        clause_match = CLAUSE_MATCH_PATTERN.match(main_query)

        assert (
            clause_match
        ), f"The query: {main_query} is not valid during clause parsing."

        if clause_match:
            (
                select_clause,
                from_clause,
                where_clause1,
                group_by_clause,
                where_clause2,
            ) = clause_match.groups()
            where_clause = where_clause1 or where_clause2
            dataset_name = from_clause
            if "." in dataset_name:
                dataset_name, dataset_version = dataset_name.split(".")
            else:
                dataset_version = None

        # extract model from main query
        # select_clause = main_query.split("SELECT")[1].split("FROM")[0]
        # if "WHERE" in main_query:
        #     where_clause = main_query.split("WHERE")[1]
        #     if "GROUP BY" in where_clause:
        #         where_clause = where_clause.split("GROUP BY")[0]
        # else:
        #     where_clause = ""
        # if "GROUP BY" in main_query:
        #     group_by_clause = main_query.split("GROUP BY")[1]
        #     if "WHERE" in group_by_clause:
        #         group_by_clause = group_by_clause.split("WHERE")[0]
        # else:
        #     group_by_clause = ""

        logger.info(f"select clause: {select_clause}")
        logger.info(f"where clause: {where_clause}")
        logger.info(f"group by clause: {group_by_clause}")

        agg_case_when_match = cast(
            List[List[str]],
            AGG_CASE_WHEN_MODEL_MATCH_PATTERN.findall(select_clause),
        )
        raw_case_when_match = cast(
            List[List[str]],
            RAW_CASE_WHEN_MODEL_MATCH_PATTERN.findall(select_clause),
        )
        agg_proj_model_match = cast(
            List[List[str]],
            AGG_PROJECTION_MODEL_MATCH_PATTERN.findall(select_clause),
        )
        raw_proj_model_match = cast(
            List[List[str]],
            RAW_PROJECTION_MODEL_MATCH_PATTERN.findall(select_clause),
        )
        agg_legacy_proj_match = cast(
            List[List[str]],
            re.findall(rf"(\w+)\({dataset_name}\.([\w-]+|\*)\)", select_clause),
        )
        raw_legacy_proj_match = cast(
            List[List[str]],
            re.findall(rf"(?<!\w\(){dataset_name}\.([\w-]+|\*)(?!\))", select_clause),
        )
        count_all_match = re.findall(r"count\(\*\)", select_clause)

        if agg_case_when_match or raw_case_when_match:
            raw_agg_func: Optional[str] = None
            if agg_case_when_match:
                raw_agg_func = agg_case_when_match[0][0]
                raw_model_or_column_and_condition = agg_case_when_match[0][1]
                raw_true_outcome = agg_case_when_match[0][2]
                raw_false_outcome = agg_case_when_match[0][3]
            elif raw_case_when_match:
                raw_model_or_column_and_condition = raw_case_when_match[0][0]
                raw_true_outcome = raw_case_when_match[0][1]
                raw_false_outcome = raw_case_when_match[0][2]
            else:
                raise ValueError("Should not reach here.")
            if dataset_name in raw_model_or_column_and_condition:
                legacy_projection_with_case_when = LegacyCaseWhen.create(
                    dataset_name=dataset_name,
                    raw_column_and_condition=raw_model_or_column_and_condition,
                    raw_true_outcome=raw_true_outcome,
                    raw_false_outcome=raw_false_outcome,
                )
                if raw_agg_func:
                    legacy_projection_agg_func = _fix_agg_func(raw_agg_func)
            else:
                projection_model_with_case_when = CaseWhen.create(
                    raw_model_and_condition=raw_model_or_column_and_condition,
                    raw_true_outcome=raw_true_outcome,
                    raw_false_outcome=raw_false_outcome,
                    model_info=model_info,
                )
                if raw_agg_func:
                    projection_model_agg_func = _fix_agg_func(raw_agg_func)
        elif agg_proj_model_match or raw_proj_model_match:
            assert not (
                agg_proj_model_match and raw_proj_model_match
            ), f"Should not have both aggregation and raw projection.\n {agg_proj_model_match} \n {raw_proj_model_match}"
            if agg_proj_model_match:
                assert (
                    len(agg_proj_model_match) == 1
                ), "Currently, at most one model can be selected."
                model_num = int(agg_proj_model_match[0][1])
                projection_model_agg_func = _fix_agg_func(agg_proj_model_match[0][0])
            elif raw_proj_model_match:
                assert (
                    len(raw_proj_model_match) == 1
                ), "Currently, at most one model can be selected."
                model_num = int(raw_proj_model_match[0][0])
            else:
                raise ValueError("Should not reach here.")
            projection_model = model_info[model_num]
        elif agg_legacy_proj_match or raw_legacy_proj_match:
            assert not (
                agg_legacy_proj_match and raw_legacy_proj_match
            ), f"Should not have both aggregation and raw projection.\n{agg_legacy_proj_match}\n{raw_legacy_proj_match}"
            if agg_legacy_proj_match:
                assert (
                    len(agg_legacy_proj_match) == 1
                ), "Currently, at most one model can be selected."
                legacy_projection_agg_func = _fix_agg_func(agg_legacy_proj_match[0][0])
                legacy_projection_column = agg_legacy_proj_match[0][1]
            elif raw_legacy_proj_match:
                assert (
                    len(raw_legacy_proj_match) == 1
                ), "Currently, at most one model can be selected."
                legacy_projection_agg_func = None
                legacy_projection_column = raw_legacy_proj_match[0][0]
            else:
                raise ValueError("Should not reach here.")
        elif count_all_match:
            legacy_projection_agg_func = "count"
            legacy_projection_column = "*"
        else:
            raise ValueError("Should not reach here.")

        # extract model from where clause
        if where_clause:
            where_model_match = cast(
                List[List[str]], WHERE_MODEL_MATCH_PATTERN.findall(where_clause)
            )
            legacy_selection_filter_matches = cast(
                List[List[str]],
                re.findall(
                    rf"{dataset_name}\.(\w+)\s*(==|!=|>|<|>=|<=)\s*('[^']*'|\"[^\"]*\"|\d+)",
                    where_clause,
                ),
            )
        else:
            where_model_match = []
            legacy_selection_filter_matches = []
        if where_model_match:
            # save in the format of [("M1", "> 0.5"), ("M2", "== 'yes'")]
            selection_models = [
                model_info[int(raw_model_num[1:])]
                for raw_model_num, op, val in where_model_match
            ]
            selection_model_conditions = [
                f"{op} {val}" for _, op, val in where_model_match
            ]

        if legacy_selection_filter_matches:
            # save in the format of [("age", "> 30"), ("edu", "== 'bachelor'")]
            legacy_selection_filters = [
                (col, f"{op} {val}") for col, op, val in legacy_selection_filter_matches
            ]

        # extract model from group by clause
        if group_by_clause:
            group_by_model_match = cast(
                List[str], GROUP_BY_MODEL_MATCH_PATTERN.findall(group_by_clause)
            )
            legacy_group_by_match = cast(
                List[str], re.findall(rf"{dataset_name}\.(\w+)", group_by_clause)
            )
        else:
            group_by_model_match = []
            legacy_group_by_match = []
        if group_by_model_match:
            assert (
                len(group_by_model_match) == 1
            ), "Currently, at most one model can be selected."
            model_num = int(group_by_model_match[0])
            group_by_model = model_info[model_num]
            # extract partition threshold
            # example: GROUP BY M1 ! 0.5
            if "!" in group_by_clause:
                raw_threshold = group_by_clause.split("!")[1].strip()
                # try to convert to float
                try:
                    group_by_partition_threshold = float(raw_threshold)
                except ValueError:
                    pass
        elif legacy_group_by_match:
            assert (
                len(legacy_group_by_match) == 1
            ), "Currently, at most one column can be selected."
            legacy_group_by_column = legacy_group_by_match[0]
        else:
            logger.warning("No group by model or column is specified.")

        return cls(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            projection_model=projection_model,
            projection_model_with_case_when=projection_model_with_case_when,
            projection_model_agg_func=projection_model_agg_func,
            selection_models=selection_models,
            selection_model_conditions=selection_model_conditions,
            group_by_model=group_by_model,
            group_by_partition_threshold=group_by_partition_threshold,
            legacy_projection_column=legacy_projection_column,
            legacy_projection_agg_func=legacy_projection_agg_func,
            legacy_selection_filters=legacy_selection_filters,
            legacy_group_by_column=legacy_group_by_column,
            legacy_projection_with_case_when=legacy_projection_with_case_when,
            main_query=main_query,
            raw_model_info=raw_model_info,
        )
