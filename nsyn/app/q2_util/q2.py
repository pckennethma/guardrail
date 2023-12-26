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
    legacy_selection_filters: Optional[List[Tuple[str, Any]]]
    legacy_group_by_column: Optional[str]

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
        selection_models: Optional[List[_T_MODEL]] = None
        selection_model_conditions: Optional[List[str]] = None
        group_by_model: Optional[_T_MODEL] = None
        group_by_partition_threshold: Optional[float] = None
        legacy_projection_column: Optional[str] = None
        legacy_projection_agg_func: Optional[_T_AGG_FUNC] = None
        legacy_selection_filters: Optional[List[Tuple[str, Any]]] = None
        legacy_group_by_column: Optional[str] = None

        # Extract dataset name
        dataset_match = re.search(r"FROM ([\w.]+)", query)
        if dataset_match:
            dataset_name = dataset_match.group(1)
            if "." in dataset_name:
                dataset_name, dataset_version = dataset_name.split(".")
            else:
                dataset_version = None

        # Extract and process model information
        query = query.strip()
        main_query = query.split("\n")[0]
        raw_model_info = query[len(main_query) :].strip()
        model_matches = re.findall(r"M(\d+)\s*:\s*([\w-]+),\s*(\w+)", raw_model_info)
        model_info: Dict[int, _T_MODEL] = {
            int(model_num): (model_path, model_type)
            for model_num, model_path, model_type in model_matches
        }

        logger.info(f"main query: {main_query}")
        logger.info(f"model info:\n{model_info}")

        # extract model from main query
        select_clause = main_query.split("SELECT")[1].split("FROM")[0]
        if "WHERE" in main_query:
            where_clause = main_query.split("WHERE")[1]
            if "GROUP BY" in where_clause:
                where_clause = where_clause.split("GROUP BY")[0]
        else:
            where_clause = ""
        if "GROUP BY" in main_query:
            group_by_clause = main_query.split("GROUP BY")[1]
            if "WHERE" in group_by_clause:
                group_by_clause = group_by_clause.split("WHERE")[0]
        else:
            group_by_clause = ""

        # extract model from select clause
        select_model_match = re.findall(r"(\w+)\(M(\d+)\)", select_clause)
        # check at most one model is selected
        assert (
            len(select_model_match) <= 1
        ), "Currently, at most one model can be selected."
        if select_model_match:
            model_num = int(select_model_match[0][1])
            projection_model = model_info[model_num]
            projection_model_agg_func = _fix_agg_func(select_model_match[0][0])
        else:
            # try non-aggregation projection
            select_model_match = re.findall(r"M(\d+)", select_clause)
            # check at most one model is selected
            assert (
                len(select_model_match) <= 1
            ), "Currently, at most one model can be selected."
            if select_model_match:
                model_num = int(select_model_match[0])
                projection_model = model_info[model_num]
                projection_model_agg_func = None

        legacy_proj_match = re.findall(
            rf"(\w+)\({dataset_name}\.([\w-]+|\*)\)", select_clause
        )
        assert (
            len(legacy_proj_match) <= 1
        ), "Currently, at most one projection can be selected. For group by column, it will be automatically selected as projection."
        if legacy_proj_match:
            legacy_projection_agg_func = _fix_agg_func(legacy_proj_match[0][0])
            legacy_projection_column = legacy_proj_match[0][1]
        else:
            # try non-aggregation projection
            legacy_proj_match = re.findall(rf"{dataset_name}\.(\w+|\*)", select_clause)
            # check at most one column is selected
            assert (
                len(legacy_proj_match) <= 1
            ), "Currently, at most one column can be selected."
            if legacy_proj_match:
                legacy_projection_column = legacy_proj_match[0]
            else:
                # try count(*)
                legacy_proj_match = re.findall(r"count\(\*\)", select_clause)
                if legacy_proj_match:
                    legacy_projection_agg_func = "count"
                    legacy_projection_column = "*"

        # extract model from where clause
        where_model_match = re.findall(
            r"(\bM\d+\b)\s*(==|!=|>|<|>=|<=)\s*('[^']*'|\"[^\"]*\"|\d+)", where_clause
        )
        # enumerate all models in match
        if where_model_match:
            selection_models, selection_model_conditions = [], []
            for raw_model_num, op, val in where_model_match:
                model_num = int(raw_model_num[1:])
                selection_models.append(model_info[model_num])
                selection_model_conditions.append(f"{op} {val}")

        legacy_sel_filters = re.findall(
            rf"{dataset_name}\.(\w+)\s*(==|!=|>|<|>=|<=)\s*('[^']*'|\"[^\"]*\"|\d+)",
            where_clause,
        )
        if legacy_sel_filters:
            # save in the format of [("age", "> 30"), ("edu", "== 'bachelor'")]
            legacy_selection_filters = [
                (col, f"{op} {val}") for col, op, val in legacy_sel_filters
            ]

        # extract model from group by clause
        group_by_model_match = re.findall(r"M(\d+)+", group_by_clause)
        # check at most one model is selected
        assert (
            len(group_by_model_match) <= 1
        ), "Currently, at most one model can be selected."
        if group_by_model_match:
            model_num = int(group_by_model_match[0])
            group_by_model = model_info[model_num]
            threshold_match = re.search(r"!\s*(\d+\.?\d*)", group_by_clause)
            if threshold_match:
                group_by_partition_threshold = float(threshold_match.group(1))

        legacy_group_by_match = re.findall(rf"{dataset_name}\.(\w+)", group_by_clause)
        # check at most one column is selected
        assert (
            len(legacy_group_by_match) <= 1
        ), "Currently, at most one column can be selected."
        assert len(legacy_group_by_match) > 0 or group_by_model
        if legacy_group_by_match:
            legacy_group_by_column = legacy_group_by_match[0]

        return cls(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            projection_model=projection_model,
            projection_model_agg_func=projection_model_agg_func,
            selection_models=selection_models,
            selection_model_conditions=selection_model_conditions,
            group_by_model=group_by_model,
            group_by_partition_threshold=group_by_partition_threshold,
            legacy_projection_column=legacy_projection_column,
            legacy_projection_agg_func=legacy_projection_agg_func,
            legacy_selection_filters=legacy_selection_filters,
            legacy_group_by_column=legacy_group_by_column,
        )
