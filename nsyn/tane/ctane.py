# type: ignore
import argparse
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from nsyn.util.logger import get_logger

logger = get_logger(__name__)


def parse_cfd_line(
    line: str,
) -> Optional[Tuple[List[Tuple[str, str]], Tuple[str, str]]]:
    # Remove leading/trailing whitespace
    line: str = line.strip()
    # Split on '=>'
    if "=>" not in line:
        # Invalid CFD line
        return None
    lhs_part: str
    rhs_part: str
    lhs_part, rhs_part = line.split("=>", 1)
    # Strip whitespace
    lhs_part = lhs_part.strip()
    rhs_part = rhs_part.strip()
    # Parse LHS conditions
    if not lhs_part.startswith("(") or not lhs_part.endswith(")"):
        # Invalid format
        return None
    lhs_content: str = lhs_part[1:-1]  # Remove the parentheses
    # Split lhs_content into attribute-value pairs
    lhs_conditions: List[Tuple[str, str]] = []
    conditions: List[str] = lhs_content.split(",")
    for cond in conditions:
        cond: str = cond.strip()
        if "=" not in cond:
            # Invalid condition
            continue
        attr: str
        value: str
        attr, value = cond.split("=", 1)
        lhs_conditions.append((attr.strip(), value.strip()))
    # Parse RHS condition
    if "=" not in rhs_part:
        # Invalid RHS
        return None
    rhs_attr: str
    rhs_value: str
    rhs_attr, rhs_value = rhs_part.split("=", 1)
    rhs_condition: Tuple[str, str] = (rhs_attr.strip(), rhs_value.strip())
    return lhs_conditions, rhs_condition


def create_condition_mask(df: pd.DataFrame, attr: str, value: str) -> pd.Series:
    # Handle ranges for numeric attributes
    if "-" in value and pd.api.types.is_numeric_dtype(df[attr]):
        try:
            low_str: str
            high_str: str
            low_str, high_str = value.split("-")
            low: float = float(low_str)
            high: float = float(high_str)
            mask: pd.Series = (df[attr] >= low) & (df[attr] <= high)
            return mask
        except ValueError:
            pass
    # Otherwise, equality check
    mask: pd.Series = df[attr] == value
    return mask


def apply_cfds(df: pd.DataFrame, cfd_lines: List[str]) -> pd.DataFrame:
    # Initialize the '_CTANE_ERROR' column
    df["_CTANE_ERROR"] = [[] for _ in range(len(df))]  # type: ignore

    idx: int
    cfd_line: str
    for idx, cfd_line in enumerate(cfd_lines):
        cfd_result: Optional[
            Tuple[List[Tuple[str, str]], Tuple[str, str]]
        ] = parse_cfd_line(cfd_line)
        if cfd_result is None:
            continue
        lhs_conditions: List[Tuple[str, str]]
        rhs_condition: Tuple[str, str]
        lhs_conditions, rhs_condition = cfd_result
        # Create the mask for LHS conditions
        mask: pd.Series = pd.Series([True] * len(df))
        attr: str
        value: str
        for attr, value in lhs_conditions:
            if attr not in df.columns:
                # Skip if attribute not in DataFrame
                mask = pd.Series([False] * len(df))
                break
            mask &= create_condition_mask(df, attr, value)
        # Now, find where RHS condition fails
        rhs_attr: str
        rhs_value: str
        rhs_attr, rhs_value = rhs_condition
        if rhs_attr not in df.columns:
            continue
        rhs_mask: pd.Series = create_condition_mask(df, rhs_attr, rhs_value)
        violation_mask: pd.Series = mask & (~rhs_mask)
        # For these rows, append the CFD ID to '_CTANE_ERROR' column
        cfd_id: str = f"CFD_{idx+1}"
        df.loc[violation_mask, "_CTANE_ERROR"] = df.loc[
            violation_mask, "_CTANE_ERROR"
        ].apply(lambda x: x + [cfd_id])
    # Convert '_CTANE_ERROR' lists to strings for readability
    df["_CTANE_ERROR"] = df["_CTANE_ERROR"].apply(lambda x: ", ".join(x) if x else "")
    return df


def main():
    parser = argparse.ArgumentParser(description="Apply CFDs to a DataFrame.")
    parser.add_argument("--test", action="store_true", help="Run with test data.")
    parser.add_argument("--csv", type=str, help="Path to input CSV file.")
    parser.add_argument("--cfd", type=str, help="Path to input CFD text file.")
    parser.add_argument("--output", type=str, help="Path to output CSV file.")

    args = parser.parse_args()

    # Default to --test mode if no arguments are provided
    if not any(vars(args).values()):
        args.test = True

    if args.test:
        # Example usage with test data
        data: Dict[str, List[Union[str, int, float, None]]] = {
            "Occupation": [
                "Machine-op-inspct",
                "Other-service",
                "Prof-specialty",
                "Other-service",
                "?",
            ],
            "Workclass": ["Private", "Private", "Self-emp", "State-gov", "?"],
            "Marital-status": [
                "Never-married",
                "Married",
                "Never-married",
                "Married",
                "Divorced",
            ],
            "Income": [
                "LessThan50K",
                "LessThan50K",
                "GreaterThan50K",
                "LessThan50K",
                "LessThan50K",
            ],
            "Relationship": [
                "Husband",
                "Own-child",
                "Not-in-family",
                "Other-relative",
                "Unmarried",
            ],
            "Age": [18, 25, 40, 18, 30],
            "Sex": ["Male", "Female", "Male", "Female", "Female"],
        }
        df: pd.DataFrame = pd.DataFrame(data)

        cfds: List[str] = [
            "(Occupation=Machine-op-inspct) => Workclass=Private",
            "(Marital-status=Never-married) => Income=LessThan50K",
            "(Relationship=Husband) => Sex=Male",
            "(Relationship=Own-child) => Income=LessThan50K",
            "(Occupation=Other-service) => Income=LessThan50K",
            "(Relationship=Other-relative) => Income=LessThan50K",
            "(Age=18-21) => Income=LessThan50K",
            "(Occupation=?) => Workclass=?",
        ]
    elif args.csv and args.cfd:
        # Load data from CSV and CFD files
        df = pd.read_csv(args.csv)
        with open(args.cfd, "r") as file:
            cfds = file.readlines()
    else:
        parser.error("Either --test or both --csv and --cfd must be provided.")

    df = apply_cfds(df, cfds)
    logger.info(df.head())

    if args.output:
        df.to_csv(args.output, index=False)
    else:
        logger.warning("No output path provided. Results will not be saved to a file.")


if __name__ == "__main__":
    main()
