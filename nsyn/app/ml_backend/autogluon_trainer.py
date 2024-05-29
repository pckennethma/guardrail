import os
import sys
from typing import List, Optional
from uuid import uuid4

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

sys.path.append(".")
from nsyn.dataset.loader import load_ml_data_by_name
from nsyn.run import run_search
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.app.ml_backend.autogluon_trainer")


def main(
    dataset_name: str,
    label_column: str,
    feature_columns: Optional[List[str]],
    disable_synthesizer: bool,
    use_balanced_acc_metric: bool,
) -> None:
    """
    The method to train a model for a given dataset.

    Args:
        dataset_name (str): The name of the dataset.
        label_column (str): The name of the label column.
        feature_columns (Optional[List[str]]): The name of the feature columns. Defaults to None, which means all columns except the label column.
    """

    train_df, test_df = load_ml_data_by_name(dataset_name, "all")
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    if feature_columns is not None:
        train_df = train_df[feature_columns + [label_column]]
        test_df = test_df[feature_columns + [label_column]]
    else:
        logger.info(
            f"Using all columns except {label_column} as features: {train_df.columns}"
        )

    if train_df[label_column].isnull().any():
        logger.warning(
            f"Train df has {train_df[label_column].isnull().sum()} missing values out of {train_df.shape[0]} rows in the label column. Dropping them..."
        )
        train_df = train_df.dropna(subset=[label_column])

    if test_df[label_column].isnull().any():
        logger.warning(
            f"Test df has {test_df[label_column].isnull().sum()} missing values out of {test_df.shape[0]} rows in the label column. Dropping them..."
        )
        test_df = test_df.dropna(subset=[label_column])

    train_dataset = TabularDataset(train_df)
    test_dataset = TabularDataset(test_df)

    model_folder = os.path.join("models", f"{dataset_name}-{uuid4()}")
    os.makedirs(model_folder, exist_ok=True)
    logger.info(f"Saving model to {model_folder}...")

    logger.info("Training model...")
    if use_balanced_acc_metric:
        logger.info("Using balanced accuracy metric...")
        predictor = TabularPredictor(
            label=label_column, path=model_folder, sample_weight="auto_weight"
        ).fit(train_dataset, presets="optimize_for_deployment")
    else:
        logger.info("Using standard accuracy metric...")
        predictor = TabularPredictor(
            label=label_column,
            path=model_folder,
        ).fit(train_dataset, presets="optimize_for_deployment")
    logger.info("Training done.")

    logger.info("Evaluating model...")
    metrics = predictor.evaluate(test_dataset, silent=True)
    logger.info(f"Performance: {metrics}")

    if not disable_synthesizer:
        logger.info("Running synthesizer...")
        run_search(
            data_name_or_df=train_df,
            output_file=os.path.join(model_folder, "nsyn_prog.pkl"),
            dag_save_path=os.path.join(model_folder, "nsyn_dag.pkl"),
        )
        logger.info("Synthesizer done.")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        "-j",
        type=str,
        help="The path to the training config. If specified, the following arguments will be ignored.",
    )
    parser.add_argument(
        "--dataset_name",
        "-d",
        type=str,
        help="The name of the dataset.",
    )
    parser.add_argument(
        "--label_column",
        "-l",
        type=str,
        help="The name of the label column.",
    )
    parser.add_argument(
        "--feature_columns",
        "-f",
        type=str,
        nargs="+",
        help="The name of the feature columns.",
    )
    parser.add_argument(
        "--disable_synthesizer",
        "-s",
        action="store_true",
        help="Whether to run the synthesizer after training.",
    )
    parser.add_argument(
        "--use_balanced_acc_metric",
        "-b",
        action="store_true",
        default=False,
        help="Whether to use balanced accuracy metric (default: standard accuracy metric).",
    )

    args = parser.parse_args()

    if args.json_path is not None:
        with open(args.json_path, "r") as f:
            train_config = json.load(f)
        dataset_name = train_config["dataset_name"]
        label_column = train_config["label_column"]
        feature_columns = train_config["feature_columns"]
        disable_synthesizer = train_config["disable_synthesizer"]
    else:
        dataset_name = args.dataset_name
        label_column = args.label_column
        feature_columns = args.feature_columns
        disable_synthesizer = args.disable_synthesizer

    if dataset_name is not None:
        main(
            dataset_name=dataset_name,
            label_column=label_column,
            feature_columns=feature_columns,
            disable_synthesizer=disable_synthesizer,
            use_balanced_acc_metric=args.use_balanced_acc_metric,
        )
    else:
        # print help message
        parser.print_help()
