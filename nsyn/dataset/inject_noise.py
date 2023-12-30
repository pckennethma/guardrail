import argparse
import os
import pickle
import random
from typing import List, Literal

import numpy as np
import pandas as pd

from nsyn.app.ml_backend.auto import get_inference_model
from nsyn.dataset.loader import get_df_path_with_version, load_ml_data_by_name
from nsyn.dsl.prog import DSLProg
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.dataset.generate_train_test_split")


def inject_noise(
    dataset_name: str,
    output_version: str,
    model_path: str,
    model_type: Literal["autogluon", "llm"],
    noise_level: float,
) -> None:
    """
    Injects noise into a test dataset based on predictions made by a given model.

    This function first loads a test dataset and a pre-trained model. It then identifies the rows
    where the model's predictions are correct. For these rows, it introduces noise into the target
    columns defined in a provided nsyn program. The level of noise injection is determined by the
    noise_level parameter. The modified dataset is then saved as a new version.

    Parameters:
    - dataset_name (str): The name of the dataset to be used.
    - output_version (str): The version identifier for the output dataset.
    - model_path (str): Path to the pre-trained model.
    - model_type (Literal['autogluon', 'llm']): The type of the model (either 'autogluon' or 'llm').
    - noise_level (float): The fraction of data in the target columns to be modified with noise.

    The function updates target columns in the following manner:
    - If the column data type is 'object', x% of the values are replaced with '<UNK>' (unknown) or other values in the column.
    - If the column data type is numeric, x% of the values are replaced with 0.

    Note:
    - The function assumes that the input DataFrame and the nsyn program are in the correct formats.
    - The noise is injected only into the rows where the model's predictions are correct.
    - The function saves the modified dataset to a CSV file with the specified output version.
    """
    logger.info(f"Injecting noise into {dataset_name}...")
    logger.info(f"Output version: {output_version}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Noise level: {noise_level}")

    model = get_inference_model(model_path, model_type)
    logger.info(f"Loaded model from {model_path}")

    test_df = load_ml_data_by_name(dataset_name, "test")
    assert isinstance(test_df, pd.DataFrame)
    logger.info(f"Labeled test df shape: {test_df.shape}")

    label_column = model.inference_model_config.label_column
    logger.info(f"Label column: {label_column}")
    correct_df = test_df[test_df[label_column] == model.predict(test_df)]
    logger.info(f"Correctly predicted df shape: {correct_df.shape}")

    nsyn_prog_path = os.path.join(model_path, "nsyn_prog.pkl")
    with open(nsyn_prog_path, "rb") as f:
        nsyn_prog = pickle.load(f)
        assert isinstance(nsyn_prog, DSLProg)
    logger.info(f"Loaded nsyn program from {nsyn_prog_path}")

    target_columns = [
        stmt.dependent for stmt in nsyn_prog.stmts if stmt.dependent != label_column
    ]
    logger.info(f"Target columns: {target_columns}")

    correct_df["_nsyn_noisy_injected"] = False
    for column in target_columns:
        # Determine the number of values to replace
        num_values_to_replace = int(noise_level * len(correct_df))
        # Replace values in randomly selected rows
        subsampled_indices = np.random.choice(
            correct_df.index, size=num_values_to_replace, replace=False
        )
        correct_df.loc[subsampled_indices, "_nsyn_noisy_injected"] = True
        if correct_df[column].dtype == object:
            # Pre-compute unique values including '<UNK>'
            possible_values = correct_df[column].unique().tolist() + ["<UNK>"]
            weights: List[float] = np.ones(len(possible_values)).tolist()
            weights[-1] = 10
            # Randomly choose new values for a subset of rows
            random_values = random.choices(
                possible_values, weights=weights, k=num_values_to_replace
            )
            correct_df.loc[subsampled_indices, column] = random_values
        elif correct_df[column].dtype == bool:
            # Flip the values of a random subset of rows for boolean columns
            correct_df.loc[subsampled_indices, column] = ~correct_df.loc[
                subsampled_indices, column
            ]
        else:
            # Directly assign NaN to a random subset of rows for numeric columns
            correct_df.loc[subsampled_indices, column] = np.nan

    correct_df.to_csv(
        get_df_path_with_version(dataset_name, output_version), index=False
    )
    logger.info(
        f"Saved modified dataset to {get_df_path_with_version(dataset_name, output_version)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        "-d",
        type=str,
        required=True,
        help="The name of the dataset.",
    )
    parser.add_argument(
        "--output_version",
        "-o",
        type=str,
        required=True,
        help="The version of the output dataset.",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        required=True,
        help="The path to the model.",
    )
    parser.add_argument(
        "--model_type",
        "-t",
        type=str,
        default="autogluon",
        choices=["autogluon", "llm"],
        help="The type of the model.",
    )
    parser.add_argument(
        "--noise_level",
        "-n",
        type=float,
        default=0.1,
        help="The noise level.",
    )
    args = parser.parse_args()
    inject_noise(
        dataset_name=args.dataset_name,
        output_version=args.output_version,
        model_path=args.model_path,
        model_type=args.model_type,
        noise_level=args.noise_level,
    )
