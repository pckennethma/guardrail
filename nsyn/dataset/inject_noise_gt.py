import argparse
import os
import pickle
import sys
from typing import Hashable, Literal

sys.path.append(".")

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
    logger.info(f"Labeled test df shape: {test_df.shape}")

    label_column = model.inference_model_config.label_column
    logger.info(f"Label column: {label_column}")
    correct_df = test_df[test_df[label_column] == model.predict(test_df)].copy()
    logger.info(f"Correctly predicted df shape: {correct_df.shape}")

    nsyn_prog_path = os.path.join(model_path, "nsyn_prog.pkl")
    with open(nsyn_prog_path, "rb") as f:
        nsyn_prog = pickle.load(f)
        assert isinstance(nsyn_prog, DSLProg)
    logger.info(f"Loaded nsyn program from {nsyn_prog_path}")

    controllable_tuples: dict[Hashable, list[str]] = {}

    for stmt in nsyn_prog.stmts:
        controllable_indices = [
            index
            for index, row in correct_df.iterrows()
            if stmt.evaluate(row.to_dict())
        ]
        controllable_tuples.update(
            {
                index: [stmt.dependent]
                if index not in controllable_tuples
                else controllable_tuples[index] + [stmt.dependent]
                for index in controllable_indices
            }
        )

    logger.info(f"# Controllable tuples: {len(controllable_tuples)}")

    controllable_df = correct_df.loc[list(controllable_tuples.keys())].copy()

    controllable_df.to_csv(
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
        default="gt",
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
