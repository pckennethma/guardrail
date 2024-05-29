import argparse
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(".")

from nsyn.dataset.loader import get_df_path, load_data_by_name, load_ml_data_by_name
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.dataset.generate_train_test_split")


def main(
    name: str,
) -> None:
    """
    The method to generate train-test split for a given dataset.
    """
    df_path = get_df_path(name)
    df = load_data_by_name(name)
    assert df_path is not None

    logger.info(f"Generating train-test split for {name}...")
    logger.info(f"Dataset path: {df_path}")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"df.head():\n{df.head()}")

    train_df_path = df_path.replace(".csv", ".train.csv")
    test_df_path = df_path.replace(".csv", ".test.csv")

    logger.info(f"Split to {train_df_path} and {test_df_path}...")
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    train_df.to_csv(train_df_path, index=False)
    test_df.to_csv(test_df_path, index=False)

    train_df, test_df = load_ml_data_by_name(name, "all")

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    logger.info(f"Train df shape: {train_df.shape}")
    logger.info(f"Test df shape: {test_df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        required=True,
        help="The name of the dataset.",
    )
    args = parser.parse_args()
    main(
        name=args.name,
    )
