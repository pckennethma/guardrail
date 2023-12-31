import os
from typing import Literal

import pandas as pd

_DATASET_PATH = {
    "adult": "datasets/adult.csv",
    "lung_cancer": "datasets/lung_cancer.csv",
    "insurance": "datasets/insurance.csv",
    "bird_strikes": "datasets/bird_strikes.csv",
}


def get_df_path(name: str) -> str | None:
    return _DATASET_PATH.get(name, None)


def get_df_path_with_version(name: str, vers: str) -> str | None:
    df_path = _DATASET_PATH.get(name, None)
    if df_path is None:
        return None
    else:
        return df_path.replace(".csv", f".{vers}.csv")


def load_df(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def load_data_by_name(name: str) -> pd.DataFrame:
    df_path = _DATASET_PATH.get(name, None)
    if df_path is None:
        raise ValueError(f"Dataset {name} does not exist.")
    else:
        return load_df(df_path)


def load_data_by_name_and_vers(name: str, vers: str) -> pd.DataFrame:
    df_path = _DATASET_PATH.get(name, None)
    if df_path is None:
        raise ValueError(f"Dataset {name} does not exist.")
    else:
        df_path = df_path.replace(".csv", f".{vers}.csv")
        if not os.path.isfile(df_path):
            raise ValueError(f"Dataset {name} with version {vers} does not exist.")
        return load_df(df_path)


def load_ml_data_by_name(
    name: str, split: Literal["train", "test", "all"]
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    df_path = _DATASET_PATH.get(name, None)
    if df_path is None:
        raise ValueError(f"Dataset {name} does not exist.")
    else:
        if split == "train":
            # use df_path as "datasets/name.train.csv"
            df_path = df_path.replace(".csv", ".train.csv")
            return load_df(df_path)
        elif split == "test":
            # use df_path as "datasets/name.test.csv"
            df_path = df_path.replace(".csv", ".test.csv")
            return load_df(df_path)
        elif split == "all":
            train_df_path = df_path.replace(".csv", ".train.csv")
            test_df_path = df_path.replace(".csv", ".test.csv")
            return load_df(train_df_path), load_df(test_df_path)
        else:
            raise ValueError(
                f"Split {split} is not supported. Use either 'train', 'test', or 'all'."
            )
