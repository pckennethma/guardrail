import pandas as pd

_DATASET_PATH = {
    "adult": "datasets/adult.csv",
}


def load_df(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)


def load_data_by_name(name: str) -> pd.DataFrame:
    if name == "adult":
        return load_df(_DATASET_PATH["adult"])
    else:
        raise ValueError(f"Dataset {name} is not supported.")
