import json
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as matcoll
import math
import os
from typing import List, Optional
import pickle
import pandas as pd
import tqdm

import sys

sys.path.append("../")
sys.path.append(".")

from nsyn.dataset.loader import _DATASET_PATH


def get_coverage_loss(dataset_name):
    model_dir = "models/"
    for filename in os.listdir(model_dir):
        if filename.startswith(dataset_name):
            model_folder = filename
            break

    # get length of the dataset
    df_path = _DATASET_PATH.get(dataset_name, None)
    df_path = df_path.replace(".csv", ".train.csv")
    df = pd.read_csv(df_path)

    coverage_ls = []
    loss_ls = []
    for epsilon in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:  # 0.01
        with open(
            os.path.join(
                model_dir,
                model_folder,
                "epsilon",
                f"nsyn_prog_epsilon_{epsilon}_idx_0.pkl",
            ),
            "rb",
        ) as f:
            data_log = pickle.load(f)
            coverage_ls.append(data_log.coverage / len(df))
            loss_ls.append(
                sum(stmt.compute_loss(df) for stmt in data_log.stmts)
                if data_log.stmts
                else 0
            )
        pickle.dump(
            data_log,
            open(
                os.path.join(
                    model_dir,
                    model_folder,
                    "epsilon",
                    f"nsyn_prog_epsilon_{epsilon}_idx_0_with_loss.pkl",
                ),
                "wb",
            ),
        )
    # min_max normalization
    coverage_ls = [
        (i - min(coverage_ls)) / (max(coverage_ls) - min(coverage_ls))
        for i in coverage_ls
    ]
    loss_ls = [(i - min(loss_ls)) / (max(loss_ls) - min(loss_ls)) for i in loss_ls]
    return coverage_ls, loss_ls


dataset_name_ls = [
    "adult",
    "lung_cancer",
    "cylinder-bands",
    "diabetes" "contraceptive",
    "blood-transfusion-service-center",
    "steel-plates-fault",
    "jungle_chess_2pcs_raw_endgame_complete",
    "telco_customer_churn",
    "bank-marketing",
    "PhishingWebsites",
    "hotel_bookings",
]
dt_name_ls_full = [
    "Adult",
    "Lung Cancer",
    "Cylinder Bands",
    "Diabetes" "Contraceptive Method",
    "Blood Trans. Serv. Ctr.",
    "Steel Plates Faults",
    "Jungle Chess",
    "Telco Customer Churn",
    "Bank Marketing",
    "Phishing Websites",
    "Hotel Bookings",
]
dt_name_dict = dict(zip(dataset_name_ls, dt_name_ls_full))
for dataset_name in tqdm.tqdm(dataset_name_ls):
    coverage_ls, loss_ls = get_coverage_loss(dataset_name)
