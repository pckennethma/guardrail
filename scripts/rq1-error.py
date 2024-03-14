import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse, glob
from pydantic.type_adapter import TypeAdapter
from typing import Optional
from nsyn.app.ml_backend.analysis import RelevanceAnalysisDumpItem
from nsyn.util.logger import get_logger

logger = get_logger(name="scripts.error_vs_misprediction")

dataset_order = [
    "adult",
    "lung_cancer",
    "insurance",
    "bird_strikes",
]
dataset_alias = {
    "adult": "ADULT",
    "lung_cancer": "LC",
    "insurance": "INS",
    "bird_strikes": "BS",
}


def process_file(stat_path: str) -> tuple[dict[str, float], str]:
    logger.info(f"Processing file: {stat_path}")
    adapter = TypeAdapter(RelevanceAnalysisDumpItem)
    with open(stat_path) as reader:
        data_samples = list(map(adapter.validate_json, reader.readlines()))
        logger.info(f"Loaded {len(data_samples)} data samples.")
        logger.info(f"Sample: {data_samples[0]}")

    statistics = {
        'total_pred_error_num': np.mean([item.total_pred_error_num for item in data_samples]),
        'detected_data_error_num': np.mean([item.detected_data_error_num for item in data_samples]),
        'actual_data_error_num': np.mean([item.actual_data_error_num for item in data_samples]),
        'detected_pred_error_num': np.mean([item.detected_pred_error_num for item in data_samples]),
        'total_input_num': np.mean([item.total_input_num for item in data_samples]),
        'falsely_detected_data_error_num': np.mean([item.falsely_detected_data_error_num for item in data_samples]),
    }

    return statistics, "na" if data_samples[0].ctx is None else data_samples[0].ctx.dataset.split(".")[0]

def main(stat_paths: list[str], output_path: str):
    logger.info(f"Processing {len(stat_paths)} statistics files: {stat_paths}")
    all_statistics = [process_file(path) for path in stat_paths]

    sorted_statistics = sorted(all_statistics, key=lambda x: dataset_order.index(x[1]))
    
    nrows = 2
    ncols = 2

    fig, stacked_axs = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(7.5, 7.5), 
                           gridspec_kw={'hspace': 0.3, 'wspace': 0.3},
                            sharex='col', sharey='row')
    sns.set(font_scale=1.2)
    axs = [
        stacked_axs[0, 0],
        stacked_axs[0, 1],
        stacked_axs[1, 0],
        stacked_axs[1, 1],
    ]
    for index, statistics_and_dataset in enumerate(all_statistics):
        statistics, dataset = statistics_and_dataset
        dataset_alias_name = dataset_alias[dataset]
        # Compute confusion matrices
        tp_data = statistics['detected_data_error_num']
        fp_data = statistics['falsely_detected_data_error_num']
        fn_data = statistics['actual_data_error_num'] - tp_data
        tn_data = statistics['total_input_num'] - statistics['actual_data_error_num'] - fp_data
        
        data_error_confusion_matrix_data = np.array([[tp_data, fp_data], [fn_data, tn_data]])
        data_error_normalized_confusion_matrix_data = data_error_confusion_matrix_data.astype('float') / data_error_confusion_matrix_data.sum(axis=1)[:, np.newaxis]

        logger.info(f"Confusion matrix for {dataset_alias_name}:\n{data_error_confusion_matrix_data}")
        logger.info(f"Normalized confusion matrix for {dataset_alias_name}:\n{data_error_normalized_confusion_matrix_data}")

        tp_pred = statistics['detected_pred_error_num']
        fp_pred = statistics['detected_data_error_num'] - tp_pred
        fn_pred = statistics['total_pred_error_num'] - tp_pred
        tn_pred = statistics['total_input_num'] - statistics['total_pred_error_num'] - fp_data

        pred_error_confusion_matrix_data = np.array([[tp_pred, fp_pred], [fn_pred, tn_pred]])
        pred_error_normalized_confusion_matrix_data = pred_error_confusion_matrix_data.astype('float') / pred_error_confusion_matrix_data.sum(axis=1)[:, np.newaxis]

        # ax_data_error = axs[0, index]  # Selects the subplot for 'Data Error' in the first row
        ax_data_error = axs[index]  # Selects the subplot for 'Data Error' in the first row
        # ax_ml_pred = axs[1, index]  # Selects the subplot for 'ML Prediction' in the second row

        sns.heatmap(data_error_normalized_confusion_matrix_data, annot=True, fmt=".2f", cmap="Blues", ax=ax_data_error, square=True, cbar=False)
        # sns.heatmap(pred_error_normalized_confusion_matrix_data, annot=True, fmt=".2f", cmap="Greens", ax=ax_ml_pred, square=True, cbar=False)

        ax_data_error.set_xlabel(f"Dataset: {dataset_alias_name}")
        ax_data_error.set_xticklabels(["Detected", "Not Detected"])

    # for ax in axs[0:1, 0]:
    #     ax.set_ylabel("Data Error")
    #     ax.set_yticklabels(["Has Error", "No Error"])
    axs[0].set_ylabel("Data Error")
    axs[0].set_yticklabels(["Has Error", "No Error"])
    axs[2].set_ylabel("Data Error")
    axs[2].set_yticklabels(["Has Error", "No Error"])
    # for ax in axs[1, 0:1]:
    #     ax.set_ylabel("ML Prediction")
    #     ax.set_yticklabels(["Has Error", "No Error"])

    # Adjust labels and ticks as before...
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the statistics of relevance analysis.")
    parser.add_argument("--output_path", "-o", type=str, help="The path to the output image.", default="")
    args = parser.parse_args()
    if not args.output_path:
        import pathlib
        args.output_path = pathlib.Path(__file__).parent.parent / "statistics" / "combined_analysis.pdf"
    
    main(glob.glob("statistics/ra-*.jsonl"), args.output_path)