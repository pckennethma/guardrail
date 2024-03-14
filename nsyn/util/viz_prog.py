import pickle
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import palettable
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

from nsyn.dataset.loader import load_data_by_name_and_vers
from nsyn.dsl.prog import DSLProg
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.util.viz")

plt.rcParams["pdf.fonttype"] = 42


def plot_dsl_program(
    prog: DSLProg,
    dataset_name: str,
    dataset_version: str,
    output_file: Optional[str],
) -> None:
    df = load_data_by_name_and_vers(dataset_name, dataset_version)
    columns = df.columns.tolist()
    # truncate column names to 5 characters
    short_columns = [f"#{i+1}" for i in range(len(columns))]

    heatmap = np.zeros((len(columns), len(columns)))

    for stmt in prog.stmts:
        for parent in stmt.determinants:
            heatmap_value = max(0.001, stmt.compute_loss(df) / len(df) * 100)
            heatmap[
                columns.index(parent), columns.index(stmt.dependent)
            ] = heatmap_value
            logger.info(
                f"Parent: {parent}, Child: {stmt.dependent}, Loss: {heatmap_value}"
            )

    # Create a custom colormap with a transparent color for zero values
    orrd_colors = palettable.colorbrewer.sequential.OrRd_9.mpl_colors[1:]
    # create a colormap object from the list of colors
    cmap = ListedColormap(orrd_colors)
    # set the bad value color to transparent
    cmap.set_bad("0", 0)

    # Replace 0 values with NaN for transparency
    heatmap[heatmap == 0] = np.nan
    heatmap[heatmap == 0.001] = 0

    plt.figure()
    heatmap_df = pd.DataFrame(heatmap, columns=short_columns, index=short_columns)
    ax = sns.heatmap(
        data=heatmap_df,
        cmap=cmap,
        cbar=True,  # Enable color bar if needed
        square=True,
        vmin=0,
        vmax=1.6,
    )
    ax.set(
        xlabel="Columns",
        ylabel="Columns",
    )
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        help="Name of the dataset to use for program synthesis",
    )
    parser.add_argument(
        "--version",
        "-v",
        type=str,
        default="train",
        help="Version of the dataset to use for program synthesis",
    )
    parser.add_argument(
        "--program",
        "-p",
        type=str,
        required=True,
        help="Path to the program to visualize",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to the output file",
    )
    args = parser.parse_args()

    with open(args.program, "rb") as f:
        prog = pickle.load(f)
        assert isinstance(prog, DSLProg)
        logger.info(f"Loaded program: {prog}")
    plot_dsl_program(
        prog=prog,
        dataset_name=args.data,
        dataset_version=args.version,
        output_file=args.output,
    )
