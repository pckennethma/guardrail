import argparse
import pickle
import sys
from typing import Optional

import networkx as nx
import pandas as pd

from nsyn.dataset.loader import load_data_by_name_and_vers
from nsyn.dsl.prog import DSLProg
from nsyn.learner import PC
from nsyn.sampler import IdentitySampler
from nsyn.search import Search
from nsyn.util.dag import DAG
from nsyn.util.logger import get_logger

# add lib/fdx to sys.path
sys.path.append("lib/fdx")
from lib.fdx.profiler.core import *  # noqa: F403, E402

logger = get_logger(name="fdx.profiler.run")


def fdx_learn(data: pd.DataFrame) -> DAG:
    """
    Abstract method to learn from the data.

    Args:
        data (np.ndarray | pd.DataFrame): The input data for the learning algorithm.
        sampling_protocol (AbstractSampler): The sampling protocol to use on the data.

    Returns:
        DAG: A directed acyclic graph derived from the data.
    """
    pf = Profiler(workers=2, tol=1e-6, eps=0.05, embedtxt=False)  # noqa: F405
    pf.session.load_data(
        src=DF, df=data, check_param=True, na_values="empty"  # noqa: F405
    )
    pf.session.load_training_data(multiplier=None, difference=True)
    _ = pf.session.learn_structure(sparsity=0, infer_order=True)
    parent_sets = pf.session.get_dependencies(score="fit_error")

    columns = data.columns.tolist()
    dag = nx.MultiDiGraph()
    dag.add_nodes_from(range(len(columns)))
    for child in parent_sets:
        child_idx = columns.index(child)
        found = parent_sets[child]
        for parent in found:
            parent_idx = columns.index(parent)
            dag.add_edge(parent_idx, child_idx, directed=True)
    return DAG(graph=dag)


def main(
    dataset_name: str,
    dataset_version: str,
    output_file: Optional[str] = "output.pkl",
) -> None:
    df = load_data_by_name_and_vers(dataset_name, dataset_version)
    learner = PC()
    sampler = IdentitySampler()
    search = Search.create(
        learning_algorithm=learner,
        sampling_algorithm=sampler,
        input_data=df,
        epsilon=1,
        min_support=-1,
    )
    result = search._synthesis_from_dag(fdx_learn(df))
    assert isinstance(result, DSLProg)
    logger.info(f"Result: {result}")
    logger.info(f"Statistics: {result.statistics}")
    logger.info(f"Writing result to {output_file}")
    if output_file is not None:
        with open(output_file, "wb") as f:
            pickle.dump(result, f)


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
        "--dataset_version",
        "-v",
        type=str,
        default="train",
        help="The version of the dataset.",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        help="The path to the output file.",
    )
    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        output_file=args.output_file,
    )
