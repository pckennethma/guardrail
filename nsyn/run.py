import argparse
import json
import pickle
import time
from typing import Optional, cast

import pandas as pd

from nsyn.app.error_detector import ErrorDetector
from nsyn.dataset.loader import load_data_by_name, load_ml_data_by_name
from nsyn.dsl.prog import DSLProg
from nsyn.learner import BLIP, GES, PC, BaseLearner
from nsyn.sampler import AuxiliarySampler, IdentitySampler
from nsyn.search import Search
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.run")


def run_search(
    data_name_or_df: str | pd.DataFrame,
    output_file: str,
    learner_name: str = "auto",
    epsilon: float = 0.01,
    enable_auxiliary_sampler: bool = True,
    num_of_dags: None | int = None,
    dag_save_path: Optional[str] = None,
) -> None:
    if isinstance(data_name_or_df, pd.DataFrame):
        data = data_name_or_df
    else:
        data = cast(pd.DataFrame, load_ml_data_by_name(data_name_or_df, "train"))
    sampler = AuxiliarySampler() if enable_auxiliary_sampler else IdentitySampler()
    learner: BaseLearner

    logger.info(f"Input data shape: {data.shape}")

    if learner_name == "auto":
        learner = BLIP()
        learner.set_dag_save_path(dag_save_path)
    elif learner_name == "pc":
        learner = PC()
    elif learner_name == "ges":
        learner = GES()
    elif learner_name == "blip":
        learner = BLIP()
    else:
        raise ValueError(f"Unknown learner name {learner_name}")
    search = Search.create(
        learning_algorithm=learner,
        sampling_algorithm=sampler,
        input_data=data,
        epsilon=epsilon,
    )
    start = time.time()
    result = search.run()
    if num_of_dags is not None:
        used_time = time.time() - start
        log_file = output_file.replace(".pkl", ".json")
        with open(log_file, "w") as f:
            json.dump({"num_of_dags": num_of_dags, "used_time": used_time}, f)

    assert isinstance(result, DSLProg)
    logger.info(f"Result: {result}")
    logger.info(f"Statistics: {result.statistics}")
    logger.info(f"Writing result to {output_file}")
    with open(output_file, "wb") as f:
        pickle.dump(result, f)


def run_error_detector(
    data_name: str,
    program_file: str,
) -> None:
    data = load_data_by_name(data_name)
    with open(program_file, "rb") as f:
        program = pickle.load(f)
    error_detector = ErrorDetector(data=data, program=program)
    errors = error_detector.run()
    logger.info(f"Found {len(errors)} errors:")
    for error in errors:
        logger.info(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        help="Name of the dataset to use for program synthesis",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to the file where the output program should be written",
    )
    parser.add_argument(
        "--program",
        "-p",
        type=str,
        help="Path to the file containing the program to be evaluated",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["s", "e"],
        help="The mode to run the program in (s for search, e for error detection)",
    )
    parser.add_argument(
        "--learner",
        "-l",
        type=str,
        choices=["pc", "ges", "auto", "blip"],
        default="auto",
        help="The learning algorithm to use for program synthesis",
    )
    args = parser.parse_args()
    if args.mode == "e":
        if args.program is None:
            raise ValueError("Must provide a program file for error detection")
        run_error_detector(
            data_name=args.data,
            program_file=args.program,
        )
    else:
        if args.output is None:
            raise ValueError("Must provide an output file for search")
        run_search(
            data_name_or_df=args.data,
            output_file=args.output,
            learner_name=args.learner,
        )
