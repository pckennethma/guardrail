import argparse
import pickle

from nsyn.app.error_detector import ErrorDetector
from nsyn.dataset.loader import load_data_by_name
from nsyn.dsl.prog import DSLProg
from nsyn.learner import PC
from nsyn.sampler import AuxiliarySampler
from nsyn.search import Search
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.run")


def run_search(
    data_name: str,
    output_file: str,
) -> None:
    data = load_data_by_name(data_name)
    sampler = AuxiliarySampler()
    learner = PC()
    search = Search.create(
        learning_algorithm=learner,
        sampling_algorithm=sampler,
        input_data=data,
    )
    result = search.run()
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
            data_name=args.data,
            output_file=args.output,
        )
