import argparse
import pickle

from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.dsl.display_prog")


def main(
    prog_file: str,
) -> None:
    with open(prog_file, "rb") as f:
        prog = pickle.load(f)
    logger.info(f"Loaded program from {prog_file}")
    logger.info(f"Program:\n{prog}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prog_file",
        "-p",
        type=str,
        help="The path to the program file.",
    )
    args = parser.parse_args()
    main(
        prog_file=args.prog_file,
    )
