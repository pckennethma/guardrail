# This is an ANSI escape code for green color.
GREEN = "\033[92m"
# This is an ANSI escape code for bold text.
BOLD = "\033[1m"
# This is an ANSI escape code to end formatting.
END = "\033[0m"
# This is an ANSI escape code for yellow color.
YELLOW = "\033[93m"


def get_keyword_text(text: str) -> str:
    """
    Returns the keyword text with bold and purple color in the terminal.

    :param text: The text to be formatted.

    :return: The formatted text.
    """
    return f"{GREEN}{BOLD}{text}{END}"


def get_output_df_text(text: str) -> str:
    """
    Returns the output dataframe text with yellow color in the terminal.

    :param text: The text to be formatted.

    :return: The formatted text.
    """
    return f"{YELLOW}{text}{END}"
