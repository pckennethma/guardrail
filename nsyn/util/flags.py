import os
from uuid import uuid4

# This flag `DISABLE_SANITIZER_FLAG` is used to control whether or not to
# sanitize ML predictions during the execution of the Query 2.0. The flag is
# utilized in the `predict` function in the `InferenceModel` class. If
# DISABLE_SANITIZER_FLAG is set to True, the function will return the raw
# predictions without sanitization. Otherwise, the function will perform
# sanitization on the predictions. Then, the sanitizer will alert the
# mismatches between the input feature and the expected feature.
DISABLE_SANITIZER_FLAG = (
    os.getenv("NSYN_Q2_DISABLE_SANITIZER", "false").lower() == "true"
)

# This flag `SAN_RELEVANCE_ANALYSIS_FLAG` is used to control whether or not to
# perform relevance analysis during the prediction stage of the ML model.
# The flag is utilized in the `predict` function in the `InferenceModel` class.
# If SAN_RELEVANCE_ANALYSIS_FLAG is set to True, the function will perform
# additional operations for checking the relevance between sanitizer alerts and
# prediction errors. For instance, a correlation matrix between sanitizer alerts
# and prediction errors will be logged in the console. Note that, the relevance
# analysis requires that the ground-truth label is given in the input dataframe
# of the `predict` function.
SAN_RELEVANCE_ANALYSIS_FLAG = (
    os.getenv("NSYN_Q2_RELEVANCE_ANALYSIS", "true").lower() == "true"
)

# This flag `SAN_COMPARATIVE_ANALYSIS_FLAG` is used to control whether or not to
# perform comparative analysis during the prediction stage of the ML model. The
# flag is only applicable when `ERROR_HANDLING_FLAG` is set to `rectify`. If
# SAN_COMPARATIVE_ANALYSIS_FLAG is set to True, a comparative analysis will be
# performed between the predictions from the original data and the predictions
# from the rectified data. Note that, the comparative analysis requires that the
# ground-truth label is given in the input dataframe of the `predict` function.
SAN_COMPARATIVE_ANALYSIS_FLAG = (
    os.getenv("NSYN_Q2_COMPARATIVE_ANALYSIS", "false").lower() == "true"
)

# This flag `SAN_ANALYSIS_OUTPUT_JSONL_PATH` is used to control the output path
# of the relevance analysis results. The output is a JSONL file containing the
# relevance analysis results. If the flag is not specified, a random path will
# be generated.
SAN_ANALYSIS_OUTPUT_JSONL_PATH = os.getenv(
    "NSYN_Q2_RELEVANCE_ANALYSIS_CSV_PATH", f"ra-{uuid4()}.jsonl"
)

# This flag `ERROR_HANDLING_FLAG` is used to control the error handling
# strategy during the execution of the Query 2.0. There are three options:
# - ignore: ignore all errors (but log them) and continue the execution
# - raise: raise an error and stop the execution
# - coerce: set the value to NaN and continue the execution
# - rectify: rectify the data errors and continue the execution
ERROR_HANDLING_FLAG = os.getenv("NSYN_Q2_ERROR_HANDLING", "ignore").lower()
