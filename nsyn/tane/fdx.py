from typing import Optional
import numpy as np
from sklearn.covariance import GraphicalLasso
import pandas as pd
from nsyn.dsl.prog import DSLProg
from nsyn.dsl.stmt import DSLStmt
from nsyn.learner import BaseLearner
from nsyn.sampler import AuxiliarySampler
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.tane.fdx")

def estimate_B_via_graphical_lasso(data: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Estimate the autoregression matrix B from data using Graphical Lasso.
    
    The autoregression matrix is obtained from the (sparse) precision matrix
    Theta = (Sigma^-1), where Sigma is the covariance estimated by Graphical Lasso.
    Specifically, for j != k,
        B[j, k] = - Theta[j, k] / Theta[j, j],
    and B[j, j] = 0.

    Parameters
    ----------
    data : np.ndarray
        A 2D NumPy array of shape (n_samples, n_features) where each row
        is an observation and each column is one of the variables.
    alpha : float, optional
        The regularization parameter for Graphical Lasso (must be >= 0).
        Higher alpha => more sparsity.

    Returns
    -------
    B_hat : np.ndarray
        Estimated autoregression matrix of shape (n_features, n_features).
    """
    # 1. Fit Graphical Lasso to estimate the sparse precision matrix
    gl_model = GraphicalLasso(alpha=alpha, max_iter=1000)
    gl_model.fit(data)  # data shape = (n_samples, n_features)
    
    # 2. Retrieve the precision matrix Theta (shape: d x d)
    Theta = gl_model.precision_
    
    # 3. Construct B_hat
    d = Theta.shape[0]
    B_hat = np.zeros_like(Theta)
    for j in range(d):
        for k in range(d):
            if j != k:
                B_hat[j, k] = -Theta[j, k] / Theta[j, j]
    
    return B_hat


def training_data_fd_violation(pair: tuple, data: np.ndarray):
    """
    Compute the fraction of rows that violate the FD (parents => child).
    For instance, if all parent columns=1 implies child=1,
    the violation ratio is the number of rows that have parents=1 but child=0
    divided by the total number of rows with parents=1.
    If no rows have all parents=1, we set the violation ratio to 0 by convention.
    
    Parameters
    ----------
    pair : tuple
        A two-element tuple (parents, child).
    data : np.ndarray
        The dataset, shape (n_samples, n_features).
    
    Returns
    -------
    (violation_ratio, debug_info)
        violation_ratio : float
            Fraction of rows violating the FD in [0,1].
        debug_info : dict
            Additional information for debugging.
    """
    parents, child = pair

    if len(parents) == 0:
        # No parents means there's nothing to violate
        return 0.0, {"detail": "No parents specified"}

    # Extract relevant columns
    parents_data = data[:, parents]
    child_data = data[:, child]

    # Rows where all parents == 1
    mask_parents_one = np.all(parents_data == 1, axis=1)
    total_with_parents_one = np.sum(mask_parents_one)
    
    if total_with_parents_one > 0:
        violations = np.sum(mask_parents_one & (child_data == 0))
        violation_ratio = violations / float(total_with_parents_one)
        debug_info = {
            "total_rows_with_parents_one": total_with_parents_one,
            "violations": violations
        }
    else:
        # If no row has all parents=1, set violation=0 by convention
        violation_ratio = 0.0
        debug_info = {"detail": "No rows have all parents=1"}
    
    return violation_ratio, debug_info

def fit_error(pair: tuple, B: np.ndarray, data: np.ndarray) -> tuple[float, float]:
    """
    Compute a mean squared error (MSE) measure for a linear relationship:
        child_data ≈ sum(coeff[i]*parents_data[:, i]) + bias
    
    Steps:
      1) Extract columns for parents and child.
      2) Coefficients come from B[parents, child].
      3) offset = child_data - parents_data.dot(coeff).
      4) bias = mean(offset).
      5) Return MSE after subtracting the bias.
    
    Parameters
    ----------
    pair : tuple
        A two-element tuple (parents, child).
    B : np.ndarray
        Coefficients matrix of shape (n_features, n_features).
    data : np.ndarray
        The dataset, shape (n_samples, n_features).
    
    Returns
    -------
    (mse, bias)
        mse : float
            Mean squared error after subtracting the best bias.
        bias : float
            The offset bias that minimizes the error.
    """
    parents, child = pair
    child_data = data[:, child]
    
    if len(parents) == 0:
        # No parents means we only fit a bias = mean(child_data)
        bias = np.mean(child_data)
        mse = np.mean((child_data - bias)**2)
        return mse, bias
    
    parents_data = data[:, parents]
    coeff = B[parents, child].flatten()  # Get the coefficients for these parents->child
    
    offset = child_data - np.dot(parents_data, coeff)
    bias = np.mean(offset)
    residuals = offset - bias
    mse = np.mean(residuals**2)
    return mse, bias

def get_dependencies(
    heatmap: Optional[np.ndarray] = None,
    score: Optional[str] = "fit_error",
    write_to: Optional[str] = None,
    by_col: bool = True,
    B: Optional[np.ndarray] = None,
    Bs: Optional[list[np.ndarray]] = None,
    data: Optional[np.ndarray] = None
) -> dict[int, list[int]]:
    """
    Gathers dependencies (FDs) by looking at non-zero entries in a matrix.
    - If by_col=True, each column is treated as the child, and row indices with non-zero entries are parents.
    - If by_col=False, each row is treated as the child, and column indices with non-zero entries are parents.

    If 'heatmap' is provided, discover from that. Otherwise, try 'B' or each matrix in 'Bs'.
    You can specify a score ("training_data_fd_vio_ratio" or "fit_error") for each discovered FD.

    Parameters
    ----------
    heatmap : np.ndarray or None
        Matrix from which to discover FDs if provided.
    score : str or None
        Type of scoring ("training_data_fd_vio_ratio" or "fit_error").
    write_to : str or None
        If provided, write discovered FDs to text files with this prefix.
    by_col : bool
        If True, columns are children. If False, rows are children.
    B : np.ndarray or None
        Single matrix if available.
    Bs : list of np.ndarray or None
        List of matrices if available.
    data : np.ndarray or None
        Dataset for scoring functions (required if using training_data_fd_violation or fit_error).

    Returns
    -------
    parent_sets : dict
        A dictionary mapping each child to its parent indices, e.g. { child_index: [parent1, parent2, ...], ... }
    """

    # Choose the scoring function
    if score == "training_data_fd_vio_ratio":
        if data is None:
            print("Warning: data must be provided for training_data_fd_vio_ratio. Using dummy score instead.")
            scoring_func = lambda x: ("n/a", None)
        else:
            scoring_func = lambda fd: training_data_fd_violation(fd, data)
    elif score == "fit_error":
        if data is None or B is None:
            print("Warning: data and B must be provided for fit_error. Using dummy score instead.")
            scoring_func = lambda x: ("n/a", None)
        else:
            scoring_func = lambda fd: fit_error(fd, B, data)
    else:
        scoring_func = lambda x: ("n/a", None)

    parent_sets = {}

    def get_dependencies_helper(B_matrix: np.ndarray, s_func: callable, write_prefix: str, by_col: bool) -> dict[ int, list[int]]:
        """
        Core FD-finding logic. If by_col=True, for each column j, gather row indices i 
        where B_matrix[i, j] != 0 => i -> j.
        If by_col=False, for each row i, gather column indices j where B_matrix[i, j] != 0 => j -> i.
        """
        discovered = {}
        rows, cols = B_matrix.shape

        # Optionally open a file for writing
        if write_prefix is not None:
            suffix = "_by_col" if by_col else "_by_row"
            fd_file = open(write_prefix + suffix + ".txt", 'w')
        else:
            fd_file = None

        if by_col:
            # Each column j is a child
            for j in range(cols):
                parents = [i for i in range(rows) if B_matrix[i, j] != 0]
                if parents:
                    discovered[j] = parents
                    score_val, _ = s_func((parents, j))
                    if fd_file is not None:
                        fd_file.write("{} -> {}\n".format(",".join(map(str, parents)), j))
                    print("{} -> {} (score={})".format(",".join(map(str, parents)), j, score_val))
        else:
            # Each row i is a child
            for i in range(rows):
                parents = [j for j in range(cols) if B_matrix[i, j] != 0]
                if parents:
                    discovered[i] = parents
                    score_val, _ = s_func((parents, i))
                    if fd_file is not None:
                        fd_file.write("{} -> {}\n".format(",".join(map(str, parents)), i))
                    print("{} -> {} (score={})".format(",".join(map(str, parents)), i, score_val))

        if fd_file is not None:
            fd_file.close()

        return discovered

    # Conduct FD discovery from heatmap, B, or Bs
    if heatmap is not None:
        parent_sets = get_dependencies_helper(heatmap, scoring_func, write_to, by_col)
    else:
        if B is not None:
            parent_sets = get_dependencies_helper(B, scoring_func, write_to, by_col)
        elif Bs is not None and len(Bs) > 0:
            for B_mat in Bs:
                result = get_dependencies_helper(B_mat, scoring_func, write_to, by_col)
                parent_sets.update(result)
        else:
            print("No matrix provided for FD discovery.")

    return parent_sets

def run_fdx(df: pd.DataFrame) -> DSLProg:
    df = df.dropna()
    data = df.to_numpy()
    data, retained_columns = BaseLearner.pd_to_np(df)
    data = AuxiliarySampler().sample(data)
    B = estimate_B_via_graphical_lasso(data)
    fds = get_dependencies(heatmap=B, score="training_data_fd_vio_ratio", write_to=None, by_col=True, data=data)
    prog = DSLProg()
    for fd in fds:
        dependant = retained_columns[fd]
        parents = [retained_columns[p] for p in fds[fd]]
        stmt = DSLStmt.create(
            dependent=dependant,
            determinants=parents
        )
        stmt.force_fit(df)
        prog.add_stmt(stmt)
    return prog

# Example usage:
if __name__ == "__main__":
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the input dataset")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    df = df.dropna()
    data = df.to_numpy()
    data, retained_columns = BaseLearner.pd_to_np(df)
    data = AuxiliarySampler().sample(data)
    B = estimate_B_via_graphical_lasso(data)
    fds = get_dependencies(heatmap=B, score="training_data_fd_vio_ratio", write_to=None, by_col=True, data=data)
    logger.info(f"Generated {len(fds)} functional dependencies.")
    logger.info(f"retained_columns: {retained_columns}, fds: {fds}")
    # {0: [4, 5], 1: [2, 3, 5, 7], 2: [1, 3, 5, 7], 3: [1, 2, 5], 4: [0], 5: [0, 1, 2, 3, 7], 6: [8], 7: [1, 2, 5, 8], 8: [6, 7]}
    for fd in fds:
        dependant = retained_columns[fd]
        parents = [retained_columns[p] for p in fds[fd]]
        logger.info(f"{parents} -> {dependant}")