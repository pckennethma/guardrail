import time
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import pandas as pd
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import chisq

from nsyn.sampler import AbstractSampler
from nsyn.util.base_model import BaseModel
from nsyn.util.convert import ggraph2mec
from nsyn.util.logger import get_logger
from nsyn.util.mec import MEC

logger = get_logger(name="nsyn.learner")


def learner_timer(func: Callable[..., MEC]) -> Callable[..., MEC]:
    def wrapper(*args: Any, **kwargs: Any) -> MEC:
        start = time.time()
        logger.info(f"Starting {func.__name__}...")
        mec = func(*args, **kwargs)
        logger.info(f"Finished {func.__name__} in {time.time() - start:.2f} seconds.")
        return mec

    return wrapper


class BaseLearner(ABC):
    """
    Abstract base class for a learning algorithm in the NSYN framework.

    This class defines the interface for learning algorithms. Subclasses must implement the learn method, which is intended to return a Markov Equivalence Class (MEC) based on the input data and a sampling protocol.

    Methods:
        learn: Abstract method to learn and return a MEC based on the input data and sampling protocol.
        pd_to_np: Utility method to convert a pandas DataFrame to a numpy array, suitable for processing in the learn method.
    """

    @abstractmethod
    def learn(
        self, data: np.ndarray | pd.DataFrame, sampling_protocol: AbstractSampler
    ) -> MEC:
        """
        Abstract method to learn from the data.

        Args:
            data (np.ndarray | pd.DataFrame): The input data for the learning algorithm.
            sampling_protocol (AbstractSampler): The sampling protocol to use on the data.

        Returns:
            MEC: A Markov Equivalence Class derived from the data.
        """
        ...

    def pd_to_np(self, data: pd.DataFrame) -> np.ndarray:
        """
        Converts a pandas DataFrame to a numpy array by dropping numeric columns and converting
        categorical values to integers.

        Args:
            data (pd.DataFrame): The input DataFrame to be converted.

        Returns:
            np.ndarray: The resulting numpy array after conversion.
        """

        # First drops all numeric columns
        # then converts the remaining columns to categorical values.
        # Finally, converts the categorical values to integers and returns the numpy array.

        # Drop all numeric columns
        data = data.select_dtypes(exclude=["number"])

        # Convert all columns to categorical values
        data = data.astype("category")

        # Convert the categorical values to integers
        data = data.apply(lambda x: x.cat.codes)

        return data.to_numpy()


class PC(BaseLearner, BaseModel):
    """
    A subclass of BaseLearner implementing the PC algorithm for learning a MEC.

    The PC (Peter-Clark) algorithm is a constraint-based method for learning causal structures from data. This class implements the PC algorithm in the context of the NSYN framework.

    Methods:
        learn: Implements the PC algorithm to learn and return a MEC from the given data and sampling protocol.
    """

    @learner_timer
    def learn(
        self, data: np.ndarray | pd.DataFrame, sampling_protocol: AbstractSampler
    ) -> MEC:
        """
        Implements the PC algorithm to learn a Markov Equivalence Class from the data.

        Args:
            data (np.ndarray | pd.DataFrame): The input data, either as a numpy array or a pandas DataFrame.
            sampling_protocol (AbstractSampler): The sampling protocol to use on the data.

        Returns:
            MEC: A Markov Equivalence Class derived from the data using the PC algorithm.
        """
        if isinstance(data, pd.DataFrame):
            data = self.pd_to_np(data)
        transformed_data = sampling_protocol.sample(data)
        logger.info(f"PC: transformed_data.shape = {transformed_data.shape}")
        cg = pc(transformed_data, indep_test=chisq)
        return ggraph2mec(cg.G)


class GES(BaseLearner, BaseModel):
    """
    A subclass of BaseLearner implementing the PC algorithm for learning a MEC.

    The GES (Greedy Equivalence Search) algorithm is a score-based method for learning causal structures from data. This class implements the GES algorithm in the context of the NSYN framework.

    Methods:
        learn: Implements the PC algorithm to learn and return a MEC from the given data and sampling protocol.
    """

    @learner_timer
    def learn(
        self, data: np.ndarray | pd.DataFrame, sampling_protocol: AbstractSampler
    ) -> MEC:
        """
        Implements the PC algorithm to learn a Markov Equivalence Class from the data.

        Args:
            data (np.ndarray | pd.DataFrame): The input data, either as a numpy array or a pandas DataFrame.
            sampling_protocol (AbstractSampler): The sampling protocol to use on the data.

        Returns:
            MEC: A Markov Equivalence Class derived from the data using the PC algorithm.
        """
        if isinstance(data, pd.DataFrame):
            data = self.pd_to_np(data)
        transformed_data = sampling_protocol.sample(data)
        logger.info(f"GES: transformed_data.shape = {transformed_data.shape}")
        cg = ges(transformed_data, score_func="local_score_BDeu", maxP=3)["G"]
        assert isinstance(cg, GeneralGraph), f"cg is not a GeneralGraph: {type(cg)}"
        return ggraph2mec(cg)
