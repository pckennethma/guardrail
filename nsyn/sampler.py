from abc import ABC, abstractmethod

import numpy as np

from nsyn.util.base_model import BaseModel


class AbstractSampler(ABC):
    """
    Abstract base class for a sampler.

    This class defines the interface for samplers in the NSYN framework. Subclasses must implement the sample method.

    Methods:
        sample: Abstract method that takes an array and returns a sampled array.
    """

    @abstractmethod
    def sample(self, data: np.ndarray) -> np.ndarray:
        """
        Abstract method to sample data.

        Args:
            data (np.ndarray): The input data array to be sampled.

        Returns:
            np.ndarray: The sampled data array.
        """
        ...


class IdentitySampler(AbstractSampler, BaseModel):
    """
    An identity sampler that returns the input data unchanged.

    Inherits from AbstractSampler and BaseModel. Implements the sample method by returning the input data as is.

    Methods:
        sample: Returns the input data without any changes.
    """

    def sample(self, data: np.ndarray) -> np.ndarray:
        """
        Returns the input data unchanged.

        Args:
            data (np.ndarray): The input data array.

        Returns:
            np.ndarray: The same input data array without any modifications.
        """
        return data


class AuxiliarySampler(AbstractSampler, BaseModel):
    """
    An auxiliary sampler that samples data based on a specified generation strategy.

    Inherits from AbstractSampler and BaseModel. Currently supports 'CircularShift' as the generation strategy.

    Attributes:
        generation_strategy (str): The strategy used for data generation. Default is "CircularShift".

    Methods:
        sample: Samples data based on the specified generation strategy.
        _circ_shift: A helper method that performs circular shift sampling on the data.
    """

    generation_strategy: str = "CircularShift"

    def sample(self, data: np.ndarray) -> np.ndarray:
        """
        Samples data based on the specified generation strategy.

        Args:
            data (np.ndarray): The input data array to be sampled.

        Returns:
            np.ndarray: The sampled data array based on the generation strategy.

        Raises:
            NotImplementedError: If the generation strategy is not implemented.
        """
        if self.generation_strategy == "CircularShift":
            return self._circ_shift(input_data=data)
        else:
            raise NotImplementedError(
                "Generation strategy {} is not implemented.".format(
                    self.generation_strategy
                )
            )

    def _circ_shift(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs circular shift sampling on the data.

        Args:
            input_data (np.ndarray): The input data array to be transformed.

        Returns:
            np.ndarray: The transformed data array after applying circular shift.
        """

        # D is a numpy ndarray with n rows and k columns
        # We need to return a ndarray Dt with n * k rows and k columns

        n, k = input_data.shape  # Get the number of rows (n) and columns (k)

        # Shuffle the rows of D
        np.random.shuffle(input_data)

        # Initialize the output ndarray Dt
        transformed_data = np.zeros((n * k, k), dtype=bool)

        # Comparison and population of transformed_data
        for i in range(k):
            # Sort D by the current column
            sorted_data = input_data[input_data[:, i].argsort()]

            # Perform a circular shift of rows by 1 in Di
            shifted_data = np.roll(sorted_data, -1, axis=0)

            # Vectorized comparison of Di and Di+1
            transformed_data[i * n : (i + 1) * n] = sorted_data == shifted_data

        return transformed_data
