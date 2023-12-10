import unittest

import pandas as pd

from nsyn.dsl.prog import DSLProg
from nsyn.learner import PC
from nsyn.sampler import AuxiliarySampler
from nsyn.search import Search
from nsyn.util.logger import get_logger

logger = get_logger(name="nsyn.tests.test_search")


class TestSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.test_data = pd.DataFrame(
            data=[
                ["California", "San Francisco"],
                ["California", "Los Angeles"],
                ["California", "San Diego"],
                ["California", "San Jose"],
                ["California", "San Francisco"],
                ["California", "Los Angeles"],
                ["California", "San Diego"],
                ["Texas", "Houston"],
                ["Texas", "Dallas"],
                ["Texas", "Austin"],
                ["Texas", "Houston"],
                ["Texas", "Dallas"],
                ["Washington", "Seattle"],
                ["Washington", "Spokane"],
                ["Washington", "Seattle"],
                ["Virginia", "Richmond"],
                ["Virginia", "Virginia Beach"],
                ["Virginia", "Richmond"],
                ["Virginia", "Virginia Beach"],
            ]
            * 100
            + [["Texs", "Austin"]],
            columns=["State", "City"],
        )
        self.sampler = AuxiliarySampler()
        self.learner = PC()
        self.search = Search.create(
            learning_algorithm=self.learner,
            sampling_algorithm=self.sampler,
            input_data=self.test_data,
            epsilon=0.1,
            min_support=10,
        )

    def test_search(self) -> None:
        # This test should verify that the search method returns a valid program.
        # You might need to adjust this depending on the exact logic of your function.
        result = self.search.run()
        # Add assertions here to check for specific logic
        self.assertTrue(isinstance(result, DSLProg) and len(result.stmts) == 1)
        logger.info(result)
