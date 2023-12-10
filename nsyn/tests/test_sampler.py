import unittest

import numpy as np

from nsyn.sampler import AuxiliarySampler


class TestCircShift(unittest.TestCase):
    def setUp(self) -> None:
        self.test_data = np.array([[1, 2], [1, 4], [3, 4], [3, 2]])
        self.expected_shape = (8, 2)
        self.sampler = AuxiliarySampler()

    def test_output_shape(self) -> None:
        result = self.sampler._circ_shift(self.test_data)
        self.assertEqual(result.shape, self.expected_shape)

    def test_circular_shift_logic(self) -> None:
        # This test should verify that the circular shift logic is correctly applied.
        # You might need to adjust this depending on the exact logic of your function.
        result = self.sampler._circ_shift(self.test_data)
        # Add assertions here to check for specific logic
        expected_result = np.array(
            [
                [True, False],
                [False, False],
                [True, False],
                [False, False],
                [False, True],
                [True, False],
                [False, True],
                [True, False],
            ]
        )

        self.assertTrue(
            np.all(result == expected_result),
            msg=f"Expected {expected_result}, got {result}",
        )


if __name__ == "__main__":
    unittest.main()
