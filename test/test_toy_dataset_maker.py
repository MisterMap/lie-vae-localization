import unittest

from mvae.utils.toy_dataset_maker import ToyDatasetMaker


class TestToyDatasetMaker(unittest.TestCase):
    def setUp(self) -> None:
        self._maker = ToyDatasetMaker(10, 10, 1.6, 0.05, 0.25, 60)

    def test_snail_trajectory(self):
        trajectory = self._maker.snail_trajectory(5, 4)
        print(trajectory)
