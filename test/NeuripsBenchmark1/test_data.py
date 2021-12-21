import unittest
import pathlib as pl

class TestBenchmark1(unittest.TestCase):

    def test(self):
        path = pl.Path("reference_data/NeuripsBenchmark1/train")
        self.assertEquals((str(path), path.is_dir()), (str(path), True))

if __name__ == "__main__":
    unittest.main(verbosity=2)