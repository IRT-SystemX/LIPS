"""
Test case for lips
"""
import unittest
import pathlib as pl

class TestData(unittest.TestCase):
    """
    Testing the benchmark classes
    """
    def test(self):
        """
        testing if data for first benchmark has been provided
        """
        path = pl.Path("reference_data/NeuripsBenchmark1/train")
        self.assertEqual((str(path), path.is_dir()), (str(path), True))

#if __name__ == "__main__":
#    unittest.main(verbosity=2)

suite = unittest.TestLoader().loadTestsFromTestCase(TestData)
unittest.TextTestRunner(verbosity=2).run(suite)