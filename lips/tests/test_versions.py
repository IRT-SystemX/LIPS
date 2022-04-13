"""
Test compatibility versions for lips

Check solver versions, etc.
"""
import unittest
from packaging import version

class TestVersion(unittest.TestCase):
    """
    version verification
    """
    def test_grid2op(self):
        """
        Verifying grid2op version
        """
        try:
            import grid2op
            self.assertTrue(version.parse(grid2op.__version__) >= version.parse("1.4.0"))
            #print(grid2op.__version__)
        except ImportError:
            self.assertRaises(ImportError)

    def test_lightsim2grid(self):
        """
        Verifying lightsim2grid version
        """
        try:
            import lightsim2grid
            self.assertTrue(version.parse(lightsim2grid.__version__) >= version.parse("0.6.0"))
            #print(grid2op.__version__)
        except ImportError:
            self.assertRaises(ImportError)

suite = unittest.TestLoader().loadTestsFromTestCase(TestVersion)
unittest.TextTestRunner(verbosity=2).run(suite)