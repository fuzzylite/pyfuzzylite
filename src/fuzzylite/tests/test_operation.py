import math
import unittest

import fuzzylite.operation as op


class TestOperation(unittest.TestCase):
    def test_validName(self):
        self.assertEqual(op.valid_name("  xx  "), "xx")  # trims
        self.assertEqual(op.valid_name("   ~!@#$%^&*()+{}[]:;\"'<>?/,   "), "unnamed")
        self.assertEqual(op.valid_name("abc123_.ABC"), "abc123_.ABC")
        self.assertEqual(op.valid_name("      "), "unnamed")

    def test_str_(self):
        self.assertEqual(op.str_(0.3), "0.300")
        self.assertEqual(op.str_(-0.3), "-0.300")
        self.assertEqual(op.str_(3), "3.000")
        self.assertEqual(op.str_(3.0001), "3.000")

        self.assertEqual(op.str_(math.inf), "inf")
        self.assertEqual(op.str_(-math.inf), "-inf")
        self.assertEqual(op.str_(math.nan), "nan")

        op.decimals = 5
        self.assertEqual(op.str_(0.3), "0.30000")

        op.decimals = 0
        self.assertEqual(op.str_(0.3), "0")

    def test_scale(self):
        self.assertEqual(op.scale(0, 0, 1, -10, 10), -10.0)
        self.assertEqual(op.scale(.5, 0, 1, -10, 10), 0.0)
        self.assertEqual(op.scale(1, 0, 1, -10, 10), 10)

        self.assertEqual(op.scale(0, 0, 1, 0, 10), 0.0)
        self.assertEqual(op.scale(.5, 0, 1, 0, 10), 5.0)
        self.assertEqual(op.scale(1, 0, 1, 0, 10), 10)

        self.assertEqual(op.scale(-1, 0, 1, 0, 10), -10.0)
        self.assertEqual(op.scale(2, 0, 1, 0, 10), 20)

        self.assertEqual(math.isnan(op.scale(math.nan, 0, 1, 0, 10)), True)
        self.assertEqual(op.scale(math.inf, 0, 1, 0, 10), math.inf)
        self.assertEqual(op.scale(-math.inf, 0, 1, 0, 10), -math.inf)


if __name__ == '__main__':
    unittest.main()
