from fuzzylite.term import Term
from fuzzylite.rule import Rule
from fuzzylite.exporter import FllExporter

import unittest


class TestFllExporter(unittest.TestCase):
    def test_to_string(self):
        print("Testing FllExporter")
        self.assertEqual(FllExporter().term(Term("X", 1.0)), "term: X Term ")
        self.assertEqual(FllExporter().rule(Rule("if x then y")), "rule: if x then y")


if __name__ == '__main__':
    unittest.main()
