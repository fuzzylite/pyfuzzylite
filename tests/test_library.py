"""
 pyfuzzylite (TM), a fuzzy logic control library in Python.
 Copyright (C) 2010-2017 FuzzyLite Limited. All rights reserved.
 Author: Juan Rada-Vilela, Ph.D. <jcrada@fuzzylite.com>

 This file is part of pyfuzzylite.

 pyfuzzylite is free software: you can redistribute it and/or modify it under
 the terms of the FuzzyLite License included with the software.

 You should have received a copy of the FuzzyLite License along with
 pyfuzzylite. If not, see <http://www.fuzzylite.com/license/>.

 pyfuzzylite is a trademark of FuzzyLite Limited
 fuzzylite is a registered trademark of FuzzyLite Limited.
"""

import threading
import unittest

import fuzzylite
from fuzzylite.operation import Op


class TestLibrary(unittest.TestCase):
    pass

    # def test_storage_mode(self):
    #     # Default storage mode is global
    #     self.assertEqual(fuzzylite.get_storage_mode(), fuzzylite.StorageMode.GLOBAL)
    #
    #     # Change to thread local
    #     fuzzylite.set_storage_mode(fuzzylite.StorageMode.THREAD_LOCAL)
    #     self.assertEqual(fuzzylite.get_storage_mode(), fuzzylite.StorageMode.THREAD_LOCAL)
    #
    #     # Change to global
    #     fuzzylite.set_storage_mode(fuzzylite.StorageMode.GLOBAL)
    #     self.assertEqual(fuzzylite.get_storage_mode(), fuzzylite.StorageMode.GLOBAL)
    #
    # def test_thread_local(self):
    #     fuzzylite.set_storage_mode(fuzzylite.StorageMode.THREAD_LOCAL)
    #
    #     # Default behavior is three decimals
    #     self.assertEqual(Op.str(20.31982), "20.320")
    #
    #     # Decimals in this thread are printed with 6 decimals from now on
    #     fuzzylite.library().decimals = 6
    #     self.assertEqual(Op.str(20.31982), "20.319820")
    #
    #     global thread_results
    #     thread_results = []
    #
    #     def thread_test():
    #         global thread_results
    #         # Default behavior is three decimals
    #         thread_results.append(Op.str(20.31982))
    #
    #         # Decimals in this thread are printed with 9 decimals from now on
    #         fuzzylite.library().decimals = 9
    #         thread_results.append(Op.str(20.31982))
    #
    #     thread = threading.Thread(target=thread_test)
    #     thread.start()
    #     thread.join()
    #     self.assertListEqual(["20.320", "20.319820000"], thread_results)
    #
    #     self.assertEqual(Op.str(20.31982), "20.319820")


if __name__ == '__main__':
    unittest.main()
