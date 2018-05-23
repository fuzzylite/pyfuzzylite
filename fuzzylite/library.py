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

import logging

from .factory import FactoryManager


class Library(object):

    def __init__(self, decimals=3, absolute_tolerance=1e-6,
                 factory_manager=FactoryManager()):
        self.decimals = decimals
        self.absolute_tolerance = absolute_tolerance
        self.factory_manager = factory_manager
        self.logger = logging.getLogger("fuzzylite")

    @property
    def name(self) -> str:
        return "pyfuzzylite"

    @property
    def version(self) -> str:
        return "7.0"

    @property
    def license(self):
        return "GNU General Public License v3.0"

    @property
    def description(self) -> str:
        return "a fuzzy logic control in Python"

    @property
    def author(self):
        return "Juan Rada-Vilela, Ph.D."

    @property
    def author_email(self):
        return "jcrada@fuzzylite.com"

    @property
    def company(self):
        return "FuzzyLite Limited"

    @property
    def website(self):
        return "https://www.fuzzylite.com/"

    @property
    def debugging(self) -> bool:
        return self.logger.level == logging.DEBUG
