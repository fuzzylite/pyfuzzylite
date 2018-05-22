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

from .activation import *
from .defuzzifier import *
from .engine import *
from .exporter import *
from .factory import *
from .hedge import *
from .importer import *
from .library import *
from .norm import *
from .operation import *
from .rule import *
from .term import *
from .variable import *

name = "fuzzylite"

__library_global: Library = None
__library_thread: threading.local = None


def __global_library() -> Library:
    global __library_global, __library_thread

    if not __library_global:
        __library_global = Library()

    if __library_thread:
        if 'library' in vars(__library_thread):
            del __library_thread.library
        __library_thread = None

    return __library_global


def __thread_local_library() -> Library:
    global __library_global, __library_thread

    if not __library_thread:
        __library_thread = threading.local()

    if 'library' not in vars(__library_thread):
        __library_thread.library = Library()

    if __library_global:
        __library_global = None

    return __library_thread.library


library = __global_library


class StorageMode(Enum):
    GLOBAL, THREAD_LOCAL = range(2)


def set_storage_mode(mode: StorageMode) -> None:
    global library
    if mode == StorageMode.GLOBAL:
        library = __global_library
    elif mode == StorageMode.THREAD_LOCAL:
        library = __thread_local_library
    else:
        raise ValueError(f"unexpected storage mode: {mode}")
    library()


def get_storage_mode() -> StorageMode:
    global library
    if library == __global_library:
        result = StorageMode.GLOBAL
    elif library == __thread_local_library:
        result = StorageMode.THREAD_LOCAL
    else:
        raise ValueError(f"unexpected storage mode: {library}")
    return result
