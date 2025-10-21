#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 21:40
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   helper.py
# @Desc     :   

from random import seed as rnd_seed, getstate, setstate
from time import perf_counter


class Timer(object):
    """ timing code blocks using a context manager """

    def __init__(self, description: str = None, precision: int = 5):
        """ Initialise the Timer class
        :param description: the description of a timer
        :param precision: the number of decimal places to round the elapsed time
        """
        self._description: str = description
        self._precision: int = precision
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self):
        """ Start the timer """
        self._start = perf_counter()
        print("*" * 50)
        print(f"{self._description} has started.")
        print("-" * 50)
        return self

    def __exit__(self, *args):
        """ Stop the timer and calculate the elapsed time """
        self._end = perf_counter()
        self._elapsed = self._end - self._start

        print("-" * 50)
        print(f"{self._description} took {self._elapsed:.{self._precision}f} seconds.")
        print("*" * 50)

    def __repr__(self):
        """ Return a string representation of the timer """
        if self._elapsed != 0.0:
            return f"{self._description} took {self._elapsed:.{self._precision}f} seconds."
        return f"{self._description} has NOT started."


class Beautifier(object):
    """ beautifying code blocks using a context manager """

    def __init__(self, description: str = None):
        """ Initialise the Beautifier class
        :param description: the description of a beautifier
        """
        self._description: str = description

    def __enter__(self):
        """ Start the beautifier """
        print("*" * 50)
        print(f"The block named {self._description!r} is starting:")
        print("-" * 50)
        return self

    def __exit__(self, *args):
        """ Stop the beautifier """
        print("-" * 50)
        print(f"The block named {self._description!r} has completed.")
        print("*" * 50)
        print()


class RandomSeed:
    """ Setting random seed for reproducibility """

    def __init__(self, description: str, seed: int = 27):
        """ Initialise the RandomSeed class
        :param description: the description of a random seed
        :param seed: the seed value to be set
        """
        self._description: str = description
        self._seed: int = seed
        self._previous_py_seed = None

    def __enter__(self):
        """ Set the random seed """
        # Save the previous random seed state
        self._previous_py_seed = getstate()

        # Set the new random seed
        rnd_seed(self._seed)

        print("*" * 50)
        print(f"{self._description!r} has been set randomness {self._seed}.")
        print("-" * 50)

        return self

    def __exit__(self, *args):
        """ Exit the random seed context manager """
        # Restore the previous random seed state
        if self._previous_py_seed is not None:
            setstate(self._previous_py_seed)

        print("-" * 50)
        print(f"{self._description!r} has been restored to previous randomness.")
        print("*" * 50)
        print()

    def __repr__(self):
        """ Return a string representation of the random seed """
        return f"{self._description!r} is set to randomness {self._seed}."


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        print(f"The content:\n{content}")
    return content
