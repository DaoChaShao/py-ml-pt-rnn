#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 21:10
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   main.py
# @Desc     :

from utils.config import EXAMPLES_CHINESE_PATH
from utils.helper import read_file


def main() -> None:
    """ Main Function """
    content = read_file(EXAMPLES_CHINESE_PATH)


if __name__ == "__main__":
    main()
