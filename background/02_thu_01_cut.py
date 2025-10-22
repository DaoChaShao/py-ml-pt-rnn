#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 23:52
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_thu_01_cut.py
# @Desc     :   

from utils.config import EXAMPLES_CHINESE_PATH
from utils.helper import read_file
from utils.THU import cut_only, cut_pos


def main() -> None:
    """ Main Function """
    content = read_file(EXAMPLES_CHINESE_PATH)

    words = cut_only(content)
    vocabs = cut_pos(content)

    print(words)
    print(vocabs)


if __name__ == "__main__":
    main()
