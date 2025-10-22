#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 21:50
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   01_jieba_01_cut.py
# @Desc     :   

from utils.config import CONFIG
from utils.helper import read_file
from utils.JB import cut_accuracy, cut_full, cut_search


def main() -> None:
    """ Main Function """
    content: str = read_file(CONFIG.FILEPATHS.EXAMPLE_PAPER_ZH)

    acc = cut_accuracy(content)
    print(acc)
    print()

    full = cut_full(content)
    print(full)
    print()

    search = cut_search(content)
    print(search)
    print()


if __name__ == "__main__":
    main()
