#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 22:49
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   01_jieba_02_stopwords.py
# @Desc     :   

from utils.config import CONFIG
from utils.helper import read_file
from utils.JB import cut_accuracy


def main() -> None:
    """ Main Function """
    content: str = read_file(CONFIG.FILEPATHS.EXAMPLE_PAPER_ZH)

    # Read stopwords from file without repeats
    stopwords: set[str] = set(read_file(CONFIG.FILEPATHS.EXAMPLE_STOPWORDS_ZH).splitlines())
    print(stopwords)

    words = cut_accuracy(content)
    # Remove stopwords from the cut words
    vocabs: list[str] = [word for word in words if word not in stopwords]
    print(words)
    print(vocabs)


if __name__ == "__main__":
    main()
