#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 23:02
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   01_jieba_03_re.py
# @Desc     :

from utils.config import EXAMPLES_CHINESE_PATH, EXAMPLES_CHINESE_STOPWORDS_PATH
from utils.helper import read_file
from utils.JB import cut_accuracy
from utils.nlp import regular_chinese


def main() -> None:
    """ Main Function """
    content: str = read_file(EXAMPLES_CHINESE_PATH)

    # Read stopwords from file without repeats
    stopwords: set[str] = set(read_file(EXAMPLES_CHINESE_STOPWORDS_PATH).splitlines())

    words = cut_accuracy(content)
    # Remove stopwords from the cut words
    words_stop: list[str] = [word for word in words if word not in stopwords]

    # Use regex to filter only Chinese characters
    vocabs = regular_chinese(words_stop)

    print(len(words), words)
    print(len(words_stop), words_stop)
    print(len(vocabs), vocabs)


if __name__ == "__main__":
    main()
