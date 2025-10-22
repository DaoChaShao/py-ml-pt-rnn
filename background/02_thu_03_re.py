#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/22 00:54
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_thu_03_re.py
# @Desc     :   

from utils.config import CONFIG
from utils.helper import read_file
from utils.nlp import regular_chinese
from utils.THU import cut_only


def main() -> None:
    """ Main Function """
    content = read_file(CONFIG.FILEPATHS.EXAMPLE_PAPER_ZH)
    words = cut_only(content)

    stopwords: set[str] = set(read_file(CONFIG.FILEPATHS.EXAMPLE_STOPWORDS_ZH).splitlines())
    # print(stopwords)
    vocabs = [word for subwords in words for word in subwords if word not in stopwords]
    # print(vocabs)

    regular = regular_chinese(vocabs)
    print(regular)


if __name__ == "__main__":
    main()
