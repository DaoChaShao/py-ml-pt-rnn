#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 23:09
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   01_jieba_04_id2word.py
# @Desc     :   

from utils.config import CONFIG
from utils.helper import read_file
from utils.JB import cut_accuracy
from utils.nlp import regular_chinese


def main() -> None:
    """ Main Function """
    content: str = read_file(CONFIG.FILEPATHS.EXAMPLE_PAPER_ZH)

    # Read stopwords from file without repeats
    stopwords: set[str] = set(read_file(CONFIG.FILEPATHS.EXAMPLE_STOPWORDS_ZH).splitlines())

    words = cut_accuracy(content)
    # Remove stopwords from the cut words
    words_stop: list[str] = [word for word in words if word not in stopwords]

    # Use regex to filter only Chinese characters
    vocabs = regular_chinese(words_stop)

    # Avoid repeats by converting to set and back to list, which is call 'id2word' technique
    id2word = list(set(vocabs))
    print(id2word)

    # word to id mapping
    word2id = {word: index for index, word in enumerate(id2word)}
    print(word2id)


if __name__ == "__main__":
    main()
