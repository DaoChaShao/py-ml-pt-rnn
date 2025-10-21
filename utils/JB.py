#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/22 01:05
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   JB.py
# @Desc     :

from jieba import lcut, lcut_for_search, analyse
from pandas import DataFrame

from utils.decorator import timer


@timer
def cut_accuracy(text: str) -> list[str]:
    """ Cut text for jieba accuracy result
    :param text: text to cut
    :return: list of cut words
    """
    words: list[str] = lcut(text, cut_all=False)

    print(f"The text has been cut into {len(words)} words.")

    return words


@timer
def cut_full(text: str) -> list[str]:
    """ Cut text for jieba full mode result
    :param text: text to cut
    :return: list of cut words
    """
    words: list[str] = lcut(text, cut_all=True)

    print(f"The text has been cut into {len(words)} words.")

    return words


@timer
def cut_search(text: str) -> list[str]:
    """ Cut text for jieba search engine mode result
    :param text: text to cut
    :return: list of cut words
    """
    words: list[str] = lcut_for_search(text)

    print(f"The text has been cut into {len(words)} words.")

    return words


@timer
def extract_tfidf_weights(text: str, top_k: int = 10, pos: bool = False) -> tuple[list[tuple[str, float]], DataFrame]:
    """ Extract TF-IDF weights from text
    :param text: the text to cut
    :param top_k: number of top keywords to extract
    :param pos: whether to filter by part of speech
    :return: list of tuples containing words and their weights, and a DataFrame of the same
    """
    mask: tuple = ("v", "vn", "a", "d") if pos else ()
    tags = analyse.extract_tags(
        text,
        topK=top_k,
        withWeight=True,
        allowPOS=mask,
    )

    cols = ["word", "weight"]
    dataframe = DataFrame(data=tags, columns=cols)

    print(f"TF-IDF words weights:\n{dataframe}")
    print(f"Extracted {len(tags)} tags using TF-IDF.")

    return tags, dataframe


@timer
def extract_textrank_words(text: str, top_k: int = 10, pos: bool = False) -> tuple[list[tuple[str, float]], DataFrame]:
    """ Extract TextRank words from text
    :param text: the text to cut
    :param top_k: number of top keywords to extract
    :param pos: whether to filter by part of speech
    :return: list of tuples containing words and their weights, and a DataFrame of the same
    """
    mask: tuple = ("v", "vn", "a", "d") if pos else ()
    words = analyse.textrank(
        text,
        topK=top_k,
        withWeight=True,
        allowPOS=mask,
    )

    cols = ["word", "weight"]
    dataframe = DataFrame(data=words, columns=cols)

    print(f"TexTank Words Weights:\n{dataframe}")
    print(f"Extracted {len(words)} words using TextRank.")

    return words, dataframe
