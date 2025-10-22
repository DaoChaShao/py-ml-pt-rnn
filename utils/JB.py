#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/22 01:05
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   JB.py
# @Desc     :

from jieba import (lcut, lcut_for_search,
                   analyse,
                   posseg)
from pandas import DataFrame
from re import compile

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
def cut_full(content: str) -> list[str]:
    """ Cut text for jieba full mode result
    :param content: text to cut
    :return: list of cut words
    """
    words: list[str] = lcut(content, cut_all=True)

    print(f"The text has been cut into {len(words)} words.")

    return words


@timer
def cut_search(content: str) -> list[str]:
    """ Cut text for jieba search engine mode result
    :param content: text to cut
    :return: list of cut words
    """
    words: list[str] = lcut_for_search(content)

    print(f"The text has been cut into {len(words)} words.")

    return words


@timer
def cut_pos(content: str, restriction: str, top_k: int = 10):
    """ Cut text and get part of speech"""
    dictionary: dict[str, str] = {}

    words = posseg.lcut(content, use_paddle=True)
    # print(words)

    pattern = compile(restriction)
    for word, pos in words:
        if pattern.match(word) and len(word.strip()) > 1:
            dictionary[word] = pos

    cols = ["word", "pos"]
    df = DataFrame(dictionary.items(), columns=cols)[:top_k]
    print(f"Cut POS Results:\n{df}")

    return dictionary, df


@timer
def extract_tfidf_weights(content: str, top_k: int = 10, pos: bool = False) -> tuple[list, DataFrame]:
    """ Extract TF-IDF weights from text
    :param content: the text to cut
    :param top_k: number of top keywords to extract
    :param pos: whether to filter by part of speech
    :return: list of tuples containing words and their weights, and a DataFrame of the same
    """
    mask: tuple = ("v", "vn", "a", "d") if pos else ()
    tags = analyse.extract_tags(
        content,
        topK=top_k,
        withWeight=True,
        allowPOS=mask,
    )

    cols: list[str] = ["word", "weight"]
    df: DataFrame = DataFrame(data=tags, columns=cols)

    print(f"TF-IDF words weights:\n{df}")
    print(f"Extracted {len(tags)} tags using TF-IDF.")

    return tags, df


@timer
def extract_textrank_words(content: str, top_k: int = 10, pos: bool = False) -> tuple[list, DataFrame]:
    """ Extract TextRank words from text
    :param content: the text to cut
    :param top_k: number of top keywords to extract
    :param pos: whether to filter by part of speech
    :return: list of tuples containing words and their weights, and a DataFrame of the same
    """
    mask: tuple = ("v", "vn", "a", "d") if pos else ()
    words = analyse.textrank(
        content,
        topK=top_k,
        withWeight=True,
        allowPOS=mask,
    )

    cols: list[str] = ["word", "weight"]
    df: DataFrame = DataFrame(data=words, columns=cols)

    print(f"TexTank Words Weights:\n{df}")
    print(f"Extracted {len(words)} words using TextRank.")

    return words, df
