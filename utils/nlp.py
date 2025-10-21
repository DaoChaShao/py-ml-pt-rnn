#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 21:59
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   nlp.py
# @Desc     :


from re import compile

from utils.decorator import timer


@timer
def regular_chinese(words: list[str]) -> list[str]:
    """ Retain only Chinese characters in the list of words
    :param words: list of words to process
    :return: list of words containing only Chinese characters
    """
    pattern = compile(r"[\u4e00-\u9fa5]+")

    chinese: list[str] = [word for word in words if pattern.fullmatch(word)]

    print(f"Retained {len(chinese)} Chinese words from the original {len(words)} words.")

    return chinese
