#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/22 01:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   THU.py
# @Desc     :   

from thulac import thulac

from utils.decorator import timer


@timer
def cut_pos(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """ Cut text using THULAC
    :param text: text to cut
    :return: list of tuples of cut words and their POS tags
    """
    # Initialize THULAC with segmentation only
    thu = thulac(seg_only=False)
    words_tag: list[tuple[str, str]] = thu.cut(text)
    words: list[str] = [word for word, tag in words_tag]

    print(f"The text has been cut into {len(words)} words using THULAC.")

    return words_tag, words


@timer
def cut_only(text: str) -> list[str]:
    """ Cut text using THULAC without POS tags
    :param text: text to cut
    :return: list of cut words
    """
    # Initialize THULAC with segmentation only
    thu = thulac(seg_only=True)
    words: list[str] = thu.cut(text)

    print(f"The text has been cut into {len(words)} words using THULAC (without POS tags).")

    return words
