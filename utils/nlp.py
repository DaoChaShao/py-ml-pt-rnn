#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 21:59
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   nlp.py
# @Desc     :

from collections import Counter
from re import compile
from pandas import DataFrame
from stanfordnlp import Pipeline, download

from utils.decorator import timer


@timer
def regular_chinese(words: list[str]) -> list[str]:
    """ Retain only Chinese characters in the list of words
    :param words: list of words to process
    :return: list of words containing only Chinese characters
    """
    pattern = compile(r"[\u4e00-\u9fa5]+")

    chinese: list[str] = [word for word in words if pattern.match(word)]

    print(f"Retained {len(chinese)} Chinese words from the original {len(words)} words.")

    return chinese


@timer
def count_words_frequency(words: list[str], top_k: int = 10) -> tuple[Counter, DataFrame]:
    """ Get frequency of Chinese words
    :param words: list of words to process
    :param top_k: number of top frequent words to return
    :return: DataFrame containing words and their frequencies
    """
    # Get word frequency using Counter
    counter = Counter(words)

    cols: list[str] = ["word", "frequency"]
    df: DataFrame = DataFrame(counter.items(), columns=cols)[:top_k]
    sorted_df = df.sort_values(by="frequency", ascending=False)

    print(f"Word Frequency Results:\n{sorted_df}")

    return counter, sorted_df


@timer
def snlp_analysis(content: str, mode: str = "cut", language: str = "zh", is_gpu: bool = False) -> list[tuple[str, str]]:
    """ Perform part-of-speech tagging using StanfordNLP
    :param content: text content to process
    :param mode: processing mode, e.g., 'cut' for word segmentation, 'pos' for get words and their pos, 'full' for full text processing,
    :param language: language code for the text (default is 'zh' for Chinese, 'en' for English)
    :param is_gpu: whether to use GPU for processing (default is False)
    :return: list of tuples containing words and their corresponding POS tags
    """
    """
    The weights_only in PyTorch 2.6 and later versions defaults to True.
    However, StanfordNLP model files require weights_only=False.
    If you plan to use StanfordNLP, you MUST set weights_only=False and downgrade to PyTorch 2.5 or earlier versions.
    """
    # Download the necessary models if not already downloaded
    download(language)

    # Set up the processors based on the mode
    processors: str = "tokenize,pos"
    match mode:
        case "cut":
            processors: str = "tokenize"
        case "pos":
            processors: str = "tokenize,pos"
        case "full":
            processors: str = "tokenize,pos,lemma"
    # Initialize the StanfordNLP pipeline
    nlp = Pipeline(processors=processors, lang=language, use_gpu=is_gpu)
    # Process the content
    doc = nlp(content)

    result: list[tuple[str] | tuple[str, str] | tuple[str, str, str]] = []
    # Extract words and their POS tags
    for sentence in doc.sentences:
        for word in sentence.words:
            match processors:
                case "tokenize":
                    result.append((word.text,))
                case "tokenize,pos":
                    result.append((word.text, word.upos))
                case "tokenize,pos,lemma":
                    result.append((word.text, word.upos, word.lemma))

    df: DataFrame = DataFrame()
    match processors:
        case "tokenize":
            df: DataFrame = DataFrame(data=result, columns=["word"])
        case "tokenize,pos":
            df: DataFrame = DataFrame(data=result, columns=["word", "pos"])
        case "tokenize,pos,lemma":
            df: DataFrame = DataFrame(data=result, columns=["word", "pos", "lemma"])

    print(f"StanfordNLP Text Analysis Result:\n{df}")

    return result
