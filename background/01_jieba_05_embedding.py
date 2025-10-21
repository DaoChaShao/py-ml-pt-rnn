#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 23:15
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   01_jieba_05_embedding.py
# @Desc     :   

from torch import nn, tensor

from utils.config import EXAMPLES_CHINESE_PATH, EXAMPLES_CHINESE_STOPWORDS_PATH
from utils.helper import read_file
from utils.JB import cut_accuracy
from utils.nlp import regular_chinese


def data_preparation() -> dict[str, int]:
    """ Data Preparation """
    content: str = read_file(EXAMPLES_CHINESE_PATH)

    # Read stopwords from file without repeats
    stopwords: set[str] = set(read_file(EXAMPLES_CHINESE_STOPWORDS_PATH).splitlines())

    words = cut_accuracy(content)
    # Remove stopwords from the cut words
    words_stop: list[str] = [word for word in words if word not in stopwords]

    # Use regex to filter only Chinese characters
    vocabs = regular_chinese(words_stop)

    # Avoid repeats by converting to set and back to list, which is call 'id2word' technique
    id2word = list(set(vocabs))

    # word to id mapping
    word2id = {word: index for index, word in enumerate(id2word)}
    print(word2id)

    return word2id


def embed_layer(dictionary: dict):
    """ Embed the words from a dictionary
    :param dictionary: the dictionary to embed
    :return : None
    """
    # Setup embedding layer
    layer = nn.Embedding(
        num_embeddings=len(dictionary),
        embedding_dim=3,
    )

    # Forward pass fakely through the embedding layer
    for word, index in dictionary.items():
        # Initialise the weights for each word index, default is random normal distribution
        vector = layer(tensor(index))
        print(f"{index:<03} {word:<12} {vector.detach().numpy()}")


def main() -> None:
    """ Main Function """
    word2id = data_preparation()
    print(f"Total unique words (vocabulary size): {len(word2id)}")
    embed_layer(word2id)


if __name__ == "__main__":
    main()
