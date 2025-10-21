#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/22 01:00
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_thu_05_embedding.py
# @Desc     :   

from torch import nn, Tensor, tensor

from utils.config import EXAMPLES_CHINESE_PATH, EXAMPLES_CHINESE_STOPWORDS_PATH
from utils.helper import read_file
from utils.nlp import regular_chinese
from utils.THU import cut_only


def data_praparation() -> dict[str, int]:
    """ Data Preparation Function """
    content = read_file(EXAMPLES_CHINESE_PATH)
    words = cut_only(content)

    stopwords: set[str] = set(read_file(EXAMPLES_CHINESE_STOPWORDS_PATH).splitlines())
    # print(stopwords)
    vocabs = [word for subwords in words for word in subwords if word not in stopwords]
    # print(vocabs)

    regular = regular_chinese(vocabs)
    # print(regular)

    # id2word mapping
    id2word = set(regular)
    # print(id2word)

    # word2id mapping
    word2id = {word: index for index, word in enumerate(id2word)}
    print(word2id)

    return word2id


def main() -> None:
    """ Main Function """
    layer = nn.Embedding(
        num_embeddings=len(data_praparation()),
        embedding_dim=3,
    )

    for word, index in data_praparation().items():
        vector: Tensor = layer(tensor(index))
        print(f"{index:<03} {word:<12} {vector.detach().numpy()}")


if __name__ == "__main__":
    main()
