#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 21:10
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   main.py
# @Desc     :

from pprint import pprint
from random import randint
from re import sub
from torch import nn, optim

from utils.config import CONFIG
from utils.helper import read_file, Timer
from utils.model import CharsRNNTorchModel
from utils.nlp import unique_characters, extract_zh_chars
from utils.PT import (SequentialTorchDataset, TorchDataLoader,
                      TorchRandomSeed)
from utils.trainer import TorchTrainer


def preprocess_data() -> tuple[list[int], list[str], dict[str, int], int, int]:
    """ Data Preparation Function """
    # Get file content
    raw: str = read_file(CONFIG.FILEPATHS.POEMS)

    # Remove unwanted characters
    pattern: str = r"[^\u4e00-\u9fa5]"
    cleaned: str = sub(pattern, "", raw)
    # print(cleaned)

    # Get unique characters and add unknown and padding token
    unique_chars: list[str] = ["<PAD>", "<UNK>"] + unique_characters(cleaned)
    # print(unique_chars)

    # Transform list of unique characters to a dictionary
    dictionary: dict[str, int] = {char: index for index, char in enumerate(unique_chars)}
    # print(dictionary)
    token_pad: int = dictionary[unique_chars[0]]
    # print(token_pad)

    # Extract poem characters and lines
    poems, _ = extract_zh_chars(CONFIG.FILEPATHS.POEMS, pattern)

    # Build poems2id sequences
    token_unk: int = dictionary[unique_chars[1]]
    sequences: list = []
    for word in poems:
        word_index: int = dictionary.get(word, token_unk)
        sequences.append(word_index)
    # print(f"Total number of poem sequences: {len(sequences)}")
    # print(sequences[:50])

    return sequences, unique_chars, dictionary, token_pad, token_unk


def prepare_data():
    """ Data Preparation Wrapper Function """
    # Get preprocessed data
    sequences, _, dictionary, pad_token, _ = preprocess_data()
    # pprint({
    #     "sequences_sample": sequences[:5],
    #     "dictionary_sample": dict(list(dictionary.items())[:5]),
    #     "pad_token": pad_token
    # })

    # Setup Sequential PyTorch Dataset
    dataset: SequentialTorchDataset = SequentialTorchDataset(
        sequences=sequences,
        sequence_length=CONFIG.PARAMETERS.SEQUENTIAL_LENGTH,
        pad_token=pad_token,
    )
    # print(dataset)
    # index = randint(0, len(dataset) - 1)
    # print(dataset[index])

    # Setup DataLoader
    loader = TorchDataLoader(dataset, CONFIG.PREPROCESSOR.BATCH_SIZE)

    return loader, dictionary


def main() -> None:
    """ Main Function """
    with Timer("Get Data for Training"):
        loader, dictionary = prepare_data()
        print(loader)
        index = randint(0, len(loader) - 1)
        print(loader[index])

    with TorchRandomSeed("Sequential Data Training"):
        # Setup rnn model
        model = CharsRNNTorchModel(
            vocab_size=len(dictionary),
            embedding_dim=CONFIG.PARAMETERS.EMBEDDING_DIM,
            hidden_size=CONFIG.PARAMETERS.HIDDEN_SIZE,
            num_layers=CONFIG.PARAMETERS.RNN_LAYERS,
        )
        print(model)
        # Setup optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=CONFIG.HYPERPARAMETERS.ALPHA)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        trainer = TorchTrainer(model, optimizer, criterion, CONFIG.HYPERPARAMETERS.ACCELERATOR)
        trainer.fit(
            train_loader=loader,
            valid_loader=loader,
            epochs=CONFIG.HYPERPARAMETERS.EPOCHS,
            model_save_path=str(CONFIG.FILEPATHS.SAVED_MODEL),
        )


if __name__ == "__main__":
    main()
