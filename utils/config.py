#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 21:39
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   config.py
# @Desc     :   

from pathlib import Path

# Set base directory
BASE_DIRECTORY = Path(__file__).resolve().parent.parent
# Model save path
MODEL_SAVE_PATH = BASE_DIRECTORY / "models/model.pth"
# Example, train and test dataset path
EXAMPLES_CHINESE_PATH = BASE_DIRECTORY / "data/Chinese.txt"
TRAIN_DATASET_PATH = BASE_DIRECTORY / "data/"
TEST_DATASET_PATH = BASE_DIRECTORY / "data/"

# Data processing parameters
RANDOM_STATE: int = 27
VALID_SIZE: float = 0.2
IS_SHUFFLE: bool = True

# PCA parameters
PCA_VARIANCE_THRESHOLD: float = 0.95

# Dataset & Dataloader settings
BATCHES: int = 32

# Training hyperparameters
HIDDEN_UNITS: int = 128
ALPHA: float = 0.01
ALPHA4REDUCTION: float = 0.3
EPOCHS: int = 20
ACCELERATOR: str = "cpu"
