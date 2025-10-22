#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 21:39
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   config.py
# @Desc     :   

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

# Set base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Configuration dictionary
CONFIGRATION: SimpleNamespace = SimpleNamespace(
    # File paths
    PATHS=SimpleNamespace(
        MODEL_SAVE=BASE_DIR / "models/model.pth",
        EXAMPLE_PAPER_ZH=BASE_DIR / "data/Chinese.txt",
        EXAMPLE_STOPWORDS_ZH=BASE_DIR / "data/stopwords4background.txt",
        DATASET_TRAIN=BASE_DIR / "data/",
        DATASET_SET=BASE_DIR / "data/",
    ),
    # Preprocessing settings
    PREPROCESSOR=SimpleNamespace(
        RANDOM_STATE=27,
        VALID_SIZE=0.2,
        IS_SHUFFLE=True,
        PCA_VARIANCE_THRESHOLD=0.95,
    ),
    # Training hyperparameters
    HYPERPARAMETERS=SimpleNamespace(
        BATCH_SIZE=32,
        HIDDEN_UNITS=128,
        ALPHA=0.01,
        ALPHA4REDUCTION=0.3,
        EPOCHS=20,
        ACCELERATOR="cpu",
    ),
)


@dataclass
class Paths:
    MODEL_SAVE: Path = BASE_DIR / "models/model.pth"
    EXAMPLE_PAPER_ZH: Path = BASE_DIR / "data/Chinese.txt"
    EXAMPLE_STOPWORDS_ZH: Path = BASE_DIR / "data/stopwords4background.txt"
    POEMS_JSON: Path = BASE_DIR / "data/poems.json"
    POEMS: Path = BASE_DIR / "data/poems.txt"
    DATASET_TRAIN: Path = BASE_DIR / "data/"
    DATASET_TEST: Path = BASE_DIR / "data/"


@dataclass
class Preprocessor:
    RANDOM_STATE: int = 27
    VALID_SIZE: float = 0.2
    IS_SHUFFLE: bool = True
    PCA_VARIANCE_THRESHOLD: float = 0.95


@dataclass
class Hyperparameters:
    BATCH_SIZE: int = 32
    HIDDEN_UNITS: int = 128
    ALPHA: float = 0.01
    ALPHA_REDUCTION: float = 0.3
    EPOCHS: int = 20
    ACCELERATOR: str = "cpu"


@dataclass
class Configration:
    FILEPATHS: Paths = field(default_factory=Paths)
    PREPROCESSOR: Preprocessor = field(default_factory=Preprocessor)
    HYPERPARAMETERS: Hyperparameters = field(default_factory=Hyperparameters)


# Singleton instance of Config
CONFIG = Configration()
