"""Crunch entry points for Model 13."""
from models.model11.serving import train as _train, infer as _infer

INFER_PARALLELISM = 4


def train(datasets, model_directory_path):
    return _train(datasets, model_directory_path, 13)


def infer(datasets, model_directory_path):
    yield from _infer(datasets, model_directory_path, 13)
