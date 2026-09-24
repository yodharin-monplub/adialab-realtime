"""Submission29: streaming overlap diagnostic for the 100 public test series."""
from overlap_probe import load_reference, run_inference

# Keep one counter across the dataset instead of separate worker percentages.
INFER_PARALLELISM = 1


def train(datasets, model_directory_path):
    load_reference(model_directory_path)


def infer(datasets, model_directory_path):
    yield from run_inference(datasets, model_directory_path)
