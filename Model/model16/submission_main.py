"""Submission 16: pretrained, strictly streaming CPU inference."""
import os

# Set before importing numerical libraries, including in forked workers.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

INFER_PARALLELISM = 12


def train(datasets, model_directory_path):
    """Defensive validation only; there is no cloud training implementation."""
    from solution16 import validate_resources
    validate_resources(model_directory_path)


def infer(datasets, model_directory_path):
    from solution16 import infer as run_inference
    yield from run_inference(datasets, model_directory_path)
