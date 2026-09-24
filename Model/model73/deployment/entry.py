"""Inference-only Model73 cloud entry point."""
import os

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

INFER_PARALLELISM = 12


def train(datasets, model_directory_path):
    from solution73 import validate_resources
    validate_resources(model_directory_path)


def infer(datasets, model_directory_path):
    from solution73 import infer as run_inference
    yield from run_inference(datasets, model_directory_path)
