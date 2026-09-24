"""Inference-only Submission32 entry point (Submission31 + Model147 student).

The models in resources/ were trained offline.  The training code is shipped in training_code/
(see training_code/README.txt for the order in which it is run); train() below keeps the platform
contract and only validates the bundled files, because training takes several days of CPU and
needs caches that do not exist in the cloud runner.
"""
import os

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

INFER_PARALLELISM = 12


def train(datasets, model_directory_path):
    from solution147 import validate_resources
    validate_resources(model_directory_path)


def infer(datasets, model_directory_path):
    from solution147 import infer as run_inference
    yield from run_inference(datasets, model_directory_path)
