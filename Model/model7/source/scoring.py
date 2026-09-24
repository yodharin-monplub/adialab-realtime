from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from competition_data import TrainingData

# Crunch runs a submitted directory as Python's import root. Reproduce that
# layout locally before importing ``main``; importing ``submission_2.main`` as
# a package would break its intentional top-level imports.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
SUBMISSION_DIRECTORY = SCRIPT_DIRECTORY / "submissions" / "submission_2"
if not SUBMISSION_DIRECTORY.is_dir():
  raise FileNotFoundError(f"Submission directory not found: {SUBMISSION_DIRECTORY}")
sys.path.insert(0, str(SUBMISSION_DIRECTORY))

from main import infer, train


TRAIN_DIRECTORY = SCRIPT_DIRECTORY / "train"
TEST_DIRECTORY = SCRIPT_DIRECTORY / "test"
MODEL_DIRECTORY = SUBMISSION_DIRECTORY / "resources"


def series_count(data: TrainingData) -> int:
  """Return the number of time series in a dataset."""
  try:
      return len(data)
  except TypeError:
      for name in ("series_count", "n_series"):
          if hasattr(data, name):
              value = getattr(data, name)
              return int(value() if callable(value) else value)

  raise TypeError("Could not determine the number of series")


def target_for(record) -> np.ndarray:
  """Create per-step labels from the known test break location."""
  length = len(record.online)
  tau = record.tau_index

  if tau is None or int(tau) < 0:
      return np.zeros(length, dtype=np.uint8)

  return (np.arange(length) >= int(tau)).astype(np.uint8)


def time_stratified_auc(
  targets: np.ndarray,
  predictions: np.ndarray,
  time_steps: np.ndarray,
) -> float:
  """Calculate the competition's pair-weighted TSAUC."""
  weighted_auc = 0.0
  total_weight = 0

  for step in np.unique(time_steps):
      mask = time_steps == step
      step_targets = targets[mask]
      step_predictions = predictions[mask]

      positives = int(step_targets.sum())
      negatives = int(step_targets.size - positives)

      if positives == 0 or negatives == 0:
          continue

      weight = positives * negatives

      weighted_auc += weight * roc_auc_score(
          step_targets,
          step_predictions,
      )
      total_weight += weight

  if total_weight == 0:
      raise ValueError(
          "No test time step contained both positive and negative labels"
      )

  return weighted_auc / total_weight


def main() -> None:
  # Read the existing train and test directories directly.
  train_data = TrainingData(TRAIN_DIRECTORY)
  test_data = TrainingData(TEST_DIRECTORY, split="test")

  train_records_raw = [
      train_data.record(i)
      for i in range(series_count(train_data))
  ]
  test_records = [
      test_data.record(i)
      for i in range(series_count(test_data))
  ]

  # Convert training records to the tuple format expected by train().
  train_records = [
      (
          record.dataset_id,
          record.historical,
          record.online,
          record.tau_index,
      )
      for record in train_records_raw
  ]

  # Construct test labels and online time indices before inference.
  test_targets = np.concatenate(
      [target_for(record) for record in test_records]
  )
  test_time_steps = np.concatenate(
      [
          np.arange(len(record.online), dtype=np.int32)
          for record in test_records
      ]
  )

  # Submission 2 contains no pretrained weights. Reproduce Crunch by fitting
  # both components from the complete labeled train iterator and writing the
  # resulting files only beneath submission_2/resources.
  print(
      f"Training submission_2 with "
      f"{len(train_records):,} available training series..."
  )
  train(train_records, str(MODEL_DIRECTORY))

  test_stream = (
      (record.historical, iter(record.online))
      for record in test_records
  )

  prediction_generator = infer(
      test_stream,
      str(MODEL_DIRECTORY),
  )

  # The first yield is Crunch's readiness signal.
  readiness = next(prediction_generator)
  if readiness is not None:
      raise RuntimeError(
          "infer() did not produce the readiness sentinel"
      )

  predictions = np.fromiter(
      prediction_generator,
      dtype=np.float64,
  )

  if predictions.size != test_targets.size:
      raise RuntimeError(
          f"Expected {test_targets.size:,} predictions, "
          f"received {predictions.size:,}"
      )

  if not np.all(np.isfinite(predictions)):
      raise RuntimeError("Inference produced non-finite predictions")

  score = time_stratified_auc(
      test_targets,
      predictions,
      test_time_steps,
  )

  print(f"Training series: {len(train_records):,}")
  print(f"Test series:     {len(test_records):,}")
  print(f"Test decisions:  {test_targets.size:,}")
  print(f"Submission-2 TSAUC: {score:.6f}")


if __name__ == "__main__":
  main()
