"""Within-fold probability mixture; the caller retains the original logit bag."""
import numpy as np


class PairedXGBVoter:
    def __init__(self, first, second, columns):
        self.first, self.second, self.columns = first, second, columns

    def predict(self, row):
        values = np.ascontiguousarray(row[self.columns], dtype=np.float32)[None, :]
        first = np.float32(np.asarray(self.first.inplace_predict(values, validate_features=False)).reshape(-1)[0])
        second = np.float32(np.asarray(self.second.inplace_predict(values, validate_features=False)).reshape(-1)[0])
        # Do not clip individual voters before the arithmetic mixture. The
        # existing five-fold logit bag clips the mixed probabilities afterward.
        return float(np.float32(.5)*first+np.float32(.5)*second)
