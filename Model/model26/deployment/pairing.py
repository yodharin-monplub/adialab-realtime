"""Nested float32 depth and synthetic-weight averaging within one fold."""
import numpy as np


class PairedXGBVoter:
    def __init__(self, first, second, depth25, depth60, columns):
        self.models = (first, second, depth25, depth60)
        self.columns = columns

    def predict(self, row):
        values = np.ascontiguousarray(row[self.columns], dtype=np.float32)[None, :]
        p = [np.float32(np.asarray(m.inplace_predict(values, validate_features=False)).reshape(-1)[0])
             for m in self.models]
        half = np.float32(.5)
        base = half*p[0] + half*p[1]
        deeper = half*p[2] + half*p[3]
        return float(half*base + half*deeper)
