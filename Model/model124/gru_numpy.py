"""Deterministic float32 numpy inference for the Model124 GRU student.

Mirrors torch.nn.GRU (gate order r, z, n; bias_hh applied inside the n gate)
followed by the head Linear -> GELU -> Linear.  One call per online step with
an explicit hidden state, so serving is O(1) per step and needs no torch.
"""
from __future__ import annotations

import math

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gelu(x):
    # torch.nn.GELU default (exact erf form)
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


class NumpyGRU:
    def __init__(self, weights: dict, layers: int):
        self.layers = layers
        self.w = {}
        for l in range(layers):
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                self.w[f"{name}_l{l}"] = np.asarray(weights[f"gru.{name}_l{l}"], dtype=np.float32)
        self.h0_w = np.asarray(weights["head.0.weight"], dtype=np.float32)
        self.h0_b = np.asarray(weights["head.0.bias"], dtype=np.float32)
        self.h2_w = np.asarray(weights["head.2.weight"], dtype=np.float32).reshape(-1)
        self.h2_b = float(np.asarray(weights["head.2.bias"], dtype=np.float32).reshape(-1)[0])
        self.hidden = self.w["weight_hh_l0"].shape[1]

    def initial_state(self):
        return [np.zeros(self.hidden, dtype=np.float32) for _ in range(self.layers)]

    def step(self, x: np.ndarray, state: list) -> float:
        """Advance one time step; returns the logit. state is updated in place."""
        inp = np.asarray(x, dtype=np.float32)
        for l in range(self.layers):
            h = state[l]
            gi = self.w[f"weight_ih_l{l}"] @ inp + self.w[f"bias_ih_l{l}"]
            gh = self.w[f"weight_hh_l{l}"] @ h + self.w[f"bias_hh_l{l}"]
            H = self.hidden
            r = _sigmoid(gi[:H] + gh[:H])
            z = _sigmoid(gi[H:2 * H] + gh[H:2 * H])
            n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
            h = (1.0 - z) * n + z * h
            state[l] = h.astype(np.float32)
            inp = state[l]
        a = _gelu(self.h0_w @ inp + self.h0_b)
        return float(self.h2_w @ a + self.h2_b)

    def run(self, X: np.ndarray) -> np.ndarray:
        state = self.initial_state()
        return np.asarray([self.step(row, state) for row in X], dtype=np.float32)
