"""Action token decoding helpers."""

from __future__ import annotations

import numpy as np


class ActionTokenizer:
    def __init__(self, vocab_size: int, bins: int = 256, min_action: float = -1.0, max_action: float = 1.0) -> None:
        self.vocab_size = int(vocab_size)
        self.n_bins = int(bins)
        self.min_action = float(min_action)
        self.max_action = float(max_action)

        self.bins = np.linspace(self.min_action, self.max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        discretized = self.vocab_size - action_token_ids
        discretized = np.clip(discretized - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized]
