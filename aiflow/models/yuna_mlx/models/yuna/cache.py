from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)

        self.offset = self.keys.shape[2]
        return self.keys, self.values