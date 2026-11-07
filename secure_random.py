"""Reproducible HMAC-backed random streams for private synthetic observations."""

import hmac
import json

import numpy as np

RANDOMNESS_VERSION = "hmac-sha256-counter-v1"


class SecureRandom:
    """Expand a secret seed into independently scoped streams using HMAC-SHA256."""

    def __init__(self, seed, stream):
        label = (
            b"nova-quant-synthetic-v3\0"
            + json.dumps(stream, separators=(",", ":")).encode()
        )
        self.key = hmac.digest(int(seed).to_bytes(16, "big"), label, "sha256")
        self.counter = 0
        self.buffer = bytearray()

    def _take(self, count):
        while len(self.buffer) < count:
            self.buffer.extend(
                hmac.digest(self.key, self.counter.to_bytes(16, "big"), "sha256")
            )
            self.counter += 1
        result = bytes(self.buffer[:count])
        del self.buffer[:count]
        return result

    def _uniform(self, count):
        values = np.frombuffer(self._take(count * 8), dtype=">u8")
        return (values >> 11).astype(float) * (1.0 / 2**53)

    def random(self):
        return float(self._uniform(1)[0])

    def integers(self, high):
        if high <= 0 or high > 2**64:
            raise ValueError(
                "The integer range must contain between 1 and 2**64 values."
            )
        limit = 2**64 - (2**64 % high)
        while True:
            value = int.from_bytes(self._take(8), "big")
            if value < limit:
                return value % high

    def choice(self, values):
        return values[self.integers(len(values))]

    def standard_normal(self, size):
        shape = (int(size),) if np.isscalar(size) else tuple(size)
        count = int(np.prod(shape))
        uniform = self._uniform(2 * ((count + 1) // 2)).reshape(-1, 2)
        radius = np.sqrt(-2 * np.log1p(-uniform[:, 0]))
        angle = 2 * np.pi * uniform[:, 1]
        values = np.column_stack(
            (radius * np.cos(angle), radius * np.sin(angle))
        ).ravel()
        return values[:count].reshape(shape)

    def normal(self, mean, scale, size):
        return np.asarray(mean) + np.asarray(scale) * self.standard_normal(size)

    def chisquare(self, degrees):
        values = self.standard_normal(degrees)
        return float(values @ values)
