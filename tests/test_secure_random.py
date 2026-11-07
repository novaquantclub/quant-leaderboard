"""Check deterministic domain separation and statistical transforms of private draws."""

import hmac
import json
import unittest

import numpy as np

from secure_random import SecureRandom


class SecureRandomTests(unittest.TestCase):
    def test_counter_stream_matches_hmac_and_is_independent_of_read_sizes(self):
        seed, stream = 123456, (2, 1, 3)
        label = (
            b"nova-quant-synthetic-v3\0"
            + json.dumps(stream, separators=(",", ":")).encode()
        )
        key = hmac.digest(seed.to_bytes(16, "big"), label, "sha256")
        expected = b"".join(
            hmac.digest(key, number.to_bytes(16, "big"), "sha256")
            for number in range(4)
        )
        rng = SecureRandom(seed, stream)
        self.assertEqual(rng._take(17) + rng._take(84), expected[:101])
        self.assertNotEqual(expected[:32], SecureRandom(seed, (2, 2, 3))._take(32))
        self.assertNotEqual(expected[:32], SecureRandom(seed + 1, stream)._take(32))

    def test_normal_prefixes_and_shapes_are_stable(self):
        short = SecureRandom(19, (0, 0)).standard_normal((21, 2))
        long = SecureRandom(19, (0, 0)).standard_normal((40, 2))
        np.testing.assert_array_equal(short, long[:21])
        self.assertEqual(
            SecureRandom(19, (0, 0)).normal([1, 2], [0, 0], (21, 2)).shape, (21, 2)
        )
        self.assertTrue(np.isfinite(long).all())

    def test_distribution_moments_and_integer_bounds(self):
        rng = SecureRandom(883, (0, 1))
        values = rng.standard_normal(100_000)
        self.assertLess(abs(values.mean()), 0.02)
        self.assertLess(abs(values.var() - 1), 0.03)
        self.assertTrue(all(0 <= rng.integers(5) < 5 for _ in range(1000)))
        self.assertIn(rng.choice(["A", "B"]), ["A", "B"])
        self.assertGreater(rng.chisquare(7), 0)
        for high in (0, -1, 2**64 + 1):
            with self.assertRaises(ValueError):
                rng.integers(high)
