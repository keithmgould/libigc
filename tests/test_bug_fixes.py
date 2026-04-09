"""Tests that expose known bugs. Written before fixes — all should FAIL initially."""

import math
import unittest
from libigc.lib.viterbi import SimpleViterbiDecoder
from libigc.utils import _rawtime_float_to_hms
from libigc.glide import Glide
from libigc.gnss_fix import GNSSFix


class TestViterbiReuse(unittest.TestCase):
    """Calling decode() should not corrupt the decoder's initial probabilities.

    Bug: _init_log is aliased (not copied) into state_log[0], then mutated.
    A second decode() starts with corrupted initial probabilities.
    """

    def test_decode_does_not_corrupt_init_probs(self):
        decoder = SimpleViterbiDecoder(
            init_probs=[0.5, 0.5],
            transition_probs=[[0.9, 0.1], [0.1, 0.9]],
            emission_probs=[[0.9, 0.1], [0.1, 0.9]],
        )
        init_before = list(decoder._init_log)
        decoder.decode([0, 0, 1, 1])
        init_after = list(decoder._init_log)
        self.assertEqual(init_before, init_after)


class TestRawtimeToHms(unittest.TestCase):
    """_rawtime_float_to_hms should return integer hours, minutes, seconds.

    Bug: Python 3 true division (/) returns floats instead of ints.
    """

    def test_returns_integers(self):
        hms = _rawtime_float_to_hms(3661.0)  # 1h 1m 1s
        self.assertIsInstance(hms.hours, int)
        self.assertIsInstance(hms.minutes, int)
        self.assertIsInstance(hms.seconds, int)

    def test_correct_values(self):
        hms = _rawtime_float_to_hms(45296.0)  # 12h 34m 56s
        self.assertEqual(hms.hours, 12)
        self.assertEqual(hms.minutes, 34)
        self.assertEqual(hms.seconds, 56)



class TestGlideZeroDuration(unittest.TestCase):
    """Glide.speed() should not crash when enter_fix == exit_fix (zero duration).

    Bug: no division-by-zero guard in speed().
    """

    def test_speed_zero_duration_does_not_crash(self):
        fix = GNSSFix(
            rawtime=1000.0, lat=45.0, lon=10.0, validity="A",
            press_alt=500.0, gnss_alt=500.0, index=0, extras="",
        )
        fix.timestamp = 1000.0
        glide = Glide(fix, fix, 0.0)
        # Should not raise ZeroDivisionError
        speed = glide.speed()
        self.assertEqual(speed, 0.0)


if __name__ == "__main__":
    unittest.main()
