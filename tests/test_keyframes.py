import unittest

from tdgnet.keyframes import default_keyframe_indices


class KeyframeTests(unittest.TestCase):
    def test_chronological_indices_match_diffusion_times(self):
        for steps in range(1, 129):
            with self.subTest(steps=steps):
                indices = default_keyframe_indices(steps)
                self.assertEqual(indices, sorted(set(indices)))
                self.assertEqual(indices[0], 0)
                self.assertEqual(indices[-1], steps)
                self.assertEqual({steps - i for i in indices}, {steps, steps // 2, steps // 4, 0})

    def test_rejects_invalid_step_counts(self):
        for count in [0, -1, True, 16.0]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                default_keyframe_indices(count)
