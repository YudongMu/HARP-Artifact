import unittest

from hardware.harp_components import attention_pairs, issue_split, phase_counts, qk_issue_split


class HarpComponentsTest(unittest.TestCase):
    def test_refresh_keeps_full_and_replay_distinct(self):
        self.assertEqual(attention_pairs(1000, 0.3, "refresh"), (1000, 300))
        self.assertEqual(attention_pairs(1000, 0.3, "steady"), (0, 300))
        self.assertEqual(attention_pairs(1000, 0.3, "plain"), (1000, 0))

    def test_fallback_and_plane_pairing_use_only_retained_work(self):
        self.assertEqual(issue_split(1000, 0.0, 0.5), (0, 500))
        self.assertEqual(qk_issue_split(10, 2, 72, 0.0, 4, 32), (0, 960))
        self.assertEqual(qk_issue_split(10, 2, 72, 1.0, 4, 32), (1440, 0))

    def test_cfg_calls_share_refresh_schedule_by_step(self):
        graph = {"steps": 11, "forwards_per_step": 2, "forward_calls": 22}
        self.assertEqual(phase_counts(graph, 5, "combined"), [("refresh", 6), ("steady", 16)])
        self.assertEqual(phase_counts(graph, 5, "planeclear"), [("plain", 22)])

