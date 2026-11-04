#!/usr/bin/env python3
"""Mocked shared-GPU scheduling tests; no CUDA or nvidia-smi is required."""

from __future__ import annotations

import unittest

from server_run.rswm_gpu_selector import (
    GPUStat,
    NoGPUAvailable,
    check_gpu_now,
    select_idle_gpu,
)


def gpu(index, *, util, used):
    return GPUStat(
        index=str(index),
        name="NVIDIA A800-SXM4-80GB",
        total_mb=81920.0,
        used_mb=float(used),
        free_mb=81920.0 - float(used),
        utilization=float(util),
    )


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += float(seconds)


class SequenceProbe:
    def __init__(self, states):
        self.states = list(states)
        self.calls = 0

    def __call__(self):
        state = self.states[min(self.calls, len(self.states) - 1)]
        self.calls += 1
        return list(state)


class GPUSchedulerTests(unittest.TestCase):
    def run_selection(
            self, states, *, visible=None, confirm=2, max_scans=None,
            processes=None):
        clock = FakeClock()
        probe = SequenceProbe(states)
        process_probe = processes or (lambda _gpu: set())
        events = []
        result = select_idle_gpu(
            probe,
            process_probe,
            cuda_visible_devices=visible,
            confirm_seconds=confirm,
            poll_seconds=1,
            max_util=10,
            max_used_mb=5000,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
            on_event=lambda name, payload: events.append((name, payload)),
            max_scans=max_scans,
        )
        return result, events, probe

    def test_all_four_busy_never_selects_from_one_scan(self):
        busy = [[gpu(i, util=70 + i, used=20000 + i) for i in range(4)]]
        with self.assertRaises(NoGPUAvailable):
            self.run_selection(busy, confirm=2, max_scans=3)

    def test_idle_candidate_that_becomes_busy_is_cancelled(self):
        states = [
            [gpu(0, util=80, used=20000), gpu(1, util=0, used=100)],
            [gpu(0, util=80, used=20000), gpu(1, util=25, used=7000)],
            [gpu(0, util=80, used=20000), gpu(1, util=0, used=100)],
            [gpu(0, util=80, used=20000), gpu(1, util=0, used=100)],
            [gpu(0, util=80, used=20000), gpu(1, util=0, used=100)],
        ]
        result, events, _probe = self.run_selection(states)
        self.assertEqual(result.stat.index, "1")
        self.assertTrue(any(name == "reset" for name, _ in events))
        candidates = [payload["gpu"] for name, payload in events
                      if name == "candidate"]
        self.assertEqual(candidates, ["1", "1"])

    def test_sustained_idle_requires_repeated_observations(self):
        steady = [[gpu(0, util=3, used=1200), gpu(1, util=50, used=10000)]]
        result, events, probe = self.run_selection(steady, confirm=3)
        self.assertEqual(result.stat.index, "0")
        self.assertEqual(result.observations, 4)
        self.assertEqual(probe.calls, 4)
        self.assertEqual(
            len([name for name, _ in events if name == "progress"]), 3)

    def test_preexisting_visibility_restriction_is_never_widened(self):
        state = [[
            gpu(0, util=0, used=0),
            gpu(1, util=90, used=30000),
            gpu(2, util=2, used=1000),
            gpu(3, util=0, used=10),
        ]]
        result, _events, _probe = self.run_selection(
            state, visible="2", confirm=1)
        self.assertEqual(result.stat.index, "2")

    def test_multiple_candidates_choose_lowest_memory_then_utilization(self):
        state = [[
            gpu(0, util=1, used=1200),
            gpu(1, util=5, used=500),
            gpu(2, util=2, used=500),
            gpu(3, util=40, used=400),
        ]]
        result, _events, _probe = self.run_selection(state, confirm=1)
        self.assertEqual(result.stat.index, "2")

    def test_new_compute_process_resets_candidate(self):
        state = [[gpu(0, util=0, used=200)]]
        calls = {"n": 0}

        def processes(_gpu):
            calls["n"] += 1
            if calls["n"] == 2:
                return {4321}
            return set()

        result, events, _probe = self.run_selection(
            state, confirm=1, processes=processes)
        self.assertEqual(result.stat.index, "0")
        self.assertTrue(any(
            name == "reset" and "new compute" in str(payload["reason"])
            for name, payload in events
        ))

    def test_immediate_check_rejects_new_compute_process(self):
        stat = gpu(3, util=1, used=900)
        ok, reason, returned, pids = check_gpu_now(
            lambda: [stat],
            lambda _gpu: {111, 222},
            "3",
            baseline_pids={111},
            max_util=10,
            max_used_mb=5000,
        )
        self.assertFalse(ok)
        self.assertIn("222", reason)
        self.assertEqual(returned, stat)
        self.assertEqual(pids, (111, 222))


if __name__ == "__main__":
    unittest.main()
