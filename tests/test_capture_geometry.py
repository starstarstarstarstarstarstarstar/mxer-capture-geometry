"""Behavioural tests for mxer_sim.capture_geometry (standard library only)."""

import math
import unittest

from mxer_sim import capture_geometry as cg

OUTCOMES = {"MISS", "GUIDED", "THROAT_ENTRY", "LATCH_CAPTURED"}


class ConeRadiusTest(unittest.TestCase):
    def test_radius_at_throat_is_throat_radius(self):
        self.assertEqual(cg.cone_radius_at_z(0.0, 1.0, math.radians(30.0)), 1.0)

    def test_radius_grows_linearly_with_tan_of_half_angle(self):
        r = cg.cone_radius_at_z(5.0, 1.0, math.radians(30.0))
        self.assertAlmostEqual(r, 1.0 + 5.0 / math.sqrt(3.0), places=12)

    def test_zero_half_angle_is_a_cylinder(self):
        self.assertEqual(cg.cone_radius_at_z(3.0, 2.0, 0.0), 2.0)


class ThresholdChecksTest(unittest.TestCase):
    def test_entry_miss_is_strictly_outside(self):
        self.assertTrue(cg.check_entry_miss(4.0001, 4.0))
        self.assertFalse(cg.check_entry_miss(4.0, 4.0))
        self.assertFalse(cg.check_entry_miss(1.0, 4.0))

    def test_throat_entry_includes_the_boundary(self):
        self.assertTrue(cg.check_throat_entry(1.0, 1.0))
        self.assertTrue(cg.check_throat_entry(0.5, 1.0))
        self.assertFalse(cg.check_throat_entry(1.0001, 1.0))

    def test_latch_capture_includes_the_boundary(self):
        self.assertTrue(cg.check_latch_capture(0.2, 0.2))
        self.assertTrue(cg.check_latch_capture(0.0, 0.2))
        self.assertFalse(cg.check_latch_capture(0.2001, 0.2))


class SimulateTrajectoryTest(unittest.TestCase):
    def sim(self, r0, z0=0.0, vr0=0.0, vz0=1.0, **kw):
        return cg.simulate_payload_trajectory(
            r0, z0, vr0, vz0, 200.0, 1000.0, 50.0, 1.0, math.radians(30.0), 5.0, **kw
        )

    def test_starting_inside_the_throat_stops_immediately(self):
        res = self.sim(0.5)
        self.assertEqual(res["status"], "THROAT_ENTRY")
        self.assertEqual(res["t"], 0.0)
        self.assertEqual(len(res["trajectory"]), 1)

    def test_starting_past_the_funnel_end_exits_immediately(self):
        res = self.sim(2.0, z0=6.0)
        self.assertEqual(res["status"], "EXIT")
        self.assertEqual(len(res["trajectory"]), 1)

    def test_off_axis_payload_is_pulled_into_the_throat(self):
        res = self.sim(2.0)
        self.assertEqual(res["status"], "THROAT_ENTRY")
        self.assertLessEqual(res["r"], 1.0)
        self.assertGreater(res["t"], 0.0)
        self.assertLess(res["z"], 5.0)
        # the recorded trajectory ends at the returned state
        last = res["trajectory"][-1]
        self.assertEqual((last["t"], last["r"], last["z"]), (res["t"], res["r"], res["z"]))

    def test_time_runs_out_before_any_stop_condition(self):
        res = self.sim(2.0, dt=0.001, max_time=0.0005)
        self.assertEqual(res["status"], "TIMEOUT")
        self.assertEqual(len(res["trajectory"]), 1)
        self.assertAlmostEqual(res["t"], 0.001, places=15)

    def test_v_tan_is_the_absolute_initial_radial_velocity(self):
        res = self.sim(2.0, vr0=-0.3)
        self.assertEqual(res["v_tan"], 0.3)


class EvaluateFunnelCaptureTest(unittest.TestCase):
    def test_miss_outside_the_entrance(self):
        res = cg.evaluate_funnel_capture(5.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self.assertEqual(res["outcome"], "MISS")
        self.assertEqual(res["status"], "MISS_AT_ENTRY")
        self.assertEqual(res["entry_radius_m"], 5.0)
        self.assertIsNone(res["throat_entry_time_s"])
        self.assertIsNone(res["residual_offset_at_throat_m"])
        self.assertIsNone(res["funnel_exit_velocity_m_s"])
        self.assertNotIn("trajectory", res)

    def test_entry_exactly_on_the_entrance_rim_is_not_a_miss(self):
        res = cg.evaluate_funnel_capture(2.4, 3.2, 0.0, 0.0, 0.0, 1.0)
        self.assertEqual(res["entry_radius_m"], 4.0)
        self.assertNotEqual(res["outcome"], "MISS")

    def test_throat_entry_outside_latch_tolerance(self):
        res = cg.evaluate_funnel_capture(2.0, 1.5, 0.0, 0.5, 0.5, 2.0)
        self.assertEqual(res["outcome"], "THROAT_ENTRY")
        self.assertEqual(res["status"], "THROAT_ENTRY")
        self.assertEqual(res["entry_radius_m"], 2.5)
        self.assertGreater(res["throat_entry_time_s"], 0.0)
        self.assertLessEqual(res["residual_offset_at_throat_m"], 1.0)
        self.assertGreater(res["residual_offset_at_throat_m"], 0.2)
        self.assertEqual(res["exit_velocity"], res["funnel_exit_velocity_m_s"])

    def test_latch_captured_when_entering_within_tolerance(self):
        res = cg.evaluate_funnel_capture(0.1, 0.1, 0.0, 0.05, 0.05, 0.5)
        self.assertEqual(res["outcome"], "LATCH_CAPTURED")
        self.assertEqual(res["throat_entry_time_s"], 0.0)
        self.assertAlmostEqual(res["residual_offset_at_throat_m"], math.sqrt(0.02), places=15)

    def test_guided_exit(self):
        res = cg.evaluate_funnel_capture(1.5, 0.0, 0.0, 0.5, 0.0, 3.0)
        self.assertEqual(res["outcome"], "GUIDED")
        self.assertEqual(res["status"], "EXIT")
        self.assertIsNone(res["throat_entry_time_s"])
        self.assertGreater(res["funnel_exit_velocity_m_s"], 0.0)

    def test_guided_timeout(self):
        res = cg.evaluate_funnel_capture(1.5, 0.0, 0.0, 0.5, 0.0, 0.0)
        self.assertEqual(res["outcome"], "GUIDED")
        self.assertEqual(res["status"], "TIMEOUT")
        self.assertIsNone(res["funnel_exit_velocity_m_s"])

    def test_outcome_depends_only_on_lateral_distance(self):
        a = cg.evaluate_funnel_capture(4.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        b = cg.evaluate_funnel_capture(0.0, 4.0, 0.0, 0.0, 0.0, 1.0)
        for key in ("outcome", "throat_entry_time_s", "residual_offset_at_throat_m"):
            self.assertEqual(a[key], b[key])

    def test_outcome_is_always_one_of_the_four(self):
        for dx in (0.0, 0.1, 1.5, 3.0, 4.0, 6.0):
            for vz in (0.0, 1.0, 3.0):
                res = cg.evaluate_funnel_capture(dx, 0.0, 0.0, 0.2, 0.0, vz)
                self.assertIn(res["outcome"], OUTCOMES)


if __name__ == "__main__":
    unittest.main()
