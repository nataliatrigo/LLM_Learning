"""The equilibrium correspondence must preserve uncovered and isolated starts."""
import unittest

from oblivious.fluid_oblivious_initial_conditions.code.run_experiment import correspondence


class CorrespondenceTests(unittest.TestCase):
    def equilibrium(self, index, lo, hi, left=True, right=True):
        return dict(equilibrium_id=index, verified=True,
                    initial_conditions=[dict(lo=lo, hi=hi, left_closed=left, right_closed=right)])

    def test_distinct_policies_can_leave_a_gap_in_equilibrium_coverage(self):
        parts = correspondence([self.equilibrium(0, 0., .4),
                                self.equilibrium(1, .6, 1., left=False)])
        self.assertEqual([row["equilibrium_ids"] for row in parts], [[0], [], [1]])
        gap = parts[1]
        self.assertEqual((gap["lo"], gap["hi"]), (.4, .6))
        self.assertFalse(gap["left_closed"])
        self.assertTrue(gap["right_closed"])

    def test_isolated_equilibrium_start_is_not_lost_between_open_cells(self):
        parts = correspondence([self.equilibrium(0, .5, .5)])
        self.assertEqual([row["equilibrium_count"] for row in parts], [0, 1, 0])
        self.assertEqual((parts[1]["lo"], parts[1]["hi"]), (.5, .5))

    def test_two_equilibria_from_the_same_start_are_reported(self):
        parts = correspondence([self.equilibrium(0, 0., .7),
                                self.equilibrium(1, .3, 1.)])
        self.assertEqual([row["equilibrium_ids"] for row in parts], [[0], [0, 1], [1]])
        self.assertEqual((parts[1]["lo"], parts[1]["hi"]), (.3, .7))


if __name__ == "__main__":
    unittest.main()
