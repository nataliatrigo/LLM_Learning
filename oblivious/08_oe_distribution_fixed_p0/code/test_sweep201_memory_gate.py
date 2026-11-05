"""Exercise weighted leases without allocating large arrays."""

import multiprocessing as mp
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from oblivious.oe_distribution_fixed_p0.code.sweep201_memory_gate import gated


def lease_worker(directory,states,capacity,token_bytes,events,hold=.06):
    with gated(states,directory=directory,capacity=capacity,token_bytes=token_bytes,
               overhead_bytes=0,poll_seconds=.005) as lease:
        events.put(('enter',time.monotonic_ns(),lease.units))
        time.sleep(hold)
        events.put(('exit',time.monotonic_ns(),lease.units))


def signal_worker(directory,ready):
    with gated(2,directory=directory,capacity=2,token_bytes=16,
               overhead_bytes=0,poll_seconds=.005):
        ready.set()
        while True:
            time.sleep(.1)


class MemoryGateTests(unittest.TestCase):
    def test_actual_201_251_state_sizes_reserve_one_and_two_units(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)/'gate'
            for grid,expected in ((201,1),(251,2)):
                states = grid*(grid+1)//2
                with gated(states,directory=directory) as lease:
                    self.assertEqual(lease.units,expected)
                    self.assertEqual(lease.capacity,6)
                    self.assertGreater(lease.reserved_bytes,8*states*states)

    def test_concurrent_weighted_leases_never_exceed_capacity_or_deadlock(self):
        context = mp.get_context('spawn')
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)/'gate'
            events = context.Queue()
            # n=2 requires 2 tokens, n=1 requires 1, each token is only 16 bytes.
            processes = [context.Process(target=lease_worker,args=(directory,n,6,16,events))
                         for n in (2,2,2,1,2,1,2,1)]
            for process in processes:
                process.start()
            records = [events.get(timeout=20) for _ in range(2*len(processes))]
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode,0)
            active = maximum = 0
            for event,_,units in sorted(records,key=lambda event:event[1]):
                active += units if event == 'enter' else -units
                maximum = max(active,maximum)
                self.assertGreaterEqual(active,0)
                self.assertLessEqual(active,6)
            self.assertEqual(active,0)
            self.assertGreaterEqual(maximum,2)

    def test_abrupt_termination_releases_all_tokens(self):
        context = mp.get_context('spawn')
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)/'gate'
            ready = context.Event()
            process = context.Process(target=signal_worker,args=(directory,ready))
            process.start()
            self.assertTrue(ready.wait(timeout=15))
            process.terminate()
            process.join(timeout=10)
            self.assertIsNotNone(process.exitcode)
            with gated(2,directory=directory,capacity=2,token_bytes=16,
                       overhead_bytes=0,poll_seconds=.005) as lease:
                self.assertEqual(lease.units,2)

    def test_exception_releases_tokens_and_impossible_request_fails(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)/'gate'
            with self.assertRaisesRegex(RuntimeError,'synthetic'):
                with gated(2,directory=directory,capacity=2,token_bytes=16,overhead_bytes=0):
                    raise RuntimeError('synthetic')
            with gated(2,directory=directory,capacity=2,token_bytes=16,overhead_bytes=0) as lease:
                self.assertEqual(lease.units,2)
            with self.assertRaisesRegex(ValueError,'budget'):
                with gated(3,directory=directory,capacity=2,token_bytes=16,overhead_bytes=0):
                    self.fail('Oversized request cannot acquire a lease')


if __name__ == '__main__':
    unittest.main()
