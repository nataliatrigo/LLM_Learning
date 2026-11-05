"""Equal late windows and explicit descriptive classification."""
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd

from forgetting.Code.experiments import run_experiments as experiments
from forgetting.Code.experiments.interaction_grid import summarize, classify, worker, run


class InteractionGridTests(unittest.TestCase):
    def setUp(self):
        self.config=experiments.read_config(experiments.CONFIG_DIR/'config_interaction_grid.json')
        self.thresholds=self.config['classification']

    def test_equal_post_burn_in_windows(self):
        choices=np.column_stack((np.arange(12)<7,np.arange(12)>=7))
        frame,stats=summarize(choices,2,self.thresholds)
        self.assertEqual((stats['first_start'],stats['first_end'],stats['second_start'],stats['second_end']),
                         (3,7,8,12))
        np.testing.assert_allclose(frame.first_window_frequency,[1,0])
        np.testing.assert_allclose(frame.second_window_frequency,[0,1])
        np.testing.assert_allclose(frame.frequency,[.5,.5])
        np.testing.assert_array_equal(frame.product2_count_after_burn_in,[5,5])
        self.assertEqual(stats['classification'],'Unstable or heterogeneous')

    def test_classification_thresholds(self):
        base={'mean':0,'p05':0,'std':0,'mean_absolute_difference':0,'approximately_zero_share':1}
        self.assertEqual(classify(base,self.thresholds),'Extinction-like')
        stable={**base,'mean':.5,'p05':.48,'std':.005,'mean_absolute_difference':.005,'approximately_zero_share':0}
        self.assertEqual(classify(stable,self.thresholds),'Stable positive frequency')
        self.assertEqual(classify({**stable,'std':.02},self.thresholds),'Unstable or heterogeneous')
        self.assertEqual(classify({**stable,'mean_absolute_difference':.02},self.thresholds),'Unstable or heterogeneous')
        self.assertEqual(classify({**stable,'p05':.001},self.thresholds),'Unstable or heterogeneous')
        self.assertEqual(classify({**base,'mean':.02,'approximately_zero_share':.96},self.thresholds),'Unstable or heterogeneous')

    def test_reproducible_worker_and_outputs(self):
        c=self.config
        c['experiment'].update(paths=3,interactions=20,burn_in=4)
        c['solver'].update(nodes=101,terminal_gap=.1)
        c['p0_values']=[.5]
        c['rho_values']=[.4,.8]
        a=worker(c,.5,.4)
        b=worker(c,.5,.4)
        pd.testing.assert_frame_equal(a[0],b[0])
        self.assertEqual(a[1],b[1])
        mature=pd.DataFrame({'p0':[.5,.5],'rho':[.4,.8],'reachable_maximum_advantage':[.01,-.01]})
        with TemporaryDirectory() as temporary, patch('forgetting.Code.experiments.interaction_grid.mature_advantages',return_value=mature):
            output=Path(temporary)
            run(c,output,workers=1)
            summary=pd.read_csv(output/'data/parameter_summary.csv')
            paths=pd.read_csv(output/'data/trajectory_summary.csv')
            self.assertEqual(len(paths),6)
            self.assertEqual(len(summary),2)
            self.assertTrue((summary.first_end==12).all())
            self.assertTrue((summary.second_start==13).all())
            self.assertEqual(len(list((output/'figures').glob('*.png'))),5)
            self.assertFalse(list(output.rglob('*.pdf')))
