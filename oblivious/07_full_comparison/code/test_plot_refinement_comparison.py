"""Provenance checks for replacing a failed case with an isolated retry."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SPEC=importlib.util.spec_from_file_location(
    "plot_refinement_retry_test",Path(__file__).with_name("plot_refinement_comparison.py"))
PLOT=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)
RUN=PLOT.RUN


class RetryProvenanceTests(unittest.TestCase):
    def prepare(self,root):
        campaign=root/"original"
        campaign.mkdir()
        original=dict(fingerprint="original",configuration=dict(
            parameters=dict(R=1.,c1=.05,c2=.65,p1=.35,p2=.8,gamma=.98),
            order=128,tol=1e-7,dp_tol=1e-10))
        RUN.atomic_json(campaign/"manifest.json",original)
        retry=root/"retry"
        retry.mkdir()
        settings=dict(model="oe",rho=.99,n=101,
            parameters=original["configuration"]["parameters"],order=128,
            tol=1e-7,dp_tol=1e-10,source_campaign=str(campaign))
        sources={}
        settings["input_sha256"]={"original/manifest.json":RUN.digest(campaign/"manifest.json")}
        manifest=dict(configuration=settings,source_sha256=sources,
            input_sha256=settings["input_sha256"],
            source_campaign_fingerprint="original",fingerprint=RUN.fingerprint(settings,sources))
        RUN.atomic_json(retry/"manifest.json",manifest)
        candidate=dict(verified=True,candidate_verified=True,policy=np.zeros(101),
            dp_diagnostics=dict(verified=True),stationary_diagnostics=dict(verified=True))
        case=dict(selected=candidate,demand_audit=dict(verified=True),
            campaign_fingerprint=manifest["fingerprint"],source_campaign_fingerprint="original")
        RUN.save_checked(retry/"solution",case)
        return campaign,original,retry,case

    def test_accepts_certified_retry_with_unchanged_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            campaign,original,retry,_=self.prepare(root)
            inputs={}
            with patch.object(PLOT,"ROOT",root):
                manifest,case=PLOT.load_oe_retry(retry,campaign,original,inputs)
            self.assertEqual(case["campaign_fingerprint"],manifest["fingerprint"])
            self.assertIn("retry/manifest.json",inputs)
            self.assertIn("original/manifest.json",inputs)

    def test_rejects_changed_inputs_and_failed_certificates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            campaign,original,retry,case=self.prepare(root)
            with patch.object(PLOT,"ROOT",root):
                case["selected"]["dp_diagnostics"]["verified"]=False
                RUN.save_checked(retry/"solution",case)
                with self.assertRaisesRegex(ValueError,"certified equilibrium"):
                    PLOT.load_oe_retry(retry,campaign,original,{})
                (campaign/"manifest.json").write_text("{}")
                with self.assertRaisesRegex(ValueError,"input or code changed"):
                    PLOT.load_oe_retry(retry,campaign,original,{})

    def test_rejects_other_source_campaign(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            campaign,original,retry,_=self.prepare(root)
            original["fingerprint"]="different"
            with patch.object(PLOT,"ROOT",root),self.assertRaisesRegex(ValueError,"another experiment"):
                PLOT.load_oe_retry(retry,campaign,original,{})


if __name__=="__main__":
    unittest.main()
