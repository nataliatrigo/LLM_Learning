"""Scientific accounting and branch preservation for the two fluid models."""
import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluid-oblivious-mpl")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oblivious.fluid_oblivious.code.reporting import (
    _curves, _normalise_fluid, quantitative_comparisons,
)


class ReportingTests(unittest.TestCase):
    def row(self, **changes):
        row=dict(p0=.5,rho=.7,model="fluid_competition",equilibrium_id=0,
                 status="verified",verified=True,x_star=.55,alpha=.4,
                 sale_rate=.3,outside_rate=.4,product2_calendar=.12)
        row.update(changes)
        return row

    def monopoly(self, **changes):
        return self.row(model="fluid_monopoly",sale_rate=.6,outside_rate=.4,
                        product2_calendar=.24,**changes)

    def test_seller_count_changes_customer_flow_not_conditional_mix(self):
        data=_normalise_fluid([self.row(),self.monopoly()])
        competition,monopoly=data.iloc[0],data.iloc[1]
        self.assertEqual(competition.product2_conditional,.4)
        self.assertEqual(monopoly.product2_conditional,.4)
        self.assertEqual(competition.seller_count,2)
        self.assertEqual(monopoly.seller_count,1)
        self.assertAlmostEqual(competition.product2_calendar,.12)
        self.assertAlmostEqual(competition.customer_product2_calendar,.24)
        self.assertAlmostEqual(monopoly.product2_calendar,.24)
        self.assertAlmostEqual(monopoly.customer_product2_calendar,.24)

    def test_rejects_wrong_market_and_customer_accounting(self):
        for invalid in [self.row(outside_rate=.3),self.row(product2_calendar=.4),
                        self.monopoly(customer_product2_calendar=.48),
                        self.monopoly(seller_count=2)]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _normalise_fluid([invalid])

    def test_no_equilibrium_is_not_shown_as_zero(self):
        row=self.row(verified=False,status="no_equilibrium_found",alpha=np.nan,
                     x_star=np.nan,sale_rate=np.nan,product2_calendar=np.nan,outside_rate=np.nan)
        data=_normalise_fluid([row])
        self.assertFalse(data.iloc[0].verified)
        self.assertTrue(np.isnan(data.iloc[0].product2_conditional))

    def test_both_models_keep_all_branches_without_arbitrary_connection(self):
        data=_normalise_fluid([self.row(),self.row(equilibrium_id=1,alpha=.8,product2_calendar=.24)])
        fig,ax=plt.subplots()
        _curves(ax,data,horizontal="rho",positions={.7:0},metric="product2_conditional")
        self.assertTrue(np.isnan(ax.lines[1].get_ydata()[0]))
        points=np.concatenate([collection.get_offsets() for collection in ax.collections])
        self.assertEqual(len(points),2)
        np.testing.assert_allclose(sorted(points[:,1]),[.4,.8])
        plt.close(fig)

    def test_quantitative_pairs_and_isolated_holding_zero_mass(self):
        summary=pd.DataFrame([self.row(),self.monopoly()])
        policies=pd.DataFrame([
            dict(p0=.5,rho=.7,model=model,equilibrium_id=0,reputation=x,
                 product2_probability=a,holding=False)
            for model,pairs in [("fluid_competition",[(0.,0.),(1.,1.)]),
                                ("fluid_monopoly",[(0.,0.),(1.,0.)])]
            for x,a in pairs])
        compared=quantitative_comparisons(summary,policies).iloc[0]
        self.assertAlmostEqual(compared.competition_minus_monopoly_sale_rate,-.3)
        self.assertAlmostEqual(compared.competition_minus_monopoly_customer_product2_calendar,0.)
        self.assertAlmostEqual(compared.policy_l1_length,.5)
        self.assertAlmostEqual(compared.policy_mismatch_fraction_of_state_interval,.5)
        policies.loc[len(policies)]=dict(p0=.5,rho=.7,model="fluid_competition",equilibrium_id=0,
                                        reputation=.75,product2_probability=.5,holding=True)
        repeated=quantitative_comparisons(summary,policies).iloc[0]
        self.assertAlmostEqual(repeated.policy_l1_length,.5)

    def test_all_stationary_branch_pairs_are_exported(self):
        rows=[self.row(),self.row(equilibrium_id=1,alpha=.8,product2_calendar=.24),
              self.monopoly(),self.monopoly(equilibrium_id=1)]
        policies=pd.DataFrame([dict(p0=.5,rho=.7,model=r["model"],equilibrium_id=r["equilibrium_id"],
                                   reputation=x,product2_probability=0.) for r in rows for x in [0.,1.]])
        pairs=quantitative_comparisons(pd.DataFrame(rows),policies)
        self.assertEqual(len(pairs),4)
        self.assertEqual(set(zip(pairs.competition_equilibrium_id,pairs.monopoly_equilibrium_id)),
                         {(0,0),(0,1),(1,0),(1,1)})
        self.assertTrue(pairs.comparison_status.eq("verified_pair").all())

    def test_stochastic_series_are_rejected(self):
        with self.assertRaises(ValueError):
            _normalise_fluid([self.row(model="competition")])


if __name__=="__main__":
    unittest.main()
