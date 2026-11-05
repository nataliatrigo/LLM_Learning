"""Freshly certify and atomically publish an independently found n101 proposal.

Run once from the repository root after stopping the original .97 worker.
The positive face is a search proposal, not a maintained policy restriction:
Fresh evaluate() verifies every action inequality, exact OE closure, and
positive invariant law with unchanged tolerances before anything is exported.
"""
from pathlib import Path
from hashlib import sha256
import json
import sys
import time
import numpy as np
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'oblivious/layout.py').exists())
sys.path.insert(0, str(ROOT))
from oblivious.oe_distribution_fixed_p0.code.refine_case import source_fingerprints, atomic_bundle
from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
    audit_demand, economic_outcomes, numerical_sources, require, solve_monopoly, write_json, now,
)
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.run_experiment import load_bundle

def main():
    directory = Path(__file__).resolve().parents[1]
    provenance = directory/'provenance'
    path = directory/'manifest.json'
    manifest = json.loads(path.read_text())
    require(manifest['numerical_sources'] == source_fingerprints(), 'Sources changed')
    require(not manifest['completed'], 'n101 is already committed; refusing overwrite')
    proposal = load_bundle(provenance/'rho097_positive_face_n101')
    require(proposal['source_fingerprints'] == numerical_sources(), 'Proposal sources changed')
    policy = proposal['competition']['policy']
    require(proposal['competition']['verified'], 'Proposal has no original certificate')
    started = time.monotonic()
    params = Parameters(rho=.97)
    experiment = FixedOutsideMarginalExperiment(n=101, params=params, p0=.5, order=256,
                                               tol=1e-7, dp_tol=1e-10)
    print('Fresh demand audit', flush=True)
    demand = audit_demand(experiment)
    monopoly = solve_monopoly(params, 101, .5, 1e-10)
    print('Fresh unchanged full OE certificate', flush=True)
    competition = experiment.evaluate(policy)
    require(competition['verified'], 'Fresh proposal certificate failed')
    competition['history'] = proposal['search_history']
    competition['search_metadata'] = dict(
        method=proposal['method'], source='provenance/positive_face097_one.py',
        exploratory_evaluations=8, fresh_certifications=1, exhaustive=False, unique=False,
        policy_restriction='None: all endpoint inequalities checked by final certificate')
    print('Actual joint discounted values', flush=True)
    n=experiment.n
    full=experiment.game.evaluate(np.broadcast_to(policy[:,None],(n,n)),
                                  np.broadcast_to(policy[None,:],(n,n)),tol=1e-10)
    competition['actual_values']=full['values']
    competition['actual_value_diagnostics']={k:v for k,v in full.items() if k!='values'}
    competition['attempts']=[dict(initialization='independent positive stationary proposal',
                                 verified=True, diagnostics=competition['dp_diagnostics'],
                                 search=competition['search_metadata'])]
    outcomes, policies = economic_outcomes(params, n, .5, monopoly, competition, experiment.game)
    case = dict(rho=.97, n=n, p0=.5, configuration=experiment.configuration(),
                monopoly=monopoly, competition=competition, demand_audit=demand,
                outcomes=outcomes, policy_rows=policies, elapsed_seconds=time.monotonic()-started,
                refinement_metadata=dict(seed_grid=51, seed_is_proposal_only=True,
                    certificate='Fresh unchanged model/optimality/stationarity/demand checks',
                    previous_completed_tag='rho_0p97_n051',
                    supplementary_search='Independent positive-stationary mixed-face root; every action checked',
                    search_log='provenance/positive_face097_one.log',
                    interrupted_prior_search='Coarse interpolated local solve stalled at .00327; signed homotopy stopped after temperature .0003 to use a fully verified positive-stationary proposal'))
    require(manifest['numerical_sources'] == source_fingerprints(), 'Sources changed during certificate')
    manifest['checkpoint_sha256'].update(atomic_bundle(directory,'rho_0p97_n101',case))
    manifest['completed'].append('rho_0p97_n101')
    manifest['failures'].pop('rho_0p97_n101',None)
    manifest.update(status='running', current_grid=None, updated_utc=now())
    manifest['supplemental_search_provenance']={str(p.relative_to(directory)):sha256(p.read_bytes()).hexdigest()
                                               for p in sorted(provenance.iterdir()) if p.is_file()}
    write_json(path,manifest)
    print('CERTIFIED n101',json.dumps(outcomes), 'elapsed',case['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
