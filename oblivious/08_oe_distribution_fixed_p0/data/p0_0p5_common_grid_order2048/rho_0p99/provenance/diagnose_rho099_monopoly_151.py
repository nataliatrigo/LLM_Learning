"""Independent monopoly-only evidence for rho=.99 grid sensitivity.

Uses the unchanged production demand integration and full demand audit, without
constructing or solving the competitive joint chain. No OE certificate is
claimed for this standalone diagnostic.
"""
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace
import json
import sys
import time

import numpy as np

ROOT = Path('/home/natalia/Projects/LLM_Learning')
sys.path.insert(0, str(ROOT))
from oblivious.oe_distribution_fixed_p0.code.model import outside_demand_matrix
from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
    HOME, audit_demand, check_bundle, numerical_sources, now,
    save_bundle, solve_monopoly, write_json,
)
from oblivious.oe_p0.code.primitives import Parameters

started = time.monotonic()
directory = HOME/'data/p0_0p5_common_grid_order2048/rho_0p99'
params = Parameters(rho=.99)
grid = np.linspace(0, 1, 151)
print('Production demand and independent audit', flush=True)
demand, production = outside_demand_matrix(grid, params, .5, 2048)
experiment = SimpleNamespace(
    grid=grid, params=params, p0=.5, order=2048,
    game=SimpleNamespace(D=demand, demand_diagnostics=production))
audit = audit_demand(experiment)
print('Monopoly certificate', flush=True)
monopoly = solve_monopoly(params, 151, .5)
manifest = json.loads((directory/'manifest.json').read_text())
coarse = check_bundle(directory, 'rho_0p99_n101', manifest)
coarse_outcome = next(r for r in coarse['outcomes'] if r['model']=='monopoly')
sold = monopoly['mu']*monopoly['demand']
rate = float(sold.sum())
use = float(sold@monopoly['policy']/rate)
profit = float(sold@(.95-.6*monopoly['policy']))
fine = dict(rho=.99, n=151, p0=.5, model='monopoly', verified=True,
    product2_conditional=use, product2_calendar=float(sold@monopoly['policy']),
    sale_rate=rate, profit_calendar=profit,
    value_initial=float(monopoly['value'][75]), outside_rate=1-rate)
comparison = dict(coarse_n=101, fine_n=151,
    product2_conditional_absolute_difference=abs(use-coarse_outcome['product2_conditional']),
    sale_rate_absolute_difference=abs(rate-coarse_outcome['sale_rate']),
    profit_calendar_relative_difference=abs(profit-coarse_outcome['profit_calendar'])/abs(profit),
    value_initial_relative_difference=abs(fine['value_initial']-coarse_outcome['value_initial'])/abs(fine['value_initial']))
comparison['passed'] = bool(comparison['sale_rate_absolute_difference']<=.05
    and comparison['product2_conditional_absolute_difference']<=.05
    and comparison['profit_calendar_relative_difference']<=.05
    and comparison['value_initial_relative_difference']<=.05)
case = dict(rho=.99, n=151, p0=.5, order=2048, scope='monopoly-only grid-sensitivity diagnostic; no competitive OE certificate',
    monopoly=monopoly, demand_audit=audit, demand_matrix=demand, outcome=fine,
    coarse_outcome=coarse_outcome, comparison=comparison,
    numerical_sources=numerical_sources(),
    source_script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    input_checkpoint_sha256={key: value for key,value in manifest['checkpoint_sha256'].items()
                            if key.startswith('rho_0p99_n101.')},
    created_utc=now(), elapsed_seconds=time.monotonic()-started)
prefix=directory/'diagnostics/rho_0p99_n151_monopoly_only'
prefix.parent.mkdir(parents=True, exist_ok=True)
save_bundle(prefix, case)
write_json(directory/'diagnostics/manifest.json', dict(status='completed_monopoly_grid_sensitivity_diagnostic',
    created_utc=now(), numerical_sources=numerical_sources(),
    source_script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    checkpoint_sha256={prefix.name+suffix:sha256(prefix.with_suffix(suffix).read_bytes()).hexdigest()
                      for suffix in ('.json','.npz')}, comparison=comparison,
    use_in_main_comparison=False))
print(json.dumps(dict(outcome=fine, comparison=comparison, elapsed_seconds=case['elapsed_seconds'])), flush=True)
