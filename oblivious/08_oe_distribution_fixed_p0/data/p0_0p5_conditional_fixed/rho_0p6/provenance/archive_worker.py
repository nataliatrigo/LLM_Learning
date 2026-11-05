"""Archive immutable completed worker diagnostics without changing any case."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

from oblivious.oe_distribution_fixed_p0.code.conditional_run import source_fingerprints
from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
    HOME, ROOT, case_tag, check_bundle, now, require, write_json,
)

rho = float(sys.argv[1])
require(rho in (.6, .7), 'This helper owns only rho .6 and .7')
directory = HOME/'data/p0_0p5_conditional_fixed'/('rho_'+str(rho).replace('.', 'p'))
manifest_path = directory/'manifest.json'
manifest = json.loads(manifest_path.read_text())
require(manifest['status'] in ('grid_stable_under_declared_thresholds',
                              'certified_but_grid_sensitive'), 'Worker has not closed')
require(manifest['current_grid'] is None, 'Worker still has an active grid')
require(manifest['numerical_sources'] == source_fingerprints(), 'Numerical source mismatch')
records = []
for n in (51, 101):
    case = check_bundle(directory, case_tag(rho, n), manifest)
    competition = case['competition']
    require(competition['verified'], 'Case is not certified')
    require(competition['conditional_diagnostics']['independent_order_verified'],
            'Reversed-order conditional certificate missing')
    for source, expected in case['reference_input_sha256'].items():
        require(sha256((ROOT/source).read_bytes()).hexdigest() == expected,
                'Historical reference changed')
    records.append(dict(n=n, verified=competition['verified'],
        dp=competition['dp_diagnostics'],
        conditional=competition['conditional_diagnostics'],
        stationary=competition['stationary_diagnostics'],
        demand_audit=case['demand_audit'],
        actual_value_diagnostics=competition['actual_value_diagnostics'],
        outcomes=case['outcomes'], elapsed_seconds=case['elapsed_seconds'],
        initialization_attempts=competition.get('attempts', [])))
provenance = directory/'provenance'
provenance.mkdir(exist_ok=True)
log = Path('/tmp')/('fixed_conditional_rho0'+str(int(round(10*rho)))+'.log')
log_hash = sha256(log.read_bytes()).hexdigest()
shutil.copyfile(log, provenance/'solver.log')
require(sha256((provenance/'solver.log').read_bytes()).hexdigest() == log_hash,
        'Archived log mismatch')
shutil.copyfile(Path(__file__), provenance/'archive_worker.py')
write_json(provenance/'verification_diagnostics.json', dict(
    archived_utc=now(), rho=rho, numerical_sources_unchanged=True,
    historical_references_unchanged=True, cases=records,
    grid_stability=manifest['grid_stability']))
artifacts = {str(path.relative_to(directory)): sha256(path.read_bytes()).hexdigest()
             for path in (provenance/'solver.log', provenance/'archive_worker.py',
                          provenance/'verification_diagnostics.json')}
manifest.update(worker_archive_sha256=artifacts, diagnostic_artifact_sha256=artifacts,
                worker_archive_utc=now())
write_json(manifest_path, manifest)
print(json.dumps(dict(rho=rho, status=manifest['status'],
                      grid_stability=manifest['grid_stability'],
                      archive_sha256=artifacts)))
