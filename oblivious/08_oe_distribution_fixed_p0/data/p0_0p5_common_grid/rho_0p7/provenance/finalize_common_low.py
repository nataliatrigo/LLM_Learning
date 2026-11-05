"""Archive interrupted optional n151 searches after certified n101 publication."""
from pathlib import Path
from hashlib import sha256
import json
import shutil
import sys

ROOT = Path('/home/natalia/Projects/LLM_Learning')
sys.path.insert(0, str(ROOT))
from oblivious.oe_distribution_fixed_p0.code.refine_case import source_fingerprints
from oblivious.oe_distribution_fixed_p0.code.run_experiment import check_bundle, now, write_json

BASE = ROOT/'oblivious/08_oe_distribution_fixed_p0/data/p0_0p5_common_grid'
processes = {'0p6': (34806, 'SIGINT', 130), '0p7': (35413, 'SIGINT', 130),
             '0p8': (35411, 'SIGINT then SIGTERM', 143),
             '0p9': (35412, 'SIGINT then SIGTERM', 143)}
for label, (pid, signal, exit_code) in processes.items():
    directory = BASE/('rho_'+label)
    path = directory/'manifest.json'
    manifest = json.loads(path.read_text())
    assert manifest['numerical_sources'] == source_fingerprints()
    tag = 'rho_'+label+'_n101'
    case = check_bundle(directory, tag, manifest)
    assert case['n'] == 101 and case['p0'] == .5
    assert case['monopoly']['verified'] and case['competition']['verified']
    assert case['demand_audit']['verified']
    assert 'actual_values' in case['competition'] and 'actual_value_diagnostics' in case['competition']
    assert all(row['passed'] for row in manifest['convergence'])
    provenance = directory/'provenance'
    provenance.mkdir(exist_ok=True)
    log = provenance/'solver.log'
    shutil.copyfile(Path('/tmp')/('common_grid_rho'+label.replace('p','')+'.log'), log)
    helper = provenance/Path(__file__).name
    shutil.copyfile(__file__, helper)
    audit = dict(timestamp_utc=now(), published_grid=101, optional_interrupted_grid=151,
        reason='User authorized either 101 or 151. Team selected common n101 after all four new n101 cases were fully certified; optional n151 computations were interrupted to avoid unnecessary delay.',
        original_process=dict(pid=pid, signal=signal, exit_code=exit_code),
        certified_checkpoint=tag,
        checkpoint_sha256={tag+s: manifest['checkpoint_sha256'][tag+s] for s in ('.json','.npz')},
        solver_log_sha256=sha256(log.read_bytes()).hexdigest(),
        archival_helper_sha256=sha256(helper.read_bytes()).hexdigest(),
        numerical_sources=source_fingerprints(),
        competition_dp_diagnostics=case['competition']['dp_diagnostics'],
        elapsed_seconds=case['elapsed_seconds'],
        comparison_51_to_101=manifest['convergence'],
        future_resume='Original configuration/grids and numerical sources are preserved. refine_case --resume can continue the optional n151 case.')
    write_json(provenance/'common_grid_101_audit.json', audit)
    manifest.update(status='stopped_after_certified_common_grid_101', current_grid=None,
        published_grid=101, updated_utc=now(), finished_utc=now(),
        common_grid_selection=dict(grid=101, comparison_grid=51,
            optional_grid_151='Interrupted after selecting common n101; not published or certified.',
            reason=audit['reason'], provenance=str((provenance/'common_grid_101_audit.json').relative_to(ROOT))))
    write_json(path, manifest)
    print(label, manifest['status'], case['elapsed_seconds'], 'all checks passed')
