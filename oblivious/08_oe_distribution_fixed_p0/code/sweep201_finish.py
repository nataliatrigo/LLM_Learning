"""Resume the authorized background sweep, then audit, publish and verify it.

This is orchestration only. A failed compute or audit stage stops the pipeline
before reporting. The final index is written only after artifact verification.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from oblivious.oe_distribution_fixed_p0.code.run_experiment import HOME, ROOT, now, write_json
from oblivious.oe_distribution_fixed_p0.code.sweep201_report import (
    CAMPAIGN, GRIDS, MODELS, P0S, RHOS, digest, p0_name, verify_inputs,
)


PACKAGE = 'oblivious.oe_distribution_fixed_p0.code.'
STAGES = (
    ('compute', PACKAGE+'sweep201_complementarity_entry', ['--batch', '--workers', '6', '--resume']),
    ('audit', PACKAGE+'sweep201_audit', ['--workers', '6']),
    ('report', PACKAGE+'sweep201_report', []),
)
STEMS = ('01_policy_comparison', '02_product2_usage_comparison',
         '03_profit_comparison', '04_outside_option_comparison')


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def exclusive_finalizer(data):
    """Hold one nonblocking pipeline lock for compute, audit and publication."""
    with (Path(data)/'finalization.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another sweep201 finalizer already holds the pipeline lock') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def active_controllers(proc=Path('/proc')):
    """Find direct sweep controllers as well as other finalizer processes."""
    found = []
    for path in Path(proc).glob('[0-9]*/cmdline'):
        if path.parent.name == str(os.getpid()):
            continue
        try:
            args = path.read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if '-m' not in args:
            continue
        index = args.index('-m')+1
        if index >= len(args):
            continue
        module = args[index]
        if (module == PACKAGE+'sweep201_finish'
                or module == PACKAGE+'sweep201_batch'
                or module.startswith(PACKAGE+'sweep201_') and '--batch' in args):
            found.append(dict(pid=int(path.parent.name), module=module))
    return found


def require_closed_campaign(data):
    path = Path(data)/'sweep_manifest.json'
    require(path.is_file(), 'An existing sweep manifest is required')
    state = json.loads(path.read_text())
    require(state.get('status') != 'running', 'The existing sweep controller is still running')
    for path in Path(data).glob('p0_*/rho_*/manifest.json'):
        case = json.loads(path.read_text())
        require(case.get('status') != 'running' and case.get('current_grid') is None
                and case.get('current_model') is None,
                'A target campaign is still running: '+str(path))
    require(not active_controllers(), 'Another sweep201 controller/finalizer process is active')


def prior_figures():
    directory = HOME/'figures/p0_sweep'
    paths = sorted(path for p0 in P0S for suffix in ('png', 'pdf')
                   for path in (directory/p0_name(p0)).glob('*.'+suffix))
    require(len(paths) == 48, 'Expected exactly 48 preserved historical n101 figure files')
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def source_hashes():
    paths = [*HOME.joinpath('code').glob('sweep201_*.py'),
             *HOME.joinpath('code').glob('sweep201_*.cpp')]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def csv_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def verify_artifacts(data, figures):
    report_path = Path(data)/'comparison_report_manifest.json'
    report = json.loads(report_path.read_text())
    require(report.get('status') in ('complete', 'complete_with_sensitivity_exclusions')
            and report.get('all_required_cases_certified') is True
            and report.get('requested_p0s') == P0S and report.get('requested_rhos') == RHOS
            and report.get('required_grids') == GRIDS and report.get('displayed_n') == 201
            and report.get('validation_n') == 251
            and report.get('independent_audit', {}).get('verified') is True,
            'Generated report does not certify the complete requested campaign')
    for name in ('source_input_sha256', 'reporting_source_sha256',
                 'generated_data_sha256', 'figure_files'):
        require(bool(report.get(name)), 'Generated report has no '+name)
        verify_inputs(report[name])
    models = {model for model, *_ in MODELS}
    expected_all = {(p0, rho, n, model) for p0 in P0S for rho in RHOS
                    for n in GRIDS for model in models}
    rows = csv_rows(Path(data)/'comparison_summary_all_grids.csv')
    keys = [(float(row['p0']), float(row['rho']), int(row['n']), row['model']) for row in rows]
    require(len(rows) == 144 and len(keys) == len(set(keys)) and set(keys) == expected_all
            and all(row['verified'].lower() == 'true' for row in rows),
            'All-grid summary does not contain exactly 144 certified model/grid outcomes')
    retained = report.get('common_retained_rhos', [])
    require(bool(retained) and retained == sorted(set(retained)) and set(retained) <= set(RHOS),
            'Generated report has no valid common retained rho set')
    expected_display = {(p0, rho, 201, model) for p0 in P0S for rho in retained for model in models}
    displayed = csv_rows(Path(data)/'comparison_summary.csv')
    keys = [(float(row['p0']), float(row['rho']), int(row['n']), row['model']) for row in displayed]
    require(len(displayed) == 6*len(retained)*3 and len(keys) == len(set(keys))
            and set(keys) == expected_display, 'Displayed summary mixes or omits original n201 outcomes')
    policies = csv_rows(Path(data)/'comparison_policies.csv')
    require(len(policies) == 201*len(displayed), 'Displayed policy export has an incorrect row count')
    grouped = {}
    for row in policies:
        key = (float(row['p0']), float(row['rho']), int(row['n']), row['model'])
        grouped.setdefault(key, []).append(float(row['reputation']))
    require(set(grouped) == expected_display
            and all(np.array_equal(sorted(nodes), np.linspace(0, 1, 201))
                    for nodes in grouped.values()), 'Exported policies do not retain the original 201 nodes')
    expected_files = {Path(figures)/p0_name(p0)/(stem+'.'+suffix)
                      for p0 in P0S for stem in STEMS for suffix in ('png', 'pdf')}
    actual_files = set(Path(figures).rglob('*.png')) | set(Path(figures).rglob('*.pdf'))
    declared_files = {ROOT/name for name in report['figure_files']}
    require(actual_files == expected_files == declared_files,
            'Expected exactly 24 PNG and 24 PDF comparison figures')
    dimensions = {}
    for path in sorted(expected_files):
        if path.suffix == '.png':
            with Image.open(path) as bitmap:
                width, height = bitmap.size
                bitmap.verify()
            with Image.open(path) as bitmap:
                bitmap.load()
            require(width > 100 and height > 100, 'Figure dimensions are invalid: '+str(path))
            dimensions[str(path.relative_to(ROOT))] = [width, height]
        else:
            with path.open('rb') as handle:
                require(handle.read(5) == b'%PDF-', 'Figure is not a PDF: '+str(path))
    return report, dict(report=str(report_path.relative_to(ROOT)), report_sha256=digest(report_path),
        all_grid_rows=len(rows), displayed_rows=len(displayed), policy_rows=len(policies),
        common_retained_rhos=retained, png_count=24, pdf_count=24, png_dimensions=dimensions)


def write_index(figures, report):
    labels = ('Políticas', 'Producto 2 entre ventas propias', 'Beneficios', 'Outside del mercado')
    retained = ', '.join(f'{rho:g}' for rho in report['common_retained_rhos'])
    lines = ['# Comparación de p₀ con grilla de 201 puntos', '',
        'Modelos: monopolio, OE marginal y OE condicional.', '',
        'Cada curva usa sus 201 nodos originales y fue contrastada con 251 puntos. '
        f'Los ρ comunes que pasan el criterio de sensibilidad son: {retained}.', '',
        'Producto 2 mide el porcentaje de las ventas propias. Outside es el porcentaje de '
        'períodos en que ningún vendedor vende. Los beneficios descontados usan valores del juego real.', '',
        'La auditoría independiente y la verificación de los artefactos finalizaron correctamente. '
        'El contraste entre grillas no prueba convergencia continua ni unicidad.', '',
        '| p₀ | Comparación | PNG | PDF |', '|---|---|---|---|']
    paths = report['figure_files']
    for p0 in P0S:
        for stem, label in zip(STEMS, labels):
            links = []
            for suffix in ('png', 'pdf'):
                path = Path(figures)/p0_name(p0)/(stem+'.'+suffix)
                require(str(path.relative_to(ROOT)) in paths, 'Index figure is absent from the report')
                links.append(f'[{suffix.upper()}]({path.relative_to(figures).as_posix()})')
            lines.append(f'| {p0:g} | {label} | {links[0]} | {links[1]} |')
    path = Path(figures)/'README.md'
    path.write_text('\n'.join(lines)+'\n')
    return path


def finish():
    data, figures = HOME/'data'/CAMPAIGN, HOME/'figures'/CAMPAIGN
    require((data/'sweep_manifest.json').is_file(), 'Existing sweep manifest is required')
    require_closed_campaign(data)
    with exclusive_finalizer(data):
        require_closed_campaign(data)
        historical = prior_figures()
        sources = source_hashes()
        path = data/'finalization_manifest.json'
        previous = json.loads(path.read_text()) if path.exists() else None
        state = dict(status='running', started_utc=now(), pid=os.getpid(), stages=[],
            source_helper_sha256=sources, historical_n101_figure_sha256=historical,
            limitations='Convergence is not asserted before complete computation, independent audit and artifact verification')
        if previous is not None:
            state['previous_attempt'] = previous
        write_json(path, state)
        stage = 'preflight'

        def record_historical_preservation():
            try:
                verify_inputs(historical)
                state['historical_n101_figures_preserved'] = True
            except Exception as error:
                state['historical_n101_figures_preserved'] = False
                state['historical_preservation_error'] = f'{type(error).__name__}: {error}'

        try:
            environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                               MKL_NUM_THREADS='1', MPLCONFIGDIR='/tmp/sweep201-mpl')
            logs = data/'logs'
            logs.mkdir(exist_ok=True)
            for stage, module, options in STAGES:
                command = [sys.executable, '-B', '-m', module, *options]
                log = logs/('finalization_'+stage+'.log')
                record = dict(stage=stage, command=command, started_utc=now(),
                              log=str(log.relative_to(ROOT)))
                state['stages'].append(record)
                state['current_stage'] = stage
                write_json(path, state)
                print(f'[{now()}] Starting {stage}; log={log}', flush=True)
                with log.open('a') as handle:
                    result = subprocess.run(command, cwd=ROOT, env=environment,
                                            stdout=handle, stderr=subprocess.STDOUT, check=False)
                record.update(exit_code=result.returncode, finished_utc=now())
                write_json(path, state)
                if result.returncode != 0:
                    state.update(status='failed', failed_stage=stage, finished_utc=now(),
                                 error=f'{stage} exited with code {result.returncode}; later stages were not run')
                    record_historical_preservation()
                    write_json(path, state)
                    return result.returncode if result.returncode > 0 else 1
            stage = 'artifact_verification'
            state['current_stage'] = stage
            write_json(path, state)
            report, verified = verify_artifacts(data, figures)
            verify_inputs(sources)
            verify_inputs(historical)
            index = write_index(figures, report)
            verify_inputs(historical)
            state.update(status='complete_verified', finished_utc=now(), current_stage=None,
                verification=verified, report_sha256=verified['report_sha256'],
                result_paths=dict(report=verified['report'], index=str(index.relative_to(ROOT)),
                                  figures=str(figures.relative_to(ROOT))),
                index_sha256=digest(index), historical_n101_figures_preserved=True,
                visual_review='PNG decoding/dimensions and PDF signatures verified automatically; no human visual review claimed')
            write_json(path, state)
            print(f'[{now()}] Complete: {index}', flush=True)
            return 0
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                         failed_stage=stage, finished_utc=now(), error=f'{type(error).__name__}: {error}')
            record_historical_preservation()
            write_json(path, state)
            raise


def main():
    raise SystemExit(finish())


if __name__ == '__main__':
    main()
