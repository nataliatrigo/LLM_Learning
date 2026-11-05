"""Finish the existing campaign's 101-point cases before running finer grids.

This is an operational controller: it invokes the original, unchanged workers
and keeps the campaign's scientific settings, hashes and fingerprint intact.
Stop the original controller with Ctrl+C before starting this one.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


SCRIPT = Path(__file__).with_name("run_refinement.py")
SPEC = importlib.util.spec_from_file_location("original_comparison_refinement", SCRIPT)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


@contextmanager
def campaign_lock(directory):
    """Exclude another invocation of this launcher, including between stages."""
    with (Path(directory) / ".finish_101.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another finish_101 controller is active for this campaign") from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()) + "\n")
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def complete_case(directory, manifest, model, rho):
    """Check the same checked arrays and provenance used by the original runner."""
    folder = RUN.case_dir(directory, model, rho, 101)
    path = folder / "status.json"
    if not path.exists() or json.loads(path.read_text()).get("status") != "complete":
        return False
    saved = RUN.load_checked(folder / "solution")
    if (saved is None or not RUN.verified(saved.get("selected"))
            or saved.get("campaign_fingerprint") != manifest["fingerprint"]):
        return False
    policy_hash = sha256(saved["selected"]["policy"].tobytes()).hexdigest()
    for mode in manifest["configuration"]["simulation_modes"]:
        simulation = RUN.load_checked(folder / ("simulation_" + mode))
        if (simulation is None
                or simulation.get("campaign_fingerprint") != manifest["fingerprint"]
                or simulation.get("policy_sha256") != policy_hash):
            return False
    return True


def stop_children(processes):
    """Signal only subprocess groups created by this controller."""
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for process in processes:
        try:
            process.wait(timeout=max(.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            except ProcessLookupError:
                pass


def active_campaign_processes(directory, proc_root=Path("/proc")):
    """Find visible local controllers/workers for this project and campaign.

    This is an additional guard for explicit recovery, not proof that a process
    in another host or PID namespace has stopped. The recovery flag means the
    caller has already stopped the original controller in its own terminal.
    """
    directory = Path(directory).resolve()
    scripts = {SCRIPT.resolve(), Path(__file__).resolve()}
    script_names = {path.name for path in scripts}
    if not proc_root.is_dir():
        raise RuntimeError("No se pueden inspeccionar los procesos locales; "
                           "la recuperación requiere acceso de lectura a /proc")
    active = []

    def option(arguments, name, default=None):
        for index, argument in enumerate(arguments):
            if argument == name:
                return arguments[index + 1] if index + 1 < len(arguments) else default
            if argument.startswith(name + "="):
                return argument.split("=", 1)[1]
        return default

    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError("No se pueden inspeccionar los procesos locales; "
                           "la recuperación requiere acceso de lectura a /proc") from exc
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            arguments = [part.decode(errors="replace") for part in
                         (entry / "cmdline").read_bytes().split(b"\0") if part]
            if not any(Path(argument).name in script_names for argument in arguments):
                continue
            cwd = (entry / "cwd").resolve(strict=True)
            script_index = next((index for index, argument in enumerate(arguments)
                                 if Path(argument).name in script_names
                                 and (cwd / argument).resolve() in scripts), None)
            if script_index is None:
                continue
            arguments = arguments[script_index + 1:]
            if "--worker" in arguments:
                target = option(arguments, "--directory")
                same_campaign = target is not None and (cwd / target).resolve() == directory
            else:
                name = option(arguments, "--name", "refinement_101_201")
                same_campaign = (RUN.OUTPUT / name).resolve() == directory
            if same_campaign:
                active.append(dict(pid=int(entry.name), command=" ".join(arguments)))
        except (FileNotFoundError, ProcessLookupError):
            continue  # A process may finish between listing and inspection.
        except PermissionError as exc:
            raise RuntimeError("No se puede inspeccionar un proceso local de este usuario; "
                               "se cancela la recuperación") from exc
    return active


def execute_101(directory, workers=3, recover_interrupted=False):
    directory = Path(directory)
    if workers < 1:
        raise ValueError("workers must be positive")
    with campaign_lock(directory):
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        recovery = None
        if manifest.get("status") == "running":
            if not recover_interrupted:
                raise RuntimeError("La campaña está marcada como running; esa marca puede quedar "
                                   "obsoleta tras una interrupción. Si el controlador anterior "
                                   "sigue ejecutándose, presiona Ctrl+C en su terminal y espera "
                                   "a que vuelva el prompt. Una vez detenido, usa "
                                   "--recover-interrupted. No ejecutes ambos controladores juntos.")
            active = active_campaign_processes(directory)
            if active:
                pids = ", ".join(str(process["pid"]) for process in active)
                raise RuntimeError(f"Todavía hay procesos activos de esta campaña (PID {pids}). "
                                   "Presiona Ctrl+C en la terminal original y espera a que vuelva "
                                   "el prompt antes de usar --recover-interrupted.")
            recovery = dict(recovered_utc=RUN.stamp(), previous_status="running",
                            confirmation="Caller supplied --recover-interrupted after stopping "
                                         "the original controller", local_process_check="clear")
        if RUN.source_hashes() != manifest["source_sha256"]:
            raise ValueError("Original campaign sources changed; refusing to mix numerical versions")
        config = manifest["configuration"]
        if 101 not in config["grids"]:
            raise ValueError("This campaign does not include a 101-point grid")

        metadata_path = directory / "finish_101.json"
        metadata = (json.loads(metadata_path.read_text()) if metadata_path.exists()
                    else dict(version=1, runs=[]))
        operational = dict(started_utc=RUN.stamp(), status="running", pid=os.getpid(),
                           workers=workers, grid=101,
                           launcher_sha256=RUN.digest(Path(__file__).resolve()),
                           campaign_fingerprint=manifest["fingerprint"])
        if recovery is not None:
            operational["recovery"] = recovery
        metadata["runs"].append(operational)
        RUN.atomic_json(metadata_path, metadata)
        manifest.update(status="running", updated_utc=RUN.stamp())
        RUN.atomic_json(manifest_path, manifest)
        children, lock, stop = {}, threading.Lock(), threading.Event()
        failures = []

        def run_one(model, rho):
            if stop.is_set():
                return None
            if complete_case(directory, manifest, model, rho):
                return 0
            folder = RUN.case_dir(directory, model, rho, 101)
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "worker.log").open("a") as log:
                with lock:
                    if stop.is_set():
                        return None
                    process = subprocess.Popen(
                        [sys.executable, "-u", "-B", str(SCRIPT), "--worker", model,
                         str(rho), "101", "--directory", str(directory)],
                        cwd=RUN.ROOT, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    children[process.pid] = process
                try:
                    return process.wait()
                finally:
                    with lock:
                        children.pop(process.pid, None)

        try:
            print("Completing grid 101 only; finer-grid checkpoints remain saved. "
                  "Do not start another campaign controller.", flush=True)
            for model in config["models"]:
                print(f"\n{RUN.LABELS[model]}: n=101, up to {workers} concurrent rho", flush=True)
                executor = ThreadPoolExecutor(max_workers=workers)
                futures = {executor.submit(run_one, model, rho): rho for rho in
                           sorted(config["rhos"], key=lambda r: (r not in (.95, .97, .99), r))}
                try:
                    for future in as_completed(futures):
                        rho = futures[future]
                        code = future.result()
                        print(f"  {model}, rho={rho:g}, n=101: " +
                              ("complete" if code == 0 else f"needs review (exit {code})"), flush=True)
                        if code != 0:
                            failures.append(dict(model=model, rho=rho, n=101, exit_code=code))
                        RUN.export_campaign(directory)
                except BaseException:
                    stop.set()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                else:
                    executor.shutdown(wait=True)
            rows = RUN.export_campaign(directory)
            finished_101 = all(complete_case(directory, manifest, model, rho)
                               for model in config["models"] for rho in config["rhos"])
            state = ("awaiting_finer_grids" if finished_101 else "partial_complete")
            if finished_101 and all(row["status"] == "complete" for row in rows):
                state = "complete"
            manifest.update(status=state, updated_utc=RUN.stamp())
            RUN.atomic_json(manifest_path, manifest)
            operational.update(status=state, finished_utc=RUN.stamp(), failures=failures)
            RUN.atomic_json(metadata_path, metadata)
            print(f"\n101-point cases: {'complete' if finished_101 else 'need review'}. "
                  f"Results: {directory / 'summary.csv'}", flush=True)
            return 0 if finished_101 else 2
        except BaseException as exc:
            stop.set()
            with lock:
                processes = list(children.values())
            stop_children(processes)
            state = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            detail = dict(type=type(exc).__name__, message=str(exc))
            manifest.update(status=state, updated_utc=RUN.stamp(), last_error=detail)
            RUN.atomic_json(manifest_path, manifest)
            operational.update(status=state, finished_utc=RUN.stamp(), error=detail)
            RUN.atomic_json(metadata_path, metadata)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="refinement_101_201")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--recover-interrupted", action="store_true",
                        help="Recuperar una marca running obsoleta después de detener el "
                             "controlador original con Ctrl+C y esperar a que vuelva el prompt")
    args = parser.parse_args(argv)
    if Path(args.name).name != args.name or args.name in ("", ".", "..") or args.workers < 1:
        parser.error("Require a campaign name and a positive worker count")
    directory = RUN.OUTPUT / args.name
    path = directory / "manifest.json"
    if not path.exists():
        parser.error("No existing campaign with this name")
    if args.status:
        manifest = json.loads(path.read_text())
        print(f"Campaign: {manifest.get('status', 'unknown')}")
        for model in manifest["configuration"]["models"]:
            for rho in manifest["configuration"]["rhos"]:
                folder = RUN.case_dir(directory, model, rho, 101)
                status_path = folder / "status.json"
                status = json.loads(status_path.read_text()).get("status") if status_path.exists() else "pending"
                print(f"{model:11s} rho={rho:g} n=101: {status}")
        return 0
    try:
        return execute_101(directory, args.workers, args.recover_interrupted)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
