"""Embed verified trajectory data in the interactive visualization."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "outputs/trajectory-data.json",
        help="Trajectory JSON to embed (default: outputs/trajectory-data.json).",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/trayectorias-fluido.html",
        help="Destination HTML (default: outputs/trayectorias-fluido.html).",
    )
    args = parser.parse_args(argv)
    data = json.loads(args.data.read_text(encoding="utf-8"))
    # Only quantities used by the visual are embedded; full precision remains
    # in the experiment's JSON and CSV outputs.
    reduced = {
        key: data[key] for key in ("parameters", "initial_states", "horizon", "times")
    }
    reduced["cases"] = [
        {
            "rho": case["rho"],
            "roots": case["roots"],
            "intervals": case["intervals"],
            "holding_points": case["holding_points"],
            "paths": [
                {"x0": path["x0"], "x": [round(x, 8) for x in path["x"]]}
                for path in case["paths"]
            ],
        }
        for case in data["cases"]
    ]
    fragment = (ROOT / "code/trajectory-view.html").read_text(encoding="utf-8")
    assert fragment.count("__TRAJECTORY_DATA__") == 1
    payload = json.dumps(reduced, separators=(",", ":"), ensure_ascii=False)
    fragment = fragment.replace("__TRAJECTORY_DATA__", payload.replace("</", "<\\/"))
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment, encoding="utf-8")
    assert output.stat().st_size < 1_000_000
    print(output)


if __name__ == "__main__":
    main()
