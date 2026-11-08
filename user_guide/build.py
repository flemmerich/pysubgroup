"""Build the user guide with Jupyter Book 2, keeping generated files book-local.

Run with a Python environment containing pysubgroup's dependencies, matplotlib,
ipykernel, and jupyter-server. Optional book-local Python tools in .tools/python
are detected automatically. The checkout's src directory is read directly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parent


def build_environment() -> dict[str, str]:
    """Route package, Jupyter, plotting, and tool caches into this book."""
    env = os.environ.copy()
    paths = {
        "HOME": BOOK / ".home",
        "XDG_CACHE_HOME": BOOK / ".cache",
        "XDG_CONFIG_HOME": BOOK / ".cache/config",
        "XDG_DATA_HOME": BOOK / ".cache/data",
        "TMPDIR": BOOK / ".cache/tmp",
        "MPLCONFIGDIR": BOOK / ".cache/matplotlib",
        "NUMBA_CACHE_DIR": BOOK / ".cache/numba",
        "IPYTHONDIR": BOOK / ".cache/ipython",
        "JUPYTER_CONFIG_DIR": BOOK / ".cache/jupyter-config",
        "JUPYTER_DATA_DIR": BOOK / ".cache/jupyter-data",
        "JUPYTER_RUNTIME_DIR": BOOK / ".cache/jupyter-runtime",
        "npm_config_cache": BOOK / ".cache/npm",
        "YARN_CACHE_FOLDER": BOOK / ".cache/yarn",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    python_paths = [BOOK / ".tools/python", BOOK.parent / "src"]
    env["PYTHONPATH"] = os.pathsep.join(map(str, python_paths))
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env["PATH"]])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # FUSE file systems may not apply Jupyter's requested mode to runtime JSON files.
    env["JUPYTER_ALLOW_INSECURE_WRITES"] = "1"
    env["npm_config_bin_links"] = "false"
    env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        env[name] = "1"
    kernel = paths["JUPYTER_DATA_DIR"] / "kernels/python3"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python 3 (pysubgroup book)",
        "language": "python",
    }, indent=2) + "\n")
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "html", "preview"])
    parser.add_argument("--no-execute", action="store_true",
                        help="render cached outputs without running chapter cells")
    args = parser.parse_args()
    cli = BOOK / "node_modules/jupyter-book/py/jupyter_book/dist/jupyter-book.cjs"
    if not cli.is_file():
        parser.error(
            "run npm ci --ignore-scripts --no-bin-links --cache .cache/npm in user_guide/ first"
        )
    node = shutil.which("node")
    if node is None:
        parser.error("Node.js is required by Jupyter Book 2")
    command = [node, "--require", str(BOOK / "one-kernel.cjs"), str(cli),
               "start" if args.action == "preview" else "build"]
    if args.action == "html":
        command += ["--html"]
    if args.action != "preview":
        command += ["--strict"]
    if not args.no_execute:
        command += ["--execute", "--execute-parallel", "1"]
    subprocess.run(command, cwd=BOOK, env=build_environment(), check=True)


if __name__ == "__main__":
    main()
