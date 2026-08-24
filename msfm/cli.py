"""Command-line entry point for MSFM applications."""

from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Sequence


LEGACY_APPS = (
    "grid_postprocessing",
    "onthefly_postprocessing",
    "single_postprocessing",
    "power_spectra_noise",
    "fiducial_postprocessing",
    "peaks",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser without importing any optional app dependencies."""
    parser = argparse.ArgumentParser(prog="msfm", description="Multiprobe Simulation Forward Model applications")
    subparsers = parser.add_subparsers(dest="app", required=True, metavar="APP")

    for app in LEGACY_APPS:
        # The legacy applications own their argument parsers.  Keeping their
        # arguments as a remainder also ensures that ``APP --help`` displays
        # the original application's help rather than a duplicate maintained
        # here.
        legacy_parser = subparsers.add_parser(app, add_help=False, help=f"run run_{app}.py")
        legacy_parser.add_argument("app_args", nargs=argparse.REMAINDER)

    calccorrs_parser = subparsers.add_parser(
        "calccorrs", help="calculate correlation functions and write a WebDataset"
    )
    calccorrs_parser.add_argument("config", help="training configuration YAML file")
    calccorrs_parser.add_argument(
        "--output-path",
        default="corrs-%06d.tar",
        help="output tar path or printf-style shard pattern (default: %(default)s)",
    )
    calccorrs_parser.add_argument(
        "--num-examples", type=int, default=100, help="maximum number of examples to process (default: %(default)s)"
    )
    calccorrs_parser.add_argument(
        "--num-batches-per-file",
        type=int,
        default=10,
        help="number of batches in each WebDataset shard (default: %(default)s)",
    )
    calccorrs_parser.add_argument("--device", help="PyTorch device, for example 'cpu' or 'cuda'")
    return parser


def _run_legacy_app(app: str, app_args: Sequence[str]) -> None:
    """Execute a legacy app exactly as if its ``run_*.py`` file was invoked."""
    module_name = f"msfm.apps.run_{app}"
    previous_argv = sys.argv
    try:
        sys.argv = [module_name, *app_args]
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = previous_argv


def main(argv: Sequence[str] | None = None) -> None:
    """Parse ``argv`` and run the selected MSFM application."""
    command_line = list(sys.argv[1:] if argv is None else argv)
    # Do not parse legacy arguments here: some of them deliberately accept
    # options belonging to esub, and argparse cannot retain the ordering of
    # unknown options when mixed with positional values.
    if command_line and command_line[0] in LEGACY_APPS:
        _run_legacy_app(command_line[0], command_line[1:])
        return

    args = build_parser().parse_args(command_line)
    # Importing calccorrs loads its numerical and ML dependencies, so defer it
    # until this application is actually selected.
    from msfm.apps.calccorrs import calccorrs

    calccorrs(
        args.config,
        output_path=args.output_path,
        num_examples=args.num_examples,
        num_batches_per_file=args.num_batches_per_file,
        device=args.device,
    )


if __name__ == "__main__":
    main()
