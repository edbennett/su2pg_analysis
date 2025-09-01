#!/usr/bin/env python3

"""
Computation of the mean plaquette for an ensemble.
"""

from argparse import ArgumentParser

import pyerrors as pe

from .io import name_ensemble
from .parsers import read_pg


def get_args():
    """
    Parse command-line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("pg_log", help="Pure gauge log file")
    parser.add_argument(
        "--output_filename",
        default=None,
        help="Where to output results",
    )
    return parser.parse_args()


def main():
    """
    Compute the mean plaquette and output
    """
    args = get_args()
    metadata, data = read_pg(args.pg_log)
    plaquette = pe.Obs(
        [data["plaquette"]],
        [name_ensemble(metadata)],
        idl=[data["trajectory"]],
    )
    plaquette.tag = "plaquette"
    plaquette.gamma_method()
    if args.output_filename is None:
        print(plaquette)
    else:
        pe.input.json.dump_to_json(
            plaquette,
            args.output_filename,
            description=metadata,
        )


if __name__ == "__main__":
    main()
