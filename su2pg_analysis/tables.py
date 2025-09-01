#!/usr/bin/env python3

"""
Shared tools for tabulation
"""


from argparse import ArgumentParser, FileType

from format_multiple_errors import format_column_errors
import pandas as pd

from .io import get_data
from .provenance import get_basic_metadata, text_metadata


def get_args(description):
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "data_filenames",
        metavar="datafile",
        help="Data files to read and plot",
        nargs="+",
    )
    parser.add_argument(
        "--output_filename",
        default=None,
        type=FileType("w"),
        help="Where to output the table",
    )
    return parser.parse_args()


def basic_table(table_callback, description=None):
    """
    Generate a simple table,
    where one set of data is read,
    and one table is generated.
    `table_callback` should generate a Pandas DataFrame
    """
    args = get_args(description)
    data = get_data(args.data_filenames)
    header, data_df = table_callback(data)
    content = data_df.to_latex(header=header, index=False)

    print(text_metadata(get_basic_metadata(), comment_char="%"), file=args.output_file)
    print(content, file=args.output_file)
