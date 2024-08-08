"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import argparse
import os
from argparse import RawTextHelpFormatter
from datetime import datetime

import pandas as pd

from adapted.config.base import load_nested_config_from_file
from adapted.config.config import Config
from adapted.config.file_proc import BatchConfig, InputConfig, OutputConfig, TaskConfig
from adapted.config.sig_proc import SigProcConfig
from adapted.io_utils import input_to_filelist


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(
    prog="adapted",
    description=(
        "***** ADAPTed: Adapter and poly(A) Detection And Profiling Tool *****\n"
        "ADAPTed is designed to pinpoint DNA adapter sequences in raw dRNA-seq signals. "
        "For command-specific help, run: `adapted <command> --help`."
    ),
    formatter_class=RawTextHelpFormatter,
)

parent_parser = argparse.ArgumentParser(add_help=False)

parent_parser.add_argument(
    "--input",
    type=str,
    nargs="+",
    help=(
        "Path(s) to pod5 file(s). If directory/directories, all .pod5 files in the"
        " paths are processed (non-recursive). "
    ),
)

parent_parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Path to where the run output folder should be created. Default is the current working directory.",
)


parent_parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to a valid configuration toml to use. Default is None.",
)

parent_parser.add_argument(
    "--read_id_csv",
    type=str,
    default=None,
    help=(
        "Path to a csv file containing read IDs to be processed. Should contain a"
        " 'read_id' column."
    ),
)
parent_parser.add_argument(
    "--read_id_csv_colname",
    type=str,
    default="read_id",
    help=(
        "Column name in 'read_id_csv' containing the read IDs to be processed. Defaults"
        " to 'read_id'. This argument is ignored if '--preprocessed' is set."
    ),
)

parent_parser.add_argument(
    "-p",
    "--num_proc",
    type=int,
    default=None,
    help=(
        "Number of num_proc to use for parallel processing. If not specified, all"
        " available cores will be used."
    ),
)

parent_parser.add_argument(
    "--batch_size",
    type=int,
    default=4000,
    help=("Number of reads per output file."),
)

parent_parser.add_argument(
    "--minibatch_size",
    type=int,
    default=50,
    help=(
        "Number of reads per worker. These reads are loaded into memory prior to"
        " processing. Choose depending on the max_obs_adapter value and the amount"
        " of memory available."
    ),
)

parent_parser.add_argument(
    "--create_subdir",
    type=str2bool,
    default=True,
    help="Whether to create a subdirectory for the output. Default is True.",
)
subparsers = parser.add_subparsers(dest="mode", required=True)

# detect
parser_detect = subparsers.add_parser(
    "detect",
    parents=[parent_parser],
    description=(
        "Detect command: Identifies adapter sequences and exports a CSV of detected boundaries to `save_path`."
    ),
    formatter_class=RawTextHelpFormatter,
)
parser_detect.add_argument(
    "--save_llr_trace",
    type=str2bool,
    default=False,
    help="Save the LLR trace for each read.",
)


# # Future feature
# parser_trim = subparsers.add_parser(
#     "trim_BAM",
#     parents=[parent_parser],
#     description=(
#         "Trim BAM command: Processes raw signal files, detects adapters, and adjusts the tandem BAM file accordingly. "
#         "Outputs adapter sequences and trimmed reads as separate BAM files to `save_path`."
#     ),
#     formatter_class=RawTextHelpFormatter,
# )

# parser_trim.add_argument("BAM_file", type=str, help="Path to the BAM file to process.")


def parse_args() -> Config:
    args = parser.parse_args()

    read_ids = []
    if args.read_id_csv:
        read_ids = pd.read_csv(
            args.read_id_csv,
        )[args.read_id_csv_colname].values

    files = input_to_filelist(args.input)

    if len(files) == 0:
        print("No valid input files found.")
        print("Provided path: {}".format(args.input))
        exit(1)

    if args.output is None:
        args.output = os.getcwd()

    # create run dir
    if args.create_subdir:
        run_dir_name = "adapted_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(args.output, run_dir_name)
    else:
        run_dir = args.output
    os.makedirs(run_dir, exist_ok=True)

    input_config = InputConfig(
        files=files,
        read_ids=read_ids,
    )

    task_config = TaskConfig(
        llr_return_trace=args.save_llr_trace,
    )

    batch_config = BatchConfig(
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
    )

    output_config = OutputConfig(
        output_dir=run_dir,
        save_llr_trace=args.save_llr_trace,
    )

    if args.config:
        spc = load_nested_config_from_file(args.config, SigProcConfig)
    else:
        spc = SigProcConfig()  # TODO: default based on chemistry

    spc.update_sig_preload_size()

    return Config(
        input=input_config,
        output=output_config,
        batch=batch_config,
        task=task_config,
        sig_proc=spc,
    )
