"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import argparse
import json
import os
import shutil
import uuid
from argparse import RawTextHelpFormatter

import pandas as pd

from adapted._version import __version__
from adapted.config.base import load_nested_config_from_file
from adapted.config.config import Config
from adapted.config.file_proc import BatchConfig, InputConfig, OutputConfig
from adapted.config.sig_proc import SigProcConfig, get_chemistry_specific_config
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
        "ADAPTed is designed to pinpoint the location of DNA adapters and poly(A) signals in raw dRNA-seq signals. "
        "\n\n"
        "For command-specific help, run: `adapted <command> --help`."
    ),
    formatter_class=RawTextHelpFormatter,
)


subparsers = parser.add_subparsers(dest="mode", required=True)

parser_detect = subparsers.add_parser(
    "detect",
    help=("Detect adapter and poly(A) signal boundaries and calculate statistics."),
    formatter_class=RawTextHelpFormatter,
)


parser_continue = subparsers.add_parser(
    "continue",
    help="Continue processing from a previous incomplete run.",
    formatter_class=RawTextHelpFormatter,
)
parser_continue.add_argument(
    "continue_from",
    type=str,
    help=(
        "Path to a folder containing the results of a previous incomplete run. "
        "ADAPTed will load in the configuration and command used in the previous run, and "
        "continue from the last processed read."
    ),
)


# make performance group arguments only visible in the subparser
performance_group = parser_detect.add_argument_group("performance")
performance_group.add_argument(
    "-j",
    "--num_proc",
    type=int,
    default=None,
    help=(
        "Number of num_proc to use for parallel processing. If not specified, all"
        " available cores will be used."
    ),
)

performance_group.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=4000,
    help=("Number of reads per output file."),
)

performance_group.add_argument(
    "-s",
    "--minibatch_size",
    type=int,
    default=1000,
    help=(
        "Number of reads per worker. These reads are loaded into memory prior to"
        " processing. Choose depending on the max_obs_adapter value and the amount"
        " of memory available."
    ),
)

processing_group = parser_detect.add_argument_group("processing")
processing_group.add_argument(
    "-i",
    "--input",
    type=str,
    nargs="+",
    help=(
        "Path(s) to pod5 file(s). If directory/directories, all .pod5 files in the"
        " paths are processed (non-recursive). "
    ),
)

processing_group.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="Path to where the run output folder should be created. Default is the current working directory.",
)


processing_group.add_argument(
    "--config",
    type=str,
    help="Path to a valid configuration toml to use. See adapted/config/config_files/ for options.",
)

processing_group.add_argument(
    "-c",
    "--chemistry",
    type=str,
    choices=["RNA002", "RNA004"],
    help="Specify the chemistry to use. If provided, --config is not required and will be ignored if both are provided.",
)

processing_group.add_argument(
    "--max_obs_trace",
    type=int,
    default=None,
    help=(
        "Maximum number of observation to search for the adapter and poly(A) boundaries. "
        "Setting this argument overrides the default value in the configuration file. "
        "Use this with a large value for rerunning the detection of truncated reads. Default is None."
    ),
)

processing_group.add_argument(
    "--read_id_csv",
    type=str,
    default=None,
    help=(
        "Path to a csv file containing the (subset) of read IDs to be processed. Should contain a"
        " 'read_id' column."
    ),
)
processing_group.add_argument(
    "--read_id_csv_colname",
    type=str,
    default="read_id",
    help=(
        "Column name in 'read_id_csv' containing the read IDs to be processed. Defaults"
        " to 'read_id'. This argument is ignored if '--preprocessed' is set."
    ),
)

parser_detect._action_groups.reverse()  #'positional arguments', 'optional arguments', 'Performance', 'Processing'; reverse to get Processing first


def parse_args() -> Config:
    args = parser.parse_args()

    if args.mode == "continue":
        try:
            # load the parser arguments from the command.json file
            with open(os.path.join(args.continue_from, "command.json"), "r") as f:
                command_dict = json.load(f)
        except FileNotFoundError:
            parser.error(
                "No command.json file found in the continue_from directory. "
                "Please provide a valid continue-from directory."
            )

        # create a backup of the command.json file
        shutil.copy(
            os.path.join(args.continue_from, "command.json"),
            os.path.join(args.continue_from, "command_previous.json"),
        )

        # update args with the keys in command_dict that are not in args
        for key, value in command_dict.items():
            if key not in args.__dict__:
                args.__dict__[key] = value

        # update the args with the command_dict
        run_dir = args.continue_from
    else:
        args.output = args.output or os.getcwd()

        # create run dir
        run_dir_name = (
            "adapted_" + __version__.replace(".", "_") + "_" + str(uuid.uuid4())[:8]
        )
        run_dir = os.path.join(args.output, run_dir_name)

    if not args.config and not args.chemistry:
        parser.error("Either --config or --chemistry must be provided.")

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

    input_config = InputConfig(
        files=files,
        read_ids=read_ids,
        continue_from=args.continue_from if "continue_from" in args else "",
    )

    batch_config = BatchConfig(
        num_proc=args.num_proc,
        batch_size_output=args.batch_size,
        minibatch_size=args.minibatch_size,
    )

    output_config = OutputConfig(
        output_dir=run_dir,
    )

    if args.config:  # when config is provided, chemistry is ignored
        spc = load_nested_config_from_file(args.config, SigProcConfig)
    else:
        spc = get_chemistry_specific_config(
            chemistry=args.chemistry
        )  # Use chemistry if config is not provided

    if args.max_obs_trace:
        spc.core.max_obs_trace = args.max_obs_trace

    spc.update_primary_method()
    spc.update_sig_preload_size()

    # Create run_dir if it doesn't exist
    os.makedirs(run_dir, exist_ok=True)
    # Create command.json file
    command_dict = vars(args)
    command_json_path = os.path.join(run_dir, "command.json")
    with open(command_json_path, "w") as f:
        json.dump(command_dict, f, indent=2)

    return Config(
        input=input_config,
        output=output_config,
        batch=batch_config,
        sig_proc=spc,
    )
