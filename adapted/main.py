"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import logging
import os
import sys
import time

from adapted.file_proc import handle_previous_results, run_detect
from adapted.logger import setup_logger
from adapted.parser import parse_args


def main(args=None):

    config = parse_args()
    setup_logger(os.path.join(config.output.output_dir, "adapted.log"))

    logging.info(f"Command: {' '.join(sys.argv)}")

    logging.info(f"Saving output to: {config.output.output_dir}")

    print_files = (
        config.input.files[: min(3, len(config.input.files))]
        + [f"..."]
        + config.input.files[-min(3, len(config.input.files)) :]
        if len(config.input.files) > 3
        else config.input.files
    )
    print_files_str = "\n".join(print_files)
    logging.info(f"Input filenames:\n{print_files_str}")
    logging.info(f"Total number of input files: {len(config.input.files)}")

    # report config
    logging.info("SigProcConfig:")
    config.sig_proc.pretty_print(file=logging.getLogger().handlers[0].stream)  # type: ignore

    # Preprocess input_read_ids into batches
    os.makedirs(config.output.output_dir, exist_ok=True)

    read_ids_excl = set()
    read_ids_incl = set()
    if config.input.continue_from:

        # Preprocess input_read_ids into batches
        logging.info(f"Indexing previous results...")
        start_time = time.time()

        read_ids_excl = handle_previous_results(config)
        logging.info(f"Indexing took: {time.time() - start_time:.2f} seconds")
        logging.info(f"Found {len(read_ids_excl)} previously processed reads.")

    files = set(config.input.files)
    read_ids_incl = set(config.input.read_ids)
    config.input.files = (
        []
    )  # clear to prevent long lists being copied around to all processes
    config.input.read_ids = (
        []
    )  # clear to prevent long lists being copied around to all processes

    logging.info(f"Config.output: {config.output.dict() }")
    logging.info(f"Config.batch: {config.batch.dict() }")

    # save spc that were used
    config.sig_proc.to_toml(os.path.join(config.output.output_dir, "config.toml"))

    run_detect(
        files=files,
        read_ids_incl=read_ids_incl,
        read_ids_excl=read_ids_excl,
        config=config,
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()
