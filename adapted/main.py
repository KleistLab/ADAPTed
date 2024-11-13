"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import logging
import os
import sys
import time

from adapted.detect.cnn import load_cnn_model
from adapted.file_proc.file_proc import get_file_read_id_map, process
from adapted.file_proc.tasks import (
    process_preloaded_signal_combined_detect_cnn,
    process_preloaded_signal_combined_detect_llr,
    save_results_batch,
)
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
    logging.info(f"Indexing read IDs...")
    start_time = time.time()

    file_read_id_map = get_file_read_id_map(config)
    logging.info(f"Indexing took: {time.time() - start_time:.2f} seconds")

    config.input.files = []  # no longer needed, save space
    config.input.read_ids = []  # no longer needed, save space

    os.makedirs(config.output.output_dir, exist_ok=True)

    # save spc that were used
    config.sig_proc.to_toml(os.path.join(config.output.output_dir, "config.toml"))

    if config.sig_proc.primary_method == "cnn":
        task_fn = process_preloaded_signal_combined_detect_cnn
        task_kwargs = {
            "model": load_cnn_model(config.sig_proc.cnn_boundaries.model_name)
        }
    else:
        task_fn = process_preloaded_signal_combined_detect_llr
        task_kwargs = {}

    process(
        file_read_id_map=file_read_id_map,
        config=config,
        task_fn=task_fn,
        task_kwargs=task_kwargs,
        results_fn=save_results_batch,
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
