"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import os
import sys
import time

from adapted.file_proc.file_proc import get_file_read_id_map, process
from adapted.file_proc.tasks import process_preloaded_signal, save_results_batch
from adapted.parser import parse_args


def main(args=None):
    print("Command executed:")
    print(" ".join(sys.argv))

    config = parse_args()

    print("Saving output to:", config.output.output_dir)

    print_files = (
        config.input.files[: min(3, len(config.input.files))]
        + [f"..."]
        + config.input.files[-min(3, len(config.input.files)) :]
        if len(config.input.files) > 3
        else config.input.files
    )
    print_files_str = "\n".join(print_files)
    print(f"Input Filenames:\n{print_files_str}")
    print(f"Total files: {len(config.input.files)}")

    # report config
    print("SigProcConfig:")
    config.sig_proc.pretty_print()

    # Preprocess input_read_ids into batches
    print(f"Preprocessing read IDs for {len(config.input.files)} files")
    start_time = time.time()

    file_read_id_map = get_file_read_id_map(config)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    config.input.files = []  # no longer needed, save space
    config.input.read_ids = []  # no longer needed, save space

    os.makedirs(config.output.output_dir, exist_ok=True)

    # save spc that were used
    config.sig_proc.to_toml(os.path.join(config.output.output_dir, "config.toml"))

    process(
        file_read_id_map=file_read_id_map,
        config=config,
        task_fn=process_preloaded_signal,
        results_fn=save_results_batch,
    )


if __name__ == "__main__":
    main()
