"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import concurrent.futures
import multiprocessing
import os
import signal
import sys
import time
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Manager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np
from tqdm import tqdm

from pod5 import Reader

from adapted.config.config import Config
from adapted.detect.combined import DetectResults


# Global flag to track if CTRL+C signal has been received
CTRL_C_RECEIVED = False


@dataclass
class ReadResult:
    read_id: Optional[str] = None
    success: bool = True
    fail_reason: Optional[str] = None

    detect_results: Optional[DetectResults] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        detect_dict = self.detect_results.to_dict() if self.detect_results else {}
        detect_dict.pop("fail_reason", None)
        return {
            "read_id": self.read_id,
            **detect_dict,
            "fail_reason": self.fail_reason,
        }


def signal_handler(sig, frame):
    global CTRL_C_RECEIVED
    for process in multiprocessing.active_children():
        process.terminate()
    logging.info("Ctrl+C detected. Stopping the script.")
    CTRL_C_RECEIVED = True


def load_signals_in_mem(
    file_obj: Reader,
    ids: List[str],
    max_obs: int = 40000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    signals = np.zeros((len(ids), max_obs), dtype=np.float32)
    in_arr_lengths = np.zeros(len(ids), dtype=np.int32)
    full_lengths = np.zeros(len(ids), dtype=np.int32)
    read_ids = []

    for i_id, read in enumerate(file_obj.reads(selection=ids)):
        s = np.array(read.signal_pa)  # calibrated signal (pA)
        l = min(s.size, max_obs)
        signals[i_id, :l] = s[:l]
        in_arr_lengths[i_id] = l
        full_lengths[i_id] = s.size
        read_ids.append(str(read.read_id))

    if not set(read_ids) == set(ids):
        set_diff = set(ids).symmetric_difference(set(read_ids))
        msg = "Read ids do not match. Check if the provided read ids are correct. "
        msg += f"Missing read ids: {set_diff}"
        logging.error(msg)
        raise ValueError(msg)

    return signals, in_arr_lengths, full_lengths, read_ids


def result_processor(
    shared_results,
    total,
    result_batch_size,
    result_fn,
    result_kwargs,
    bidx_passed=0,
    bidx_failed=0,
):
    pbar = tqdm(total=total, desc="Processing reads", file=sys.stdout)
    failed, passed = [], []
    n_seen, n_passed, n_failed = 0, 0, 0

    while n_seen < total:
        while len(shared_results):
            res = shared_results.pop(0)
            n_seen += 1
            pbar.update(1)
            # try:
            if res.success:
                passed.append(res)
                if len(passed) == result_batch_size:
                    result_fn(
                        "pass",
                        results=passed,
                        batch_idx=bidx_passed,
                        **result_kwargs,
                    )
                    n_passed += result_batch_size
                    bidx_passed += 1
                    passed = []
            else:
                failed.append(res)
                if len(failed) == result_batch_size:
                    result_fn(
                        "fail",
                        results=failed,
                        batch_idx=bidx_failed,
                        **result_kwargs,
                    )
                    n_failed += result_batch_size
                    bidx_failed += 1
                    failed = []
        time.sleep(0.1)  # Avoid busy waiting

    # Process remaining results
    if passed:
        result_fn("pass", results=passed, batch_idx=bidx_passed, **result_kwargs)
    if failed:
        result_fn("fail", results=failed, batch_idx=bidx_failed, **result_kwargs)

    pbar.close()
    logging.info(f"Read processing completed in {pbar.format_dict['elapsed']:.2f}s")

    n_passed += len(passed)
    n_failed += len(failed)
    divisor = n_passed + n_failed
    fraction = n_passed / divisor if divisor != 0 else 0
    logging.info(
        f"Successfully detected an adapter in {n_passed} / {divisor} reads ({fraction * 100:.2f}%)."
    )


def worker(
    filename: str,
    minibatch_ids: List[str],
    config: Config,
    task_fn: Callable,
    shared_results: List[ReadResult],
):
    try:
        with Reader(path=filename) as file_obj:
            signals, in_arr_lengths, full_sig_lengths, read_ids = load_signals_in_mem(
                file_obj,
                minibatch_ids,
                max_obs=config.sig_proc.sig_preload_size,
            )

        results = []
        for signal, signal_len, full_sig_len, read_id in zip(
            signals, in_arr_lengths, full_sig_lengths, read_ids
        ):
            try:
                result = task_fn(
                    signal=signal,
                    signal_len=signal_len,
                    full_sig_len=full_sig_len,
                    read_id=read_id,
                    spc=config.sig_proc,
                    **config.task.dict(),
                )
                results.append(result)
            except Exception as e:
                logging.error(
                    f"Error processing read {read_id}: {e}. Read will be considered failed."
                )
                results.append(
                    ReadResult(read_id=read_id, success=False, fail_reason=str(e))
                )

        shared_results.extend(results)

    except Exception as e:
        logging.error(f"Error in worker process: {e}")
        logging.error(traceback.format_exc())


def process_in_batches(
    file_read_id_map: Dict[str, List[str]],
    config: Config,
    task_fn: Callable,
    result_fn: Callable,
):
    max_concurrent_tasks = config.batch.num_proc or multiprocessing.cpu_count()

    with Manager() as manager:
        shared_results: List[ReadResult] = manager.list()  # type: ignore

        result_process = Process(
            target=result_processor,
            args=(
                shared_results,
                number_of_reads(file_read_id_map),
                config.batch.batch_size,
                result_fn,
                config.output.dict(),
                config.batch.bidx_passed,
                config.batch.bidx_failed,
            ),
        )
        result_process.start()

        logging.info("Starting read processing...")
        with ProcessPoolExecutor(max_concurrent_tasks) as executor:
            futures = []
            for filename, read_ids in file_read_id_map.items():
                for i in range(0, len(read_ids), config.batch.minibatch_size):
                    end_idx = min(i + config.batch.minibatch_size, len(read_ids))
                    minibatch_ids = read_ids[i:end_idx]
                    future = executor.submit(
                        worker,
                        filename,
                        minibatch_ids,
                        config,
                        task_fn,
                        shared_results,
                    )
                    futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Task failed: {e}")

                if CTRL_C_RECEIVED:
                    logging.info("Terminating due to CTRL+C")
                    break

        result_process.join(timeout=60)
        if result_process.is_alive():
            logging.error("Result processor did not finish in time. Terminating.")
            result_process.terminate()
        else:
            logging.info("All barcode fingerprints are saved.")


def number_of_reads(file_read_id_map: Dict[str, List[str]]) -> int:
    return sum(len(read_ids) for read_ids in file_read_id_map.values())


def get_file_read_id_map(config: Config) -> Dict[str, List[str]]:
    file_read_id_map = {}
    if len(config.input.read_ids) > 0:
        read_id_set = set(config.input.read_ids)
        n_requested_read_ids = len(read_id_set)
    else:
        read_id_set = set()
        n_requested_read_ids = None

    n_found_read_ids = 0
    for filename in config.input.files:
        try:
            file_obj = Reader(path=filename)
            file_read_id_list = file_obj.read_ids
            file_read_id_set = set(file_read_id_list)

            if n_requested_read_ids is None:
                file_read_id_map[filename] = file_read_id_list
            else:
                file_read_id_map[filename] = list(
                    read_id_set.intersection(file_read_id_set)
                )

            n_found_read_ids += len(file_read_id_map[filename])
        except Exception as e:
            print(f"Error reading in {filename}: {e}")

    if config.input.continue_from:
        processed_reads, max_pass_bidx, max_fail_bidx = scan_processed_reads(
            config.input.continue_from
        )

        for filename in file_read_id_map:
            file_read_id_map[filename] = [
                read_id
                for read_id in file_read_id_map[filename]
                if read_id not in processed_reads
            ]
        n_found_read_ids -= len(processed_reads)
        config.batch.bidx_passed = max_pass_bidx + 1
        config.batch.bidx_failed = max_fail_bidx + 1

    if config.input.continue_from:
        processed_reads, max_pass_bidx, max_fail_bidx = scan_processed_reads(
            config.input.continue_from
        )

        for filename in file_read_id_map:
            file_read_id_map[filename] = [
                read_id
                for read_id in file_read_id_map[filename]
                if read_id not in processed_reads
            ]
        n_found_read_ids -= len(processed_reads)
        config.batch.bidx_passed = max_pass_bidx + 1
        config.batch.bidx_failed = max_fail_bidx + 1

    remaining_str = "remaining" if config.input.continue_from else ""

    requested_str = (
        f"out of {n_requested_read_ids} requested"
        if n_requested_read_ids is not None
        else ""
    )
    logging.info(f"Found {n_found_read_ids} {remaining_str} read_ids {requested_str}")

    return file_read_id_map


def scan_processed_reads(continue_from_path: str) -> Tuple[Set[str], int, int]:
    processed_reads = set()
    max_pass_bidx = -1
    max_fail_bidx = -1

    # Scan detected_boundaries files
    for file in os.listdir(continue_from_path):
        if file.startswith("detected_boundaries_") and file.endswith(".csv"):
            bidx = int(file.split("_")[-1].split(".")[0])
            max_pass_bidx = max(max_pass_bidx, bidx)
            with open(os.path.join(continue_from_path, file), "r") as f:
                processed_reads.update(line.split(",")[0] for line in f.readlines()[1:])

    # Scan failed_reads files
    for file in os.listdir(continue_from_path):
        if file.startswith("failed_reads_") and file.endswith(".csv"):
            bidx = int(file.split("_")[-1].split(".")[0])
            max_fail_bidx = max(max_fail_bidx, bidx)
            with open(os.path.join(continue_from_path, file), "r") as f:
                processed_reads.update(line.split(",")[0] for line in f.readlines()[1:])
    return processed_reads, max_pass_bidx, max_fail_bidx


def process(
    file_read_id_map: Dict[str, List[str]],
    config: Config,
    task_fn: Callable,
    results_fn: Callable,
) -> None:
    # catch ctrl+c
    signal.signal(signal.SIGINT, signal_handler)

    try:
        process_in_batches(
            file_read_id_map=file_read_id_map,
            config=config,
            task_fn=task_fn,
            result_fn=results_fn,
        )

    except:
        logging.error("Failed to process files")
        logging.error(traceback.format_exc())
        return
