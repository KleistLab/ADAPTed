"""
WarpDemuX

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import concurrent.futures
import inspect
import logging
import multiprocessing
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from typing import Callable, Dict, Generator, List, Set, Tuple, Union

import numpy as np
import pandas as pd
from adapted.config.config import Config
from adapted.container_types import ReadResult
from adapted.detect.cnn import load_cnn_model
from adapted.detect.combined import combined_detect_cnn, combined_detect_llr2
from adapted.output import save_detected_boundaries
from pod5.reader import Reader
from tqdm import tqdm

_STOP_SIGNAL = threading.Event()

# Initialize the shared batch index and a threading lock
_BIDX_LOCK_PASS = threading.Lock()
_BIDX_LOCK_FAIL = threading.Lock()

_BIDX_PASS = 0
_BIDX_FAIL = 0

_TOTAL_READS_LOCK = threading.Lock()
_TOTAL_READS = -1


def set_global_var(name, value):
    if name == "total_reads":
        with _TOTAL_READS_LOCK:
            global _TOTAL_READS
            _TOTAL_READS = value
    elif name == "bidx_pass":
        with _BIDX_LOCK_PASS:
            global _BIDX_PASS
            _BIDX_PASS = value
    elif name == "bidx_fail":
        with _BIDX_LOCK_FAIL:
            global _BIDX_FAIL
            _BIDX_FAIL = value
    else:
        raise ValueError(f"Invalid name: {name}")


def get_global_var(name: str) -> int:
    if name == "total_reads":
        with _TOTAL_READS_LOCK:
            return _TOTAL_READS
    elif name == "bidx_pass":
        with _BIDX_LOCK_PASS:
            return _BIDX_PASS
    elif name == "bidx_fail":
        with _BIDX_LOCK_FAIL:
            return _BIDX_FAIL
    else:
        raise ValueError(f"Invalid name: {name}")


def increment_ridx(ridx_dict, ridx_name, ridx_lock, value=1):
    with ridx_lock:
        ridx_dict[ridx_name] += value


def increment_bidx(bidx_dict, bidx_name, bidx_lock, value=1):
    with bidx_lock:
        bidx_dict[bidx_name] += value


def bidx_name_from_ridx_name(ridx_name: str) -> str:
    if "pass" in ridx_name:
        return "pass"
    elif "fail" in ridx_name:
        return "fail"
    else:
        raise ValueError(f"Invalid ridx_name: {ridx_name}")


def determine_bidx_from_file(file: str) -> int:
    return int(file.split("_")[-1].split(".")[0])


def determine_max_bidx_from_dir(dir_path: str) -> int:

    return max(determine_bidx_from_file(f) for f in os.listdir(dir_path))


def scan_processed_reads(
    config: Config, failed_only: bool = False
) -> Tuple[Set[str], int, int]:
    processed_reads = set()
    max_pass_bidx = -1
    max_fail_bidx = -1

    fail_sub = os.path.join(config.output.output_dir_fail)
    for file in os.listdir(fail_sub):
        if file.startswith("failed_reads_") and file.endswith(".csv"):
            bidx = int(file.split("_")[-1].split(".")[0])
            max_fail_bidx = max(max_fail_bidx, bidx)
            with open(os.path.join(fail_sub, file), "r") as f:
                processed_reads.update(line.split(",")[0] for line in f.readlines()[1:])

    if not failed_only:
        pass_sub = os.path.join(config.output.output_dir_boundaries)
        for file in os.listdir(pass_sub):
            if file.startswith("detected_boundaries_") and file.endswith(".csv"):
                bidx = int(file.split("_")[-1].split(".")[0])
                max_pass_bidx = max(max_pass_bidx, bidx)
                with open(os.path.join(pass_sub, file), "r") as f:
                    processed_reads.update(
                        line.split(",")[0] for line in f.readlines()[1:]
                    )
    return processed_reads, max_pass_bidx, max_fail_bidx


def handle_previous_results(config: Config, failed_only: bool = False) -> Set[str]:
    processed_reads, max_pass_bidx, max_fail_bidx = scan_processed_reads(
        config, failed_only
    )
    config.batch.bidx_pass = max_pass_bidx + 1
    config.batch.bidx_fail = max_fail_bidx + 1
    return processed_reads


def yield_signals_from_pod5(
    pod5_files: Union[List[str], Set[str]],
    read_ids_incl: Set[str],
    read_ids_excl: Set[str],
    batch_size: int,
    preload_size: int,
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    if read_ids_incl and read_ids_excl:
        read_ids_incl = read_ids_incl.difference(read_ids_excl)
        read_ids_excl = set()

    selection = list(read_ids_incl) if read_ids_incl else None

    N = batch_size
    m = preload_size
    i_id = 0

    signals = np.empty((N, m), dtype=np.float32)
    full_lengths = np.empty(N, dtype=np.int32)
    read_ids = np.empty(N, dtype=object)

    for filename in pod5_files:
        with Reader(filename) as file_obj:
            for read_record in file_obj.reads(selection=selection, missing_ok=True):
                if str(read_record.read_id) in read_ids_excl:
                    continue

                _m = min(m, read_record.num_samples)
                full_lengths[i_id] = read_record.num_samples
                signals[i_id, :_m] = read_record.signal_pa[:_m]
                if _m < m:
                    signals[i_id, _m:] = np.nan
                read_ids[i_id] = str(read_record.read_id)

                if i_id == N - 1:
                    yield signals, full_lengths, read_ids
                    signals = np.empty(
                        (N, m),
                        dtype=np.float32,
                    )
                    full_lengths = np.empty(N, dtype=np.int32)
                    read_ids = np.empty(N, dtype=object)
                    i_id = 0
                else:
                    i_id += 1

    if i_id > 0:
        yield signals[:i_id], full_lengths[:i_id], read_ids[:i_id]


def worker_enqueue_minibatches(
    files: Set[str],
    read_ids_incl: Set[str],
    read_ids_excl: Set[str],
    config: Config,
    preloaded_minibatch_queue: multiprocessing.Queue,
    ridx_dict: dict,
    ridx_lock: multiprocessing.Lock,  # type: ignore
):

    for minibatch in yield_signals_from_pod5(
        pod5_files=files,
        read_ids_incl=read_ids_incl,
        read_ids_excl=read_ids_excl,
        batch_size=config.batch.minibatch_size,
        preload_size=config.sig_proc.sig_preload_size,
    ):
        preloaded_minibatch_queue.put(minibatch)  # has maxsize
        increment_ridx(ridx_dict, "enqueued", ridx_lock, len(minibatch[0]))

    preloaded_minibatch_queue.put(None)  # signal end of work
    logging.debug("Worker enqueue_minibatches finished")


def worker_detect_on_preloaded_signals(
    preloaded_minibatch,
    model_detect,
    config,
    ridx_dict,
    ridx_lock,
    save_pass_queue,
    save_fail_queue,
):
    success = []
    fail = []
    signals, full_lengths, read_ids = preloaded_minibatch

    if config.sig_proc.primary_method == "llr":
        detect_results = combined_detect_llr2(
            batch_of_signals=signals,
            full_signal_lens=full_lengths,
            spc=config.sig_proc,
        )
    else:
        detect_results = combined_detect_cnn(
            batch_of_signals=signals,
            full_signal_lens=full_lengths,
            model=model_detect,
            spc=config.sig_proc,
        )
    del model_detect
    assert isinstance(detect_results, list)

    for detect_result, read_id in zip(detect_results, read_ids):
        read_result = ReadResult(
            read_id=read_id,
            success=detect_result.success,
            fail_reason=detect_result.fail_reason,
            detect_results=detect_result,
        )
        if read_result.success:
            success.append(read_result)
        else:
            fail.append(read_result)

    del preloaded_minibatch, signals, full_lengths, read_ids

    if fail:
        save_fail_queue.put(fail)
        increment_ridx(ridx_dict, "done_fail", ridx_lock, len(fail))

    if success:
        save_pass_queue.put(success)
        increment_ridx(ridx_dict, "done_pass", ridx_lock, len(success))


def _queue_batch_processor_df(
    queue, process_fn, ridx_dict, ridx_name, ridx_lock, bidx_dict, bidx_lock, config
):
    """Collects pd.DataFrame results of variable length from the queue and processes them in batches of batch_size rows."""
    batch: List[pd.DataFrame] = []
    batch_read_count = 0

    while True:
        # Wait for results from the queue
        result = queue.get()
        if result is None:
            break
        if not isinstance(result, pd.DataFrame):
            raise ValueError(f"Invalid result type: {type(result)}")
        batch.append(result)
        batch_read_count += len(result)

        # If the batch reaches the desired size, process it
        if batch_read_count >= config.batch.batch_size_output:
            current_batch = pd.concat(batch, axis=0)
            process_fn(
                current_batch.iloc[: config.batch.batch_size_output].copy(),
                config=config,
                bidx_dict=bidx_dict,
                bidx_lock=bidx_lock,
            )
            batch = [current_batch.iloc[config.batch.batch_size_output :]]
            batch_read_count -= config.batch.batch_size_output
            increment_ridx(
                ridx_dict, ridx_name, ridx_lock, config.batch.batch_size_output
            )
            increment_bidx(bidx_dict, bidx_name_from_ridx_name(ridx_name), bidx_lock, 1)

    # save remainder
    if batch_read_count > 0:
        current_batch = pd.concat(batch, axis=0)
        process_fn(
            current_batch, config=config, bidx_dict=bidx_dict, bidx_lock=bidx_lock
        )
        increment_ridx(ridx_dict, ridx_name, ridx_lock, batch_read_count)
        increment_bidx(bidx_dict, bidx_name_from_ridx_name(ridx_name), bidx_lock, 1)


def _queue_batch_processor_list(
    queue, process_fn, ridx_dict, ridx_name, ridx_lock, bidx_dict, bidx_lock, config
):
    """Collects ReadResult results from the queue and processes them in batches."""
    batch: List[ReadResult] = []
    batch_read_count = 0

    while True:
        # Wait for results from the queue
        result = queue.get()

        if result is None:
            break

        if not isinstance(result, list):
            raise ValueError(f"Invalid result type: {type(result)}")

        batch.extend(result)
        batch_read_count += len(result)

        if batch_read_count >= config.batch.batch_size_output:
            process_fn(
                batch[: config.batch.batch_size_output],
                config=config,
                bidx_dict=bidx_dict,
                bidx_lock=bidx_lock,
            )
            batch = batch[config.batch.batch_size_output :]
            batch_read_count -= config.batch.batch_size_output
            increment_ridx(
                ridx_dict, ridx_name, ridx_lock, config.batch.batch_size_output
            )
            increment_bidx(bidx_dict, bidx_name_from_ridx_name(ridx_name), bidx_lock, 1)

    # save remainder
    if batch_read_count > 0:
        process_fn(batch, config=config, bidx_dict=bidx_dict, bidx_lock=bidx_lock)
        increment_ridx(ridx_dict, ridx_name, ridx_lock, batch_read_count)
        increment_bidx(bidx_dict, bidx_name_from_ridx_name(ridx_name), bidx_lock, 1)


def queue_batch_processor(
    queue: multiprocessing.Queue,
    process_fn: Callable,
    ridx_dict: Dict[str, int],
    ridx_name: str,
    ridx_lock: multiprocessing.Lock,  # type: ignore
    bidx_dict: Dict[str, int],
    bidx_lock: multiprocessing.Lock,  # type: ignore
    config: Config,
):
    """Collects results from the queue and processes them in batches."""

    # Infer the type based on the process_fn signature
    fn_sig = inspect.signature(process_fn)
    first_param_type = fn_sig.parameters[next(iter(fn_sig.parameters))].annotation

    try:
        if first_param_type == pd.DataFrame:
            _queue_batch_processor_df(
                queue,
                process_fn,
                ridx_dict,
                ridx_name,
                ridx_lock,
                bidx_dict,
                bidx_lock,
                config,
            )
        elif first_param_type == List[ReadResult]:
            _queue_batch_processor_list(
                queue,
                process_fn,
                ridx_dict,
                ridx_name,
                ridx_lock,
                bidx_dict,
                bidx_lock,
                config,
            )
        else:
            raise ValueError(
                f"Unsupported process_fn first argument type: {first_param_type}"
            )
    except Exception as e:
        logging.error(f"Error in queue_batch_processor: {e}")
        logging.error(traceback.format_exc())
        raise e


def save_batch_outputs_pass(
    read_results: List[ReadResult], bidx_dict, bidx_lock, config: Config
):
    with bidx_lock:
        bidx = bidx_dict["pass"]
    save_detect_batch(
        "pass",
        results=read_results,
        batch_idx=bidx,
        output_dir_boundaries=config.output.output_dir_boundaries,
    )
    del read_results


def save_batch_outputs_fail(
    read_results: List[ReadResult], bidx_dict, bidx_lock, config: Config
):
    with bidx_lock:
        bidx = bidx_dict["fail"]
    save_detect_batch(
        "fail",
        results=read_results,
        batch_idx=bidx,
        output_dir_fail=config.output.output_dir_fail,
    )
    del read_results


def save_detect_batch(
    pass_or_fail: str,
    results: List[ReadResult],
    batch_idx: int,
    output_dir_boundaries: str = "",
    output_dir_fail: str = "",
    **kwargs,
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    """Save detected boundaries and return read_ids and signals if pass_or_fail == 'pass'."""

    if pass_or_fail == "pass":
        fn = "detected_boundaries"
        dirn = output_dir_boundaries
    elif pass_or_fail == "fail":
        fn = "failed_reads"
        dirn = output_dir_fail
    else:
        msg = f"Invalid pass_or_fail: {pass_or_fail}. Must be 'pass' or 'fail'."
        logging.error(msg)
        raise ValueError(msg)

    save_detected_boundaries(
        results,  # type: ignore
        os.path.join(dirn, f"{fn}_{batch_idx}.csv"),
        save_fail_reasons=pass_or_fail == "fail",
    )

    return None


def worker_progress_report(ridx_dict, ridx_lock):
    total_reads = get_global_var("total_reads")
    total_reads_set = total_reads != -1

    # Create progress bars for total, failed, and passed reads
    pbar_total = tqdm(
        desc="Total progress",
        unit="reads",
        position=0,
        total=total_reads if total_reads_set else None,
    )
    pbar_fail = tqdm(
        desc="Failed reads",
        position=1,
        total=total_reads if total_reads_set else None,
        bar_format="{desc}",  # Custom format without speed
    )
    pbar_pass = tqdm(
        desc="Passed reads",
        position=2,
        total=total_reads if total_reads_set else None,
        bar_format="{desc}",  # Custom format without speed
    )

    last_fail = 0
    last_pass = 0
    time_passed = 0

    while not _STOP_SIGNAL.is_set():
        if time_passed >= 1:  # update every 1 seconds
            if not total_reads_set:
                total_reads = get_global_var("total_reads")
                total_reads_set = total_reads != -1
                if total_reads_set:
                    for pbar in [pbar_total, pbar_fail, pbar_pass]:
                        pbar.total = total_reads

            with ridx_lock:
                n_fail = ridx_dict["done_fail"]
                n_pass = ridx_dict["done_pass"]
                n_total = n_fail + n_pass

                new_fail = n_fail - last_fail
                new_pass = n_pass - last_pass
                new_total = new_fail + new_pass

                # Update progress bars
                if new_fail > 0:
                    pbar_fail.update(new_fail)
                if new_pass > 0:
                    pbar_pass.update(new_pass)
                pbar_total.update(new_total)

                # Update descriptions with percentages
                if n_total > 0:
                    fail_pct = (n_fail / n_total) * 100
                    pass_pct = (n_pass / n_total) * 100
                    pbar_fail.set_description_str(
                        f"Failed reads   {n_fail:,} | {fail_pct:.1f}%"
                    )
                    pbar_pass.set_description_str(
                        f"Passed reads   {n_pass:,} | {pass_pct:.1f}%"
                    )

                last_fail = n_fail
                last_pass = n_pass

            time_passed = 0
        time.sleep(0.1)
        time_passed += 0.1

    # Final update
    with ridx_lock:
        n_fail = ridx_dict["done_fail"]
        n_pass = ridx_dict["done_pass"]
        n_total = n_fail + n_pass

        # Update progress bars one last time
        pbar_fail.update(n_fail - last_fail)
        pbar_pass.update(n_pass - last_pass)
        pbar_total.update((n_fail - last_fail) + (n_pass - last_pass))

        if n_total > 0:
            fail_pct = (n_fail / n_total) * 100
            pass_pct = (n_pass / n_total) * 100
            pbar_fail.set_description_str(
                f"Failed reads   {n_fail:,} | {fail_pct:.1f}%"
            )
            pbar_pass.set_description_str(
                f"Passed reads   {n_pass:,} | {pass_pct:.1f}%"
            )

    # Close all progress bars
    pbar_total.close()
    pbar_fail.close()
    pbar_pass.close()


def check_total_num_reads(pod5_files: Set[str]) -> int:
    """Count total number of reads across all POD5 files using pod5 inspect."""

    total_reads = 0

    pod5_exec = os.path.join(sys.exec_prefix, "bin", "pod5")
    logging.debug(f"pod5_exec: {pod5_exec}")

    for file in pod5_files:
        try:
            # Run pod5 inspect command and capture output
            result = subprocess.run(
                [pod5_exec, "inspect", "summary", file],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running pod5 inspect on {file}: {e}")
            raise

        try:
            # Get the last line of output
            last_line = result.stdout.strip().split("\n")[-1]

            # Extract number of reads using regex
            match = re.search(r"Found\s+\d+\s+batches,\s+(\d+)\s+reads", last_line)
            if match:
                reads = int(match.group(1))
                total_reads += reads
            else:
                logging.warning(f"Could not parse read count from summary for {file}")

        except subprocess.CalledProcessError as e:
            logging.error(f"Error parsing pod5 inspect output on {file}: {e}")
            raise

    return total_reads


def set_total_num_reads(
    pod5_files: Set[str], read_ids_incl: Set[str], read_ids_excl: Set[str]
) -> None:
    if read_ids_incl:
        total_reads = len(read_ids_incl)
    else:
        total_reads_pod5 = check_total_num_reads(pod5_files)
        total_reads = total_reads_pod5 - len(read_ids_excl)
    set_global_var("total_reads", total_reads)
    logging.debug(f"total_reads: {total_reads}")

    # logging.info(f"Total number of reads that will be processed: {total_reads}")


def run_detect(
    files: Set[str], read_ids_incl: Set[str], read_ids_excl: Set[str], config: Config
) -> Dict[str, int]:
    """enqueue_minibatches_thread takes care of preloading the signals in the queue.
    Maxsize of queue is set to num_proc. Detect is the most time consuming step, we want to limit the signal preloading to be in sync with the speed of detect.
    """

    save_output_pass = True
    save_pass_thread = None

    # set global variables for batch indices for continue mode
    set_global_var("bidx_pass", config.batch.bidx_pass)
    set_global_var("bidx_fail", config.batch.bidx_fail)

    logging.debug(f"starting mp manager")
    with multiprocessing.Manager() as manager:

        if config.sig_proc.primary_method == "cnn":
            model_detect = load_cnn_model(config.sig_proc.cnn_boundaries.model_name)
        else:
            model_detect = None

        preloaded_minibatch_queue = manager.Queue(maxsize=config.batch.num_proc)
        save_pass_queue = manager.Queue()
        save_fail_queue = manager.Queue()

        ridx_lock = manager.Lock()
        bidx_lock = manager.Lock()

        # Create a shared dictionary that will be shared across processes
        ridx_dict = manager.dict()
        ridx_dict["enqueued"] = 0
        ridx_dict["done_pass"] = 0
        ridx_dict["done_fail"] = 0
        ridx_dict["saved_pass"] = 0
        ridx_dict["saved_fail"] = 0

        bidx_dict = manager.dict()
        bidx_dict["pass"] = get_global_var("bidx_pass")
        bidx_dict["fail"] = get_global_var("bidx_fail")

        # puts None into the preloaded_minibatch_queue when having preloaded all minibatches
        # and then returns
        # None is the sentinel value that allows the worker processes to exit
        total_reads_thread = threading.Thread(
            target=set_total_num_reads,
            args=(files, read_ids_incl, read_ids_excl),
        )

        enqueue_minibatches_thread = threading.Thread(
            target=worker_enqueue_minibatches,
            args=(
                files,
                read_ids_incl,
                read_ids_excl,
                config,
                preloaded_minibatch_queue,
                ridx_dict,
                ridx_lock,
            ),
        )

        save_pass_thread = (
            threading.Thread(
                target=queue_batch_processor,
                args=(
                    save_pass_queue,
                    save_batch_outputs_pass,
                    ridx_dict,
                    "saved_pass",
                    ridx_lock,
                    bidx_dict,
                    bidx_lock,
                    config,
                ),
            )
            if save_output_pass
            else None
        )

        save_fail_thread = threading.Thread(
            target=queue_batch_processor,
            args=(
                save_fail_queue,
                save_batch_outputs_fail,
                ridx_dict,
                "saved_fail",
                ridx_lock,
                bidx_dict,
                bidx_lock,
                config,
            ),
        )

        progress_bar_thread = threading.Thread(
            target=worker_progress_report,
            args=(
                ridx_dict,
                ridx_lock,
            ),
        )

        logging.debug(f"all threads created")

        total_reads_thread.start()

        if save_output_pass and save_pass_thread:
            save_pass_thread.start()
        save_fail_thread.start()
        enqueue_minibatches_thread.start()

        progress_bar_thread.start()
        logging.debug("all threads started")

        def handle_completed_future(future):
            try:
                future.result()  # This will raise any exceptions that occurred
            except Exception as e:
                logging.error(f"Error in worker: {e}")
                logging.error(traceback.format_exc())

        # submit jobs to the process pool as minibatches are preloaded
        # the job is finished when the processed signal and predictions have been added to the save queues
        # not done logic is used to manage the number of concurrent jobs and the preloaded queue
        # when a job is started, it takes from the preloaded queue
        # submitting all jobs at once would require a large memory footprint for the preloaded queue
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.batch.num_proc,
        ) as process_pool:
            not_done_futures = set()
            not_done_count = 0

            all_enqueued = False

            while not all_enqueued:
                # First, check for completed futures
                done_futures, not_done_futures = concurrent.futures.wait(
                    not_done_futures,
                    timeout=0.001,  #  prevent busy waiting
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                # Handle completed futures
                for future in done_futures:
                    handle_completed_future(future)
                    not_done_count -= 1

                # Then try to submit new work if we have capacity
                while not_done_count < 1.1 * config.batch.num_proc:
                    try:
                        minibatch = preloaded_minibatch_queue.get()
                    except queue.Empty:
                        if not not_done_futures:  # If no pending work and queue empty
                            all_enqueued = True
                            break
                        continue

                    if minibatch is None:
                        all_enqueued = True
                        break  # all minibatches have been submitted to the worker pool

                    future = process_pool.submit(
                        worker_detect_on_preloaded_signals,
                        minibatch,
                        model_detect,
                        config,
                        ridx_dict,
                        ridx_lock,
                        save_pass_queue,
                        save_fail_queue,
                    )
                    not_done_futures.add(future)
                    not_done_count += 1

        logging.debug("All detect workers have been launched")
        enqueue_minibatches_thread.join()
        logging.debug("Enqueue minibatches thread finished")

        # Wait for all remaining futures to complete
        concurrent.futures.wait(not_done_futures)

        # Handle any remaining completed futures
        for future in not_done_futures:
            handle_completed_future(future)

        logging.debug("All detect workers have finished")

        # signal the save threads that no more minibatches will be added to the queues
        save_pass_queue.put(None)
        save_fail_queue.put(None)

        logging.debug("Signalled save threads to finish")

        logging.debug("Waiting for save threads to finish")

        total_reads_thread.join()
        if save_output_pass and save_pass_thread:
            save_pass_thread.join()
        save_fail_thread.join()

        logging.debug("Save threads finished")

        _STOP_SIGNAL.set()
        progress_bar_thread.join()
        logging.debug("Progress bar thread finished")

        if ridx_dict["done_pass"] > 0:
            logging.info(
                f"Adapter was successfully detected in {ridx_dict['done_pass']} / { ridx_dict['done_fail'] + ridx_dict['done_pass']} reads  ({ridx_dict['done_pass'] / ridx_dict['enqueued'] * 100:.2f}%)."
            )
        else:
            logging.info(f"No detections were made.")

        return dict(ridx_dict)
