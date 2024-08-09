"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import concurrent.futures
import multiprocessing
import signal
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import zip_longest
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from tqdm import tqdm

from pod5 import Reader

from adapted.config.config import Config
from adapted.config.sig_proc import SigProcConfig
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
    print("Ctrl+C detected. Stopping the script.")
    CTRL_C_RECEIVED = True


def process_minibatch_of_preloaded_signal(
    signals: np.ndarray,
    in_arr_lengths: np.ndarray,
    full_sig_lengths: np.ndarray,
    read_ids: List[str],
    spc: SigProcConfig,
    task_fn: Callable,
    task_kwargs: Dict[str, Any] = {},
    output_queue=None,
) -> List[ReadResult]:
    results = []

    for signal, signal_len, full_sig_len, read_id in zip(
        signals, in_arr_lengths, full_sig_lengths, read_ids
    ):
        proc_res = task_fn(
            signal=signal,
            signal_len=signal_len,
            full_sig_len=full_sig_len,
            read_id=read_id,
            spc=spc,
            **task_kwargs,
        )
        if output_queue is not None:
            output_queue.put(proc_res)
        else:
            results.append(proc_res)

    return results


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
        print("Read ids do not match")
        set_diff = set(ids).symmetric_difference(set(read_ids))
        print(f"Missing read ids: {set_diff}")
        raise ValueError("Read ids do not match")

    return signals, in_arr_lengths, full_lengths, read_ids


def grouper(
    iterable: Iterable, n: int, fillvalue=None
) -> Generator[Tuple[Any], None, None]:
    """
    Collect data into fixed-length chunks or blocks.
    From https://docs.python.org/3/library/itertools.html#itertools-recipes
    """

    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)  # type: ignore


def result_processor(
    output_queue: multiprocessing.Queue,  # individual read results
    total: int,  # number of reads expected
    result_batch_size: int,
    result_fn: Callable,
    result_kwargs: Dict[str, Any] = {},
) -> None:
    try:
        pbar = tqdm(
            total=total,
            desc=f"Processing reads",
        )

        failed, passed = [], []
        bidx_failed = 0
        bidx_passed = 0
        while True:
            res = output_queue.get()
            if res is None:  # stop signal
                break
            pbar.update(1)

            if res.success:
                passed.append(res)
                if len(passed) == result_batch_size:
                    result_fn(
                        "pass", results=passed, batch_idx=bidx_passed, **result_kwargs
                    )
                    bidx_passed += 1
                    passed = []
            else:
                failed.append(res)
                if len(failed) == result_batch_size:
                    result_fn(
                        "fail", results=failed, batch_idx=bidx_failed, **result_kwargs
                    )
                    bidx_failed += 1
                    failed = []

        # cleanup
        if passed:
            result_fn("pass", results=passed, batch_idx=bidx_passed, **result_kwargs)
        if failed:
            result_fn("fail", results=failed, batch_idx=bidx_failed, **result_kwargs)

        # report
        n_passed = bidx_passed * result_batch_size + len(passed)
        n_failed = bidx_failed * result_batch_size + len(failed)

        divisor = n_passed + n_failed
        fraction = n_passed / divisor if divisor != 0 else 0

        pbar.close()
        print(
            f"Succesfully processed {n_passed} / {divisor} reads ({fraction * 100:.2f}%)."
        )
        return
    except:
        print("Failed to process results")
        traceback.print_exc()


def process_in_batches(
    file_read_id_map: Dict[str, List[str]],
    config: Config,
    task_fn: Callable,
    result_fn: Callable,
):
    max_concurrent_tasks = (
        multiprocessing.cpu_count()
        if config.batch.num_proc is None
        else config.batch.num_proc
    )

    n_minibatches = number_of_minibatches(file_read_id_map, config.batch.minibatch_size)

    data_loader = minibatch_generator(file_read_id_map, config.batch.minibatch_size)

    try:
        result_executor = ProcessPoolExecutor(1)
        with multiprocessing.Manager() as manager:
            output_queue = manager.Queue()  # type: ignore
            output_queue = (
                output_queue
            )  # type: multiprocessing.Queue[Union[ReadResult, None]]

            result_executor.submit(
                result_processor,
                output_queue=output_queue,
                total=number_of_reads(file_read_id_map),
                result_batch_size=config.batch.batch_size,
                result_fn=result_fn,
                result_kwargs=config.output.dict(),
            )

            task = partial(
                process_minibatch_of_preloaded_signal,
                output_queue=output_queue,
                spc=config.sig_proc,
                task_fn=task_fn,
                task_kwargs=config.task.dict(),
            )

            with ProcessPoolExecutor(max_concurrent_tasks) as executor:

                def submit_task(file_obj, batch_of_read_ids):
                    sig_data = load_signals_in_mem(
                        file_obj,
                        batch_of_read_ids,
                        max_obs=config.sig_proc.sig_preload_size,
                    )  # sig_data = signals, in_arr_lengths, full_lengths, read_ids

                    return executor.submit(
                        task,
                        *sig_data,
                    )

                # wait with preloading data untill a processor is free to limit memory use

                # Start the initial batch of tasks
                futures = set()
                tasks_submitted = min(max_concurrent_tasks, n_minibatches)
                for _ in range(tasks_submitted):
                    futures.add(submit_task(*next(data_loader)))

                while tasks_submitted < n_minibatches:
                    # Wait for at least one task to complete
                    _, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )  # futures now contains the not-done-yet tasks

                    # Submit new tasks as long as there are tasks left
                    while (
                        len(futures) < max_concurrent_tasks
                        and tasks_submitted < n_minibatches
                    ):
                        futures.add(submit_task(*next(data_loader)))
                        tasks_submitted += 1

                executor.shutdown(wait=True)
            output_queue.put(None)

        result_executor.shutdown(wait=True)  # wait till all are saved
    except:
        print(f"Failed to process batches")
        traceback.print_exc()
        return []


def minibatch_generator(
    file_read_id_map: Dict[str, List[str]],
    minibatch_size: int,
) -> Generator[Tuple[Reader, List[str]], None, None]:
    """
    Generator that yields batches of read_ids from the file_read_id_map.
    """
    for filename, read_id_list in file_read_id_map.items():
        file_obj = Reader(path=filename)

        for batch in grouper(read_id_list, minibatch_size):
            try:  # last minibatch might be smaller than minibatch_size and contain None
                none_idx = batch.index(None)
                batch = batch[:none_idx]
            except:  # no None
                pass

            yield file_obj, list(batch)


def number_of_reads(file_read_id_map: Dict[str, List[str]]) -> int:
    return sum(len(read_ids) for read_ids in file_read_id_map.values())


def number_of_minibatches(
    file_read_id_map: Dict[str, List[str]], minibatch_size: int
) -> int:
    return int(np.ceil(number_of_reads(file_read_id_map) / minibatch_size))


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

    requested_str = (
        f"out of {n_requested_read_ids} requested"
        if n_requested_read_ids is not None
        else ""
    )
    print(f"Found {n_found_read_ids} read_ids {requested_str}")

    return file_read_id_map


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

        print("Done.")

    except:
        print("Failed to process files")
        traceback.print_exc()
        return
