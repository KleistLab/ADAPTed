"""
ADAPTed (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import logging
import os
import re
from typing import List


def validate_filename(
    filename: str,
    endswiths: List[str] = [],
    basenameprefix: str = "",
    raise_: bool = True,
) -> bool:
    check1, check2, check3, check4 = True, True, True, True
    if not os.path.exists(filename):
        check1 = False
        if raise_:
            msg = f"The provided file {filename} does not exist."
            logging.error(msg)
            raise ValueError(msg)
    if os.path.isdir(filename):
        check2 = False
        if raise_:
            msg = "The provided file should not be a directory."
            logging.error(msg)
            raise ValueError(msg)
    if not any([filename.endswith(endswith) for endswith in endswiths]):
        check3 = False
        if raise_:
            msg = "The provided file should have one of the following extensions: {}".format(
                endswiths
            )
            logging.error(msg)
            raise ValueError(msg)

    if basenameprefix is not None:
        if not os.path.basename(filename).startswith(basenameprefix):
            check4 = False
            if raise_:
                msg = "The provided file should have the following basename: {}".format(
                    basenameprefix
                )
                logging.error(msg)
                raise ValueError(msg)

    return all([check1, check2, check3, check4])


def get_valid_files(
    basedir: str, endswiths: List[str] = [], basenameprefix: str = ""
) -> List[str]:
    if not len(endswiths) and basenameprefix is None:
        msg = "Either `endswiths` or `basenameprefix` should be specified."
        logging.error(msg)
        raise ValueError(msg)

    valid_files = []
    # walk through directory structure searching for pod5 files

    basedir_slash = (
        basedir if basedir.endswith(os.sep) else os.path.join(basedir, "")
    )  # add trailing os.sep if not present

    for root, _, fns in os.walk(basedir_slash):
        for fn in fns:
            if validate_filename(
                os.path.join(root, fn),
                endswiths=endswiths,
                basenameprefix=basenameprefix,
                raise_=False,
            ):
                valid_files.append(os.path.join(root, fn))

    return lexsort_num_suffix(valid_files)


def lexsort_num_suffix(filenames: List[str]) -> List[str]:
    """
    Sorts a list of filenames lexicographically, taking into account numerical values,
    ensuring that "xx_10" comes after "xx_2", for example.

    Parameters:
    - filenames: A list of filenames to sort.

    Returns:
    - A new list of filenames sorted lexicographically with numerical ordering.
    """

    def parts(file):
        base, _ = os.path.splitext(file)
        match = re.search(r"(\d+)$", base)
        if match:
            return base[: match.start()], int(match.group())
        else:
            return base, float(0)  # Use zero to sort non-numeric suffixes first

    return sorted(filenames, key=parts)


def input_to_filelist(
    input: List[str],
    endswiths: List[str] = [".pod5"],
    basenameprefix: str = "",
) -> List[str]:

    files = []
    for path in input:
        if path == " ":
            continue

        if os.path.isdir(path):
            files.extend(
                get_valid_files(
                    path, endswiths=endswiths, basenameprefix=basenameprefix
                )
            )
        else:
            validate_filename(
                path, endswiths=endswiths, basenameprefix=basenameprefix, raise_=True
            )
            files.append(path)
    files = lexsort_num_suffix(files)

    return files


def construct_filename(
    path_to_dir: str = "", prefix: str = "", suffix: str = "", extension: str = ""
) -> str:
    """
    Generates a complete file path with the specified directory, prefix, suffix, and extension.

    Allows for the flexible construction of a filename with various components and ensures that
    the suffix is separated by an underscore for readability.

    Parameters:
    - path_to_dir (str, optional): The directory in which the file will be created. Defaults to the current working directory.
    - prefix (str, optional): The beginning part of the filename.
    - suffix (str, optional): The ending part of the filename. Preceded by an underscore if provided.
    - extension (str, optional): The file extension to be appended. Includes a dot ('.') if provided.

    Returns:
    - str: The fully constructed file path.
    """

    # Default to the current working directory if none is provided
    if path_to_dir == "":
        path_to_dir = os.getcwd()

    # Ensure the extension is prefixed with a dot
    extension = (
        f".{extension}" if extension and not extension.startswith(".") else extension
    )

    # Prefix the suffix with an underscore for separation if it is provided
    suffix = f"_{suffix}" if suffix else suffix

    # Construct and return the full file path
    return os.path.join(path_to_dir, f"{prefix}{suffix}{extension}")
