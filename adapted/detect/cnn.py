import importlib.resources as pkg_resources
from typing import List

import numpy as np
import torch
import torch.nn as nn
from adapted import models
from adapted.config.sig_proc import CNNBoundariesConfig, CoreConfig
from adapted.container_types import Boundaries
from adapted.detect.downscale import downscale_signal
from scipy.signal import find_peaks

SCORE_EXCL = -5.0


class BoundariesCNN(nn.Sequential):
    """
    CNN for detecting poly(A) boundaries.
    """

    def __init__(self, channels=64, kernel_size=7):
        super(BoundariesCNN, self).__init__(
            nn.Conv1d(
                1,
                channels,
                kernel_size=kernel_size,
                stride=kernel_size // 2,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                channels,
                2,
                kernel_size=kernel_size,
                stride=kernel_size // 2,
                padding=kernel_size // 2,
            ),
        )


def load_cnn_model(path: str) -> BoundariesCNN:
    model = BoundariesCNN()
    try:
        model.load_state_dict(torch.load(path, weights_only=True))
    except FileNotFoundError:

        try:
            with pkg_resources.path(models, path) as repo_path:
                model.load_state_dict(torch.load(repo_path, weights_only=True))
        except FileNotFoundError:
            raise FileNotFoundError(f"Model weights not found at {path}")

    return model


def prepare_data(batch_of_signals: np.ndarray, core_params: CoreConfig) -> torch.Tensor:

    # clip signal?
    # keep track of clipped locations and original values to use in open pore detection later?

    downscaled = downscale_signal(
        batch_of_signals[:, core_params.min_obs_adapter :],
        core_params.downscale_factor,
    )
    med = np.nanmedian(downscaled, axis=-1, keepdims=True)
    mad = np.nanmedian(np.abs(downscaled - med), axis=-1, keepdims=True)

    return torch.tensor((downscaled - med) / mad).nan_to_num(SCORE_EXCL)[:, None, :]


def cnn_score(
    batch_of_prepared_signals: torch.Tensor, model: BoundariesCNN
) -> torch.Tensor:

    # check that model weights were loaded
    if model.state_dict() == {}:
        raise ValueError("Model weights were not loaded")

    # check that model is in evaluation mode
    # model.eval()

    scores = model(batch_of_prepared_signals)

    return scores


def cnn_predict(
    batch_of_prepared_signals: torch.Tensor,
    model: BoundariesCNN,
    params: CNNBoundariesConfig,
    core_params: CoreConfig,
) -> np.ndarray:
    """
    Predict adapter and poly(A) ends using a CNN.

    First, the CNN scores are computed for the adapter and poly(A) ends.
    Then, the scores are post-processed to predict the adapter and poly(A) ends.
    Poly(A) end positions are predicted after setting all positions before the predicted adapter end + min_obs_polya to -10.
    """

    scores = cnn_score(batch_of_prepared_signals, model).detach().numpy()

    adapter_end_pos = np.argmax(
        scores[
            :,
            0,
            : (core_params.max_obs_adapter - core_params.min_obs_adapter)
            // core_params.downscale_factor,
        ],
        axis=1,
    )
    k = params.polya_cand_k
    if k >= 1:
        # set all positions before adapter end to SCORE_EXCL
        mask = np.arange(scores.shape[2])[None, :] < (adapter_end_pos)[:, None]
        scores[:, 1, :][mask] = SCORE_EXCL

        polyA_end_pos = np.argmax(scores[:, 1, :], axis=1)

    else:
        polyA_end_pos = np.full(scores.shape[0], 0)
    if k > 1:
        mask = np.arange(scores.shape[2]) > (polyA_end_pos)[:, None]
        scores[:, 1, :][mask] = SCORE_EXCL
        # flatten scores works because of flanking SCORE_EXCL regions
        polyA_end_candidates, _ = find_peaks(scores[:, 1, :].flatten(), distance=5)
        heights = scores[:, 1, :].flatten()[polyA_end_candidates]
        read_idx = polyA_end_candidates // scores.shape[2]
        order = np.lexsort(
            (-heights, read_idx)
        )  # sort within read, then by descending height
        polyA_end_candidates = polyA_end_candidates[order]

        # find indices where i switches to the next value
        switches = np.where(np.diff(read_idx) != 0)[0]
        polyA_end_candidates = np.split(
            np.mod(polyA_end_candidates, scores.shape[2]), switches + 1
        )

        padded_polyA_end_pos = np.zeros((scores.shape[0], k), dtype=np.int64)
        for i, peaks in enumerate(polyA_end_candidates):
            padded_polyA_end_pos[i, : len(peaks)] = peaks[
                :k
            ]  # Pad or truncate to ensure consistent shape

        return np.column_stack((adapter_end_pos[:, None], padded_polyA_end_pos))

    return np.column_stack((adapter_end_pos, polyA_end_pos))


def cnn_detect(
    batch_of_signals: np.ndarray,
    model: BoundariesCNN,
    params: CNNBoundariesConfig,
    core_params: CoreConfig,
) -> np.ndarray:

    prepared_data = prepare_data(batch_of_signals, core_params)
    preds = (
        cnn_predict(prepared_data, model, params, core_params)
        * core_params.downscale_factor
        + core_params.min_obs_adapter
    ).astype(int)
    # where predict was zero, set back to zero
    preds[preds == core_params.min_obs_adapter] = 0

    del prepared_data
    return preds


def cnn_detect_boundaries(
    batch_of_signals: np.ndarray,
    model: BoundariesCNN,
    params: CNNBoundariesConfig,
    core_params: CoreConfig,
) -> List[Boundaries]:
    preds = cnn_detect(batch_of_signals, model, params, core_params)

    return [
        Boundaries(
            adapter_start=0,
            adapter_end=pred[0],
            polya_end=pred[1],
            polya_end_topk=pred[1:],
        )
        for pred in preds
    ]
