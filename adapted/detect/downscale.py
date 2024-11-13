import numpy as np


def efficient_average_pooling(data: np.ndarray, pool_size: int = 2) -> np.ndarray:
    """
    Efficient average pooling for downscaling a signal.
    Takes a 2D array of shape (n_samples, n_features) and downscales it by a factor of pool_size
    by averaging over non-overlapping blocks of size pool_size.

    Args:
        data (np.ndarray): The data to downscale.
        pool_size (int): The size of the pooling window.

    Returns:
        np.ndarray: The downscaled data.
    """
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")

    n_samples, n_features = data.shape
    if n_features % pool_size != 0:
        # If odd, pad the data with an additional column of zeros
        padded_data = np.pad(
            data, ((0, 0), (0, pool_size - n_features % pool_size)), mode="constant"
        )
        n_features += pool_size - n_features % pool_size
    else:
        padded_data = data

    pooled_data = padded_data.reshape(
        n_samples, n_features // pool_size, pool_size
    ).mean(axis=2)

    return pooled_data


def downscale_signal(signal: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Downscale a signal by a given factor.
    """
    return efficient_average_pooling(signal, pool_size=factor)
