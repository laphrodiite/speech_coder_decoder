import numpy as np
from scipy.signal import lfilter

def RPE_frame_st_decoder(LARc: np.ndarray, curr_frame_st_resd: np.ndarray) -> np.ndarray:
    """
    Decode the short-term residual and LAR coefficients to reconstruct the original frame signal.

    Args:
        LARc (np.ndarray): Array of 8 quantized Log-Area Ratios.
        curr_frame_st_resd (np.ndarray): The short-term residual signal (160 samples).

    Returns:
        np.ndarray: Reconstructed signal (160 samples).
    """
    # Convert LARc to reflection coefficients (r) - placeholder for provided utility
    r = convert_larc_to_reflection(LARc)

    # Convert reflection coefficients to predictor coefficients (ak)
    ak = convert_reflection_to_predictor(r)

    # Short-Term Synthesis Filtering
    # Hs(z) = 1 / (1 - ∑ ak * z^(-k)) -> Convolution in the time domain
    s0 = lfilter([1], np.concatenate(([1], -ak)), curr_frame_st_resd)

    return s0


def convert_larc_to_reflection(LARc: np.ndarray) -> np.ndarray:
    """
    Converts quantized Log-Area Ratios (LAR) to reflection coefficients (r).

    Args:
        LARc (np.ndarray): Array of 8 quantized Log-Area Ratios.

    Returns:
        np.ndarray: Array of 8 reflection coefficients.
    """
    r = np.zeros_like(LARc)

    # Apply transformations as per ETSI GSM 06.10 spec
    for i in range(8):
        if abs(LARc[i]) < 0.675:
            r[i] = LARc[i]
        elif abs(LARc[i]) < 1.225:
            r[i] = np.sign(LARc[i]) * (0.5 * abs(LARc[i]) + 0.3375)
        else:
            r[i] = np.sign(LARc[i]) * (0.125 * abs(LARc[i]) + 0.796875)

    return r


def convert_reflection_to_predictor(r: np.ndarray) -> np.ndarray:
    """
    Converts reflection coefficients (r) to predictor coefficients (a_k)
    using the Levinson-Durbin algorithm.

    Args:
        r (np.ndarray): Array of 8 reflection coefficients.

    Returns:
        np.ndarray: Array of 8 predictor coefficients (a_k).
    """
    order = len(r)
    a = np.zeros(order + 1)  # Predictor coefficients (a_k)
    a[0] = 1  # By definition, a_0 = 1

    for i in range(1, order + 1):
        a[i] = r[i - 1]
        for j in range(1, i):
            a[j] -= r[i - 1] * a[i - j]

    return a[1:]
