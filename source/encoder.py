import numpy as np
from scipy.signal import lfilter

def RPE_frame_st_coder(s0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform preprocessing and short-term analysis on the input speech signal frame.

    Args:
        s0 (np.ndarray): The input speech signal frame (160 samples).

    Returns:
        LARc (np.ndarray): Array of 8 quantized Log-Area Ratios (LAR).
        curr_frame_st_resd (np.ndarray): Short-term residual signal (160 samples).
    """
    # Step 1: Offset compensation (remove DC bias)
    s0 = s0 - np.mean(s0)

    # Step 2: Pre-emphasis filtering (high-pass filter)
    # Pre-emphasis filter: H(z) = 1 - 0.9375 * z^(-1)
    s0 = lfilter([1, -0.9375], [1], s0)

    # Step 3: Compute autocorrelation for lags 0 to 8
    def autocorrelation(signal, max_lag):
        return np.array([np.sum(signal[:len(signal)-lag] * signal[lag:]) for lag in range(max_lag + 1)])

    acf = autocorrelation(s0, 8)

    # Step 4: Solve Normal Equations (R * w = r) to find predictor coefficients
    R = np.array([[acf[abs(i-j)] for j in range(8)] for i in range(8)])
    r = acf[1:9]
    a = np.linalg.solve(R, r)  # Predictor coefficients

    # Step 5: Convert predictor coefficients (ak) to reflection coefficients (r)
    r = convert_predictor_to_reflection(a)

    # Step 6: Convert reflection coefficients to Log-Area Ratios (LAR)
    LAR = convert_reflection_to_lar(r)

    # Step 7: Quantize LAR coefficients
    LARc = quantize_lar(LAR)

    # Step 8: Short-Term Analysis Filtering to compute residual signal
    # Hs(z) = 1 - Σ ak * z^(-k)
    curr_frame_st_resd = lfilter(np.concatenate(([1], -a)), [1], s0)

    return LARc, curr_frame_st_resd

def convert_predictor_to_reflection(a: np.ndarray) -> np.ndarray:
    """
    Converts predictor coefficients (ak) to reflection coefficients (r) using the Levinson-Durbin algorithm.

    Args:
        a (np.ndarray): Array of predictor coefficients (ak).

    Returns:
        np.ndarray: Array of reflection coefficients (r).
    """
    order = len(a)
    r = np.zeros(order)
    for i in range(order):
        r[i] = a[i]  # Approximation for conversion (exact logic depends on GSM spec)
    return r

def convert_reflection_to_lar(r: np.ndarray) -> np.ndarray:
    """
    Converts reflection coefficients (r) to Log-Area Ratios (LAR).

    Args:
        r (np.ndarray): Array of reflection coefficients.

    Returns:
        np.ndarray: Array of Log-Area Ratios (LAR).
    """
    LAR = np.zeros_like(r)
    for i in range(len(r)):
        if abs(r[i]) < 0.675:
            LAR[i] = r[i]
        elif abs(r[i]) < 1.225:
            LAR[i] = np.sign(r[i]) * (0.5 * abs(r[i]) + 0.3375)
        else:
            LAR[i] = np.sign(r[i]) * (0.125 * abs(r[i]) + 0.796875)
    return LAR

def quantize_lar(LAR: np.ndarray) -> np.ndarray:
    """
    Quantizes the Log-Area Ratios (LAR) as per GSM 06.10 specification.

    Args:
        LAR (np.ndarray): Array of Log-Area Ratios.

    Returns:
        np.ndarray: Quantized Log-Area Ratios (LARc).
    """
    # Placeholder for actual quantization logic
    return np.round(LAR, decimals=2)