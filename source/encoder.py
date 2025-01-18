import numpy as np
from scipy.signal import lfilter
from hw_utils import polynomial_coeff_to_reflection_coeff

def preprocess_signal(s0: np.ndarray) -> np.ndarray:
    """
        Preprocessing steps according to GSM 06.10 standard
        s_o: The original input signal
    """
    # Constants
    alpha = 32735 * (2 ** -15)  # Offset compensation parameter
    beta = 28180 * (2 ** -15)   # Pre-emphasis parameter

    # Initialize arrays for the processed signals
    s0f = np.zeros_like(s0)  # Offset-compensated signal
    s = np.zeros_like(s0)     # Final pre-emphasized signal

    # Offset compensation: s0f(k) = s0(k) - s0(k-1) + alpha * s0f(k-1)
    for k in range(1, len(s0)):
        s0f[k] = s0[k] - s0[k - 1] + alpha * s0f[k - 1]

    # Pre-emphasis filtering: s(k) = s0f(k) - beta * s0f(k-1)
    for k in range(1, len(s0f)):
        s[k] = s0f[k] - beta * s0f[k - 1]
    return s

def calculate_acf(signal: np.ndarray, lag: int) -> np.ndarray:
    """Calculate autocorrelation for a signal."""
    N = len(signal)
    acf = np.zeros(lag + 1)
    for k in range(lag + 1):
        acf[k] = np.sum(signal[k:] * signal[:N - k])
    return acf


def reflection_to_LAR(reflection_coefficients):
    """
        Convert reflection coefficients to Log-Area Ratios (LAR) 
        with the method described in GSM 06.10 standard
    """
    LAR = []
    for r in reflection_coefficients:
        abs_r = abs(r)
        if abs_r < 0.675:
            LAR.append(r)
        elif 0.675 <= abs_r < 0.950:
            LAR.append(np.sign(r) * (2 * abs_r - 0.675))
        else:  # 0.950 <= abs_r <= 1
            LAR.append(np.sign(r) * (8 * abs_r - 6.375))

    return np.array(LAR)


def quantize_LAR(LAR, A, B, min_values, max_values):
    """
        Quantize and encode LAR values based on table parameters
        with the method described in GSM 06.10 standard.
    """
    if len(LAR) != len(A):
        raise ValueError(f"LAR size ({len(LAR)}) does not match A/B size ({len(A)}).")
    LARc = []
    for i, lar in enumerate(LAR):
        quantized = int(A[i] * lar + B[i] + 0.5 * np.sign(A[i] * lar + B[i]))
        LARc.append(np.clip(quantized, min_values[i], max_values[i]))

    return np.array(LARc)


def decode_LARc_to_reflection(LARc: np.ndarray, A: list, B: list) -> np.ndarray:
    """
        Decode quantized LARc values back into reflection coefficients.
    """
    LAR = (LARc - np.array(B)) / np.array(A)
    reflection_coefficients = []
    for lar in LAR:
        abs_lar = abs(lar)
        if abs_lar < 0.675:
            reflection_coefficients.append(lar)
        elif 0.675 <= abs_lar < 1.35:
            reflection_coefficients.append(np.sign(lar) * (0.5 * (abs_lar + 0.675)))
        else:  # abs_lar >= 1.35
            reflection_coefficients.append(np.sign(lar) * ((abs_lar + 6.375) / 8))
    return np.array(reflection_coefficients)


def RPE_frame_st_coder(s0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    A = [20.0, 20.0, 20.0, 20.0, 13.637, 15.0, 8.334, 8.824]
    B = [0.0, 0.0, -5.0, -5.0, 0.184, -0.5, -0.666, -2.235]
    min_values = [-32, -32, -16, -16, -8, -8, -4, -4]
    max_values = [31, 31, 15, 15, 7, 7, 3, 3]
    
    # Preprocessing
    s = preprocess_signal(s0)

    # Calculating prediction coefficients through ACF
    lag = 8
    acf = calculate_acf(s, lag)
    prediction_coefficients = acf[0:lag + 1]

    # Converting to reflection coefficients
    reflection_coefficients = polynomial_coeff_to_reflection_coeff(prediction_coefficients)

    # Converting to LAR
    LAR = reflection_to_LAR(reflection_coefficients)

    # Quantizing and encoding LAR
    LARc = quantize_LAR(LAR, A, B, min_values, max_values)

    """
    !! Not sure about this chief !!
    """
    # Decode LARc to modified reflection coefficients
    decoded_reflection_coefficients = decode_LARc_to_reflection(LARc, A, B)

    # Step 8: Compute residual signal using modified coefficients
    fir_coefficients = np.concatenate(([1], -decoded_reflection_coefficients))
    curr_frame_st_resd = lfilter(fir_coefficients, [1], s)

    return LARc, curr_frame_st_resd
