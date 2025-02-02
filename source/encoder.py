import numpy as np
from scipy.signal import lfilter

from hw_utils import polynomial_coeff_to_reflection_coeff
from source.utils import A, B, min_values, max_values, decode_LARc_to_reflection

def preprocess_signal(s0: np.ndarray) -> np.ndarray:
    """
        Preprocessing steps according to GSM 06.10 standard
        s_0: The original input signal
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

def calculate_acf(signal: np.ndarray, lag=8) -> np.ndarray:
    """Calculate autocorrelation for a signal."""
    N = len(signal)
    acf = np.zeros(lag + 1)
    for k in range(lag + 1):
        acf[k] = np.sum(signal[k:] * signal[:N - k])
    return acf[0:lag + 1]

def convert_reflection_to_LAR(reflection_coefficients):
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

def quantize_LAR(LAR):
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

def RPE_subframe_slt_lte(d: np.ndarray, prev_d: np.ndarray) -> tuple[int, float]:
    """
        Computes the pitch period (N) and gain factor (b)
        using cross-correlation to find the best match.
    """
    min_lag, max_lag = len(d), len(prev_d)
    best_N, best_prev_window = min_lag, []
    max_correlation = -np.inf

    # Compute N using cross-correlation
    for N in range(min_lag, max_lag+1):
        d_past = prev_d[-N:min_lag-N] if min_lag != N else prev_d[-N:]
        R_N = np.sum(d * d_past)  # Compute R(λ)
        if R_N > max_correlation:
            max_correlation = R_N
            best_N, best_prev_window = N, d_past

    # Compute b
    numerator = np.sum(d * best_prev_window)
    denominator = np.sum(best_prev_window ** 2)
    b = numerator / denominator if denominator != 0 else 0

    return best_N, b

def RPE_frame_st_coder(s0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Preprocessing
    s = preprocess_signal(s0)

    # Calculating prediction coefficients through ACF
    prediction_coefficients = calculate_acf(s)

    # Converting to reflection coefficients
    reflection_coefficients = polynomial_coeff_to_reflection_coeff(prediction_coefficients)

    # Converting to LAR
    LAR = convert_reflection_to_LAR(reflection_coefficients)

    # Quantizing and encoding LAR
    LARc = quantize_LAR(LAR)

    # Decode LARc to modified reflection coefficients
    decoded_reflection_coefficients = decode_LARc_to_reflection(LARc)

    # Compute residual signal using modified coefficients
    fir_coefficients = np.concatenate(([1], -decoded_reflection_coefficients))
    curr_frame_st_resd = lfilter(fir_coefficients, [1], s)

    return LARc, curr_frame_st_resd

def quantize_b(b:list) -> list:
    """
        Quantize the gain parameter according to paragraph 3.1.15
    """
    from utils import DLB
    bc = []
    for bj in b:
        for i in range(len(DLB)):
            if bj <= DLB[i]:
                bc.append(i)
                break
    return bc

# TODO MAYBE SPLIT INTO 2D MATRICES TO AVOID DUPLICATE CODE REGARDING FRAME SPLITTING IN THE FOLLOWING 2 FUNCTIONS

def find_optimal_pitch_gain_subframes(double_frame: np.ndarray, num_subframes=4)\
        -> tuple[list, list]:
    """
        Get the optimal N and b for each subframe
    """
    # Parameter initialization
    frame_len = len(double_frame) // 2
    subframe_len = frame_len // num_subframes
    N, b = [], []

    # Loop through the concatenated frames and split into subframes
    for j in range(num_subframes):
        ind_min, ind_max = frame_len + j * subframe_len, frame_len + (j + 1) * subframe_len
        subframe = double_frame[ind_min:ind_max]
        previous_frame = double_frame[ind_min - 3 * subframe_len : ind_max - subframe_len]

        Nj, bj = RPE_subframe_slt_lte(subframe, previous_frame)
        N.append(Nj)
        b.append(bj)

    return N, b

def calculate_reconstructed_lt_resd(double_frame: np.ndarray, N:list, b:list, num_subframes=4)\
        -> tuple[np.ndarray, np.ndarray]:
    # Parameter initialization
    frame_len = len(double_frame) // 2
    subframe_len = frame_len // num_subframes
    estimated_samples = []

    # Loop through the concatenated frames and split into subframes
    for j, (bj, Nj) in enumerate(zip(b, N)):
        ind_min, ind_max = frame_len + j * subframe_len, frame_len + (j + 1) * subframe_len
        estimated_samples.extend(bj * double_frame[ind_min - Nj: ind_max - Nj])

    reconstructed_lt_resd = double_frame[frame_len:] - estimated_samples
    reconstructed_st_resd = reconstructed_lt_resd + estimated_samples
    assert len(reconstructed_lt_resd) == frame_len, "Reconstructed lr residual signal should match frame length."
    return np.array(reconstructed_lt_resd), np.array(reconstructed_st_resd)


def RPE_frame_slt_coder(s0: np.ndarray, prev_frame_st_resd: np.ndarray)\
        -> tuple[np.ndarray, list, list, np.ndarray, np.ndarray]:
    """
    both s0 and prev_frame_st_resd have len 160

    returns will be:
    LARc: np.ndarray,
    Nc: int,
    bc: int,
    curr_frame_ex_full: np.ndarray,
    curr_frame_st_resd: np.ndarray

    """
    LARc, current_frame_st_resd = RPE_frame_st_coder(s0)
    concatenated_frames = np.concatenate((prev_frame_st_resd, current_frame_st_resd))

    # Calculate and encode N, b values
    N, b = find_optimal_pitch_gain_subframes(concatenated_frames)
    Nc = N  # unnecessary but equation 3.13 says to do so ???
    bc = quantize_b(b)

    reconstructed_lt_resd, reconstructed_st_resd = calculate_reconstructed_lt_resd(concatenated_frames, Nc, bc)

    # don't bother me with return types
    array = np.zeros(1)
    return LARc, Nc, bc, array, array

