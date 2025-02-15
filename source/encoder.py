import numpy as np
from scipy.signal import lfilter
import struct

from hw_utils import polynomial_coeff_to_reflection_coeff
from utils import decode_LARc_to_reflection, compose_frame, alpha, beta


def preprocess_signal(s0: np.ndarray) -> np.ndarray:
    """
    Preprocess the input signal according to the GSM 06.10 standard.

    :param s0: The original input signal.
    :return: The preprocessed signal.
    """
    # Initialize offset-compensated signal & final pre-emphasized signal arrays
    s0f = np.zeros_like(s0)
    s = np.zeros_like(s0)

    # Offset compensation: s0f(k) = s0(k) - s0(k-1) + alpha * s0f(k-1)
    for k in range(1, len(s0)):
        s0f[k] = s0[k] - s0[k - 1] + alpha * s0f[k - 1]

    # Pre-emphasis filtering: s(k) = s0f(k) - beta * s0f(k-1)
    for k in range(1, len(s0f)):
        s[k] = s0f[k] - beta * s0f[k - 1]

    return s


def calculate_acf(signal: np.ndarray, lag: int = 8) -> np.ndarray:
    """
    Calculate the autocorrelation function (ACF) for a given signal.

    :param signal: The input signal.
    :param lag: The maximum lag for which ACF is calculated.
    :return: The autocorrelation values for lags 0 to `lag`.
    """
    N = len(signal)
    acf = np.zeros(lag + 1)

    # Compute ACF for each lag
    for k in range(lag + 1):
        acf[k] = np.sum(signal[k:] * signal[:N - k])

    return acf


def convert_reflection_to_LAR(reflection_coefficients: np.ndarray) -> np.ndarray:
    """
    Convert reflection coefficients to Log-Area Ratios (LAR) as per GSM 06.10 standard.

    :param reflection_coefficients: The reflection coefficients to convert.
    :return: The corresponding Log-Area Ratios (LAR).
    """
    assert np.max(reflection_coefficients) <= 1, 'Reflection coefficients should not be larger than 1'
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


def quantize_LAR(LAR: np.ndarray) -> np.ndarray:
    """
    Quantize and encode LAR values based on GSM 06.10 standard.

    :param LAR: The Log-Area Ratios to quantize.
    :return: The quantized and encoded LAR values.
    """
    # Import constant parameters
    from utils import A, B, min_values, max_values
    if len(LAR) != len(A):
        raise ValueError(f"LAR size ({len(LAR)}) does not match A/B size ({len(A)}).")

    LARc = []
    for i, lar in enumerate(LAR):
        quantized = int(A[i] * lar + B[i] + 0.5 * np.sign(A[i] * lar + B[i]))
        LARc.append(np.clip(quantized, min_values[i], max_values[i]))

    return np.array(LARc)


def RPE_subframe_slt_lte(d: np.ndarray, prev_d: np.ndarray) -> tuple[int, float]:
    """
    Compute the pitch period (N) and gain factor (b) using cross-correlation.

    :param d: The current subframe signal.
    :param prev_d: The previous frame signal.
    :return: A tuple containing the optimal pitch period (N) and gain factor (b).
    """
    min_lag, max_lag = len(d), len(prev_d)
    best_N, best_prev_window = min_lag, []
    max_correlation = -np.inf

    # Compute N using cross-correlation
    for N in range(min_lag, max_lag + 1):
        d_past = prev_d[-N:min_lag - N] if min_lag != N else prev_d[-N:]
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
    """
    Encode the current frame using short-term prediction.

    :param s0: The input signal frame.
    :return: A tuple containing the encoded LARc and the short-term residual signal.
    """
    # Preprocess the input signal
    s = preprocess_signal(s0)

    # Calculate prediction coefficients using ACF
    prediction_coefficients = calculate_acf(s)

    # Convert to reflection coefficients
    reflection_coefficients = polynomial_coeff_to_reflection_coeff(prediction_coefficients)

    # Convert to Log-Area Ratios (LAR)
    LAR = convert_reflection_to_LAR(reflection_coefficients)

    # Quantize and encode LAR values
    LARc = quantize_LAR(LAR)

    # Decode LARc to modified reflection coefficients
    decoded_reflection_coefficients = decode_LARc_to_reflection(LARc)

    # Compute residual signal using modified coefficients
    fir_coefficients = np.concatenate(([1], -decoded_reflection_coefficients))
    curr_frame_st_resd = lfilter(fir_coefficients, [1], s)

    return LARc, curr_frame_st_resd


def quantize_b(b: list) -> list:
    """
    Quantize the gain parameter (b) according to GSM 06.10 standard.

    :param b: The list of gain factors to quantize.
    :return: The quantized gain factors.
    """
    from utils import DLB
    bc = []
    for bj in b:
        if bj <= DLB[0]:
            bc.append(0)
        elif bj > DLB[2]:
            bc.append(3)
        else:
            for i in [1,2]:
                if DLB[i-1] < bj <= DLB[i]:
                    bc.append(i)
    return bc


def find_optimal_pitch_gain_subframes(double_frame: np.ndarray, num_subframes: int = 4) -> tuple[list, list]:
    """
    Find the optimal pitch delay (N) and gain factor (b) for each subframe.

    :param double_frame: The concatenated frame (previous + current frame).
    :param num_subframes: The number of subframes to process.
    :return: A tuple containing lists of optimal pitch delays (N) and gain factors (b).
    """
    frame_len = len(double_frame) // 2
    subframe_len = frame_len // num_subframes
    N, b = [], []

    for j in range(num_subframes):
        # Calculate index window for the subframe
        ind_min, ind_max = frame_len + j * subframe_len, frame_len + (j + 1) * subframe_len
        subframe = double_frame[ind_min:ind_max]
        previous_frame = double_frame[ind_min - 3 * subframe_len: ind_max - subframe_len]

        # Find optimal N and b for this subframe
        Nj, bj = RPE_subframe_slt_lte(subframe, previous_frame)
        N.append(Nj)
        b.append(bj)

    return N, b


def reconstruct_frame_residuals(double_frame: np.ndarray, N: list, b: list, num_subframes: int = 4) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct the excitation and short-term residual signals.

    :param double_frame: The concatenated frame (previous + current frame).
    :param N: The list of pitch delays for each subframe.
    :param b: The list of gain factors for each subframe.
    :param num_subframes: The number of subframes.
    :return: A tuple containing the reconstructed excitation and short-term residual signals.
    """
    frame_len = len(double_frame) // 2
    subframe_len = frame_len // num_subframes
    estimated_samples = []

    # Iterate over subframes and compute estimated samples
    for j, (bj, Nj) in enumerate(zip(b, N)):
        ind_min, ind_max = frame_len + j * subframe_len, frame_len + (j + 1) * subframe_len
        estimated_samples.extend(bj * double_frame[ind_min - Nj: ind_max - Nj])

    assert len(estimated_samples) == 160, f"Estimated samples are not 160 but {len(estimated_samples)} "
    # Compute error and short-term residual
    curr_frame_ex_full = double_frame[frame_len:] - estimated_samples
    curr_frame_st_resd = curr_frame_ex_full + estimated_samples

    # Validate the length of the reconstructed signal
    assert len(curr_frame_ex_full) == frame_len, "Reconstructed lt residual signal should match frame length."

    return np.array(curr_frame_ex_full), np.array(curr_frame_st_resd)


def RPE_frame_slt_coder(s0: np.ndarray, prev_frame_st_resd: np.ndarray) \
        -> tuple[np.ndarray, list, list, np.ndarray, np.ndarray]:
    """
    Encode the current frame using Regular Pulse Excitation (RPE) with long-term prediction (LTP).

    :param s0: The input signal frame.
    :param prev_frame_st_resd: The short-term residual of the previous frame.
    :return: A tuple containing:
             - LARc: The encoded Log-Area Ratios (LAR).
             - Nc: The optimal pitch delays for each subframe.
             - bc: The quantized gain factors for each subframe.
             - curr_frame_ex_full: The reconstructed excitation signal.
             - curr_frame_st_resd: The reconstructed short-term residual signal.
    """
    # Encode the current frame's short-term residual
    LARc, current_frame_st_resd = RPE_frame_st_coder(s0)

    # Concatenate previous and current frame residuals
    concat_frames_resd = np.concatenate((prev_frame_st_resd, current_frame_st_resd))

    # Find optimal pitch delay (N) and gain factor (b)
    N, b = find_optimal_pitch_gain_subframes(concat_frames_resd)

    # Quantize the factors
    Nc = N
    bc = quantize_b(b)

    # Reconstruct the excitation and short-term residual signals
    curr_frame_ex_full, curr_frame_st_resd = reconstruct_frame_residuals(concat_frames_resd, Nc, bc)

    return LARc, Nc, bc, curr_frame_ex_full, curr_frame_st_resd

# # --------------------------------- This is the thiiiiiiird paradoteo --------------------------------- #

# def RPE_frame_coder(s0: np.ndarray, prev_frame_resd: np.ndarray) -> tuple[str, np.ndarray]:
#     """
#     Encode the current frame and generate the bitstream.

#     :param s0: The input signal frame (160 samples).
#     :param prev_frame_resd: The short-term residual of the previous frame (160 samples).
#     :return: A tuple containing the bitstream (260 bits) and the current frame's short-term residual.
#     """
#     # Step 1: Encode the frame using short-term and long-term prediction
#     LARc, Nc, bc, curr_frame_ex_full, curr_frame_st_resd = RPE_frame_slt_coder(s0, prev_frame_resd)

#     # Step 2: Compose the bitstream
#     bitstream = ""

#     # Encode LARc (Log-Area Ratios)
#     bitstream += format(LARc[0], '06b')  # LARc[1] - 6 bits
#     bitstream += format(LARc[1], '06b')  # LARc[2] - 6 bits
#     bitstream += format(LARc[2], '05b')  # LARc[3] - 5 bits
#     bitstream += format(LARc[3], '05b')  # LARc[4] - 5 bits
#     bitstream += format(LARc[4], '04b')  # LARc[5] - 4 bits
#     bitstream += format(LARc[5], '04b')  # LARc[6] - 4 bits
#     bitstream += format(LARc[6], '03b')  # LARc[7] - 3 bits
#     bitstream += format(LARc[7], '03b')  # LARc[8] - 3 bits

#     # Encode subframe parameters
#     for i in range(4):  # 4 subframes
#         # Nc - 7 bits (LTP lag)
#         bitstream += format(Nc[i], '07b')
#         # bc - 2 bits (LTP gain)
#         bitstream += format(bc[i], '02b')
#         # xMc (curr_frame_ex_full) - 40 samples per subframe, 3 bits per sample
#         subframe_ex = curr_frame_ex_full[i * 40 : (i + 1) * 40]
#         for sample in subframe_ex:
#             bitstream += format(int(sample), '03b')  # Quantize to 3 bits

#     # Validate bitstream length
#     assert len(bitstream) == 260, f"Bitstream length is {len(bitstream)} instead of 260."

#     return bitstream, curr_frame_st_resd