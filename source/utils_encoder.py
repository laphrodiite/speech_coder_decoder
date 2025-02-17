import numpy as np
from utils import alpha, beta, DLB


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


def quantize_b(b: list) -> list:
    """
    Quantize the gain parameter (b) according to GSM 06.10 standard.

    :param b: The list of gain factors to quantize.
    :return: The quantized gain factors.
    """
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

    # Initialize the reconstructed residuals for the current frame
    curr_frame_st_resd = np.zeros(frame_len)

    # Iterate over subframes and compute estimated samples
    for j, (bj, Nj) in enumerate(zip(b, N)):
        # Calculate the start and end indices for the current subframe
        start = j * subframe_len
        end = start + subframe_len

        # Extract the current subframe from the double_frame
        subframe = double_frame[frame_len + start:frame_len + end]

        # Initialize the reconstructed residuals for the current subframe
        subframe_st_resd = np.zeros(subframe_len)

        # Compute the reconstructed residuals for the current subframe
        for n in range(subframe_len):
            # Compute the index for the delayed sample: n - Nj
            delayed_index = n - Nj

            # If the delayed index is within the current subframe, use the reconstructed residual.
            # Otherwise, use the previous frame's residual.
            if delayed_index >= 0:
                d_prev = subframe_st_resd[delayed_index]
            else:
                # Adjust negative indices to fetch from the end portion of the previous frame.
                d_prev = curr_frame_st_resd[frame_len + delayed_index]

            # Long term prediction:
            # d'_pred(n) = bj * d'(n - Nj)
            predicted = bj * d_prev

            # Reconstruct the short term residual for the current sample:
            # d'(n) = e(n) + d'_pred(n)
            subframe_st_resd[n] = subframe[n] - predicted

        # Update the reconstructed residuals for the current subframe
        curr_frame_st_resd[start:end] = subframe_st_resd

        # Compute the estimated samples for the current subframe
        estimated_samples.extend(subframe_st_resd)

    assert len(estimated_samples) == 160, f"Estimated samples are not 160 but {len(estimated_samples)} "
    # Compute error and short-term residual
    curr_frame_ex_full = double_frame[frame_len:] - estimated_samples
    curr_frame_st_resd = curr_frame_ex_full + estimated_samples

    # Validate the length of the reconstructed signal
    assert len(curr_frame_ex_full) == frame_len, "Reconstructed lt residual signal should match frame length."

    return np.array(curr_frame_ex_full), np.array(curr_frame_st_resd)